from __future__ import annotations

import copy
import os
import random
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from glintiqa.configs.defaults import build_parser
from glintiqa.data import canonical_dataset_name, create_dataloader, create_iqa_dataset, dataset_indices
from glintiqa.engine.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from glintiqa.engine.evaluate import evaluate_iqa
from glintiqa.engine.train import move_batch_to_device
from glintiqa.models import create_iqa_model


def parse_args():
    parser = build_parser()
    parser.description = "Distributed GlintIQA pretraining on generated SAQT-IQA data."
    parser.set_defaults(dataset="generated_dataset")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--train-index-start", type=int, default=None)
    parser.add_argument("--train-index-end", type=int, default=None)
    parser.add_argument(
        "--per-gpu-batch-size",
        type=int,
        default=None,
        help="Batch size per GPU. Defaults to --batch-size divided by world size.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="Number of random SAQT-IQA indices used for validation. Set 0 to disable validation.",
    )
    parser.add_argument("--val-seed", type=int, default=None, help="Random seed for SAQT-IQA validation sampling.")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--test-dataset", default=None, help="Optional external test dataset, e.g. kadid-10k.")
    parser.add_argument("--test-dataset-root", type=Path, default=None)
    parser.add_argument("--test-label-path", type=Path, default=None)
    parser.add_argument("--test-similarity-csv", type=Path, default=None)
    parser.add_argument("--test-every", type=int, default=0, help="Run optional test every N epochs. 0 disables it.")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training.")
    return parser.parse_args()


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def resolve_indices(dataset_name: str, start: int | None = None, end: int | None = None) -> List[int]:
    dataset_name = canonical_dataset_name(dataset_name)
    try:
        indices = dataset_indices(dataset_name)
    except ValueError:
        if dataset_name == "generated_dataset":
            indices = dataset_indices("saqt-iqa")
        else:
            raise
    return indices[slice(start, end)]


def build_model(args):
    return create_iqa_model(
        args.model,
        arch=args.arch,
        img_size=args.img_size,
        patch_size=args.vit_patch_size,
        embed_dim=args.embed_dim,
    )


def split_saqt_train_val_indices(args) -> tuple[List[int], List[int]]:
    indices = resolve_indices("saqt-iqa", args.train_index_start, args.train_index_end)
    if args.val_size <= 0:
        return indices, []
    if args.val_size >= len(indices):
        raise ValueError(f"--val-size={args.val_size} must be smaller than available SAQT-IQA indices {len(indices)}.")

    rng = random.Random(args.val_seed if args.val_seed is not None else args.seed)
    val_indices = sorted(rng.sample(indices, args.val_size))
    val_set = set(val_indices)
    train_indices = [index for index in indices if index not in val_set]
    return train_indices, val_indices


def create_distributed_train_loader(args, rank: int, world_size: int, local_rank: int, train_indices: List[int]):
    dataset_name = canonical_dataset_name(args.dataset)
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for distributed training.")
    dataset = create_iqa_dataset(
        dataset_name=dataset_name,
        root=args.dataset_root,
        indices=train_indices,
        patch_size=args.patch_size,
        patch_num=args.train_patch_num,
        is_train=True,
        args=args,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    per_gpu_batch_size = args.per_gpu_batch_size
    if per_gpu_batch_size is None:
        per_gpu_batch_size = max(args.batch_size // max(world_size, 1), 1)
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if is_main_process(rank):
        print(
            f"Distributed train samples={len(dataset)} world_size={world_size} "
            f"local_rank={local_rank} per_gpu_batch_size={per_gpu_batch_size}"
        )
    return loader


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if distributed_is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def train_one_epoch_distributed(
    train_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    scaler,
    device,
    epoch: int,
    world_size: int,
    use_amp: bool,
):
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    losses = []
    for batch_idx, batch in enumerate(train_loader, start=1):
        images, labels, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            preds = model(images).flatten()
            loss = criterion(preds, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        reduced_loss = reduce_mean(loss.detach(), world_size)
        losses.append(float(reduced_loss.cpu()))
        if batch_idx % 50 == 0 and (not distributed_is_initialized() or dist.get_rank() == 0):
            print(
                f"Epoch {epoch:03d} step {batch_idx:05d}/{len(train_loader):05d} "
                f"loss={losses[-1]:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )
    return sum(losses) / max(len(losses), 1)


def evaluate_saqt_validation_on_main_process(model, args, device, val_indices: List[int]):
    if not val_indices:
        return None, None

    eval_args = copy.copy(args)
    dataset_name = "saqt-iqa"
    data_loader = create_dataloader(
        args=eval_args,
        dataset_name=dataset_name,
        root=args.dataset_root,
        indices=val_indices,
        patch_size=args.patch_size,
        patch_num=args.test_patch_num,
        is_train=False,
    )
    metrics, predictions, targets, filenames = evaluate_iqa(
        model=unwrap_model(model),
        data_loader=data_loader,
        device=device,
        patch_num=args.test_patch_num,
    )
    predictions_df = pd.DataFrame(
        {
            "filename": filenames,
            "predicted_score": predictions,
            "ground_truth": targets,
        }
    )
    return metrics, predictions_df


def evaluate_optional_test_on_main_process(model, args, device):
    if args.test_dataset is None:
        return None, None
    if args.test_dataset_root is None:
        raise ValueError("--test-dataset-root is required when --test-dataset is set.")

    test_args = copy.copy(args)
    if args.test_label_path is not None:
        test_args.label_path = args.test_label_path
    if args.test_similarity_csv is not None:
        test_args.similarity_csv = args.test_similarity_csv

    dataset_name = canonical_dataset_name(args.test_dataset)
    data_loader = create_dataloader(
        args=test_args,
        dataset_name=dataset_name,
        root=args.test_dataset_root,
        indices=resolve_indices(dataset_name),
        patch_size=args.patch_size,
        patch_num=args.test_patch_num,
        is_train=False,
    )
    metrics, predictions, targets, filenames = evaluate_iqa(
        model=unwrap_model(model),
        data_loader=data_loader,
        device=device,
        patch_num=args.test_patch_num,
    )
    predictions_df = pd.DataFrame(
        {
            "filename": filenames,
            "predicted_score": predictions,
            "ground_truth": targets,
        }
    )
    return metrics, predictions_df


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    try:
        if is_main_process(rank):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

        train_indices, val_indices = split_saqt_train_val_indices(args)
        if is_main_process(rank):
            print(
                f"SAQT-IQA split: train_indices={len(train_indices)}, "
                f"val_indices={len(val_indices)}"
            )
        train_loader = create_distributed_train_loader(args, rank, world_size, local_rank, train_indices)
        model = build_model(args).to(device)
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.t_max,
            eta_min=args.eta_min,
        )
        criterion = torch.nn.L1Loss().to(device)

        if args.resume is not None:
            load_checkpoint(args.resume, model, optimizer=None, strict=False)

        if distributed_is_initialized():
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        best_srocc = float("-inf")
        metrics_history = []

        for epoch in range(1, args.epochs + 1):
            avg_loss = train_one_epoch_distributed(
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                scaler=scaler if scaler.is_enabled() else None,
                device=device,
                epoch=epoch,
                world_size=world_size,
                use_amp=args.amp,
            )

            if distributed_is_initialized():
                dist.barrier()

            if is_main_process(rank):
                metrics = None
                predictions_df = None
                if val_indices and epoch % args.val_every == 0:
                    metrics, predictions_df = evaluate_saqt_validation_on_main_process(
                        model,
                        args,
                        device,
                        val_indices,
                    )
                    metrics_history.append(
                        {
                            "epoch": epoch,
                            "train_loss": avg_loss,
                            "split": "val",
                            "dataset": "saqt-iqa",
                            "srocc": metrics.srocc,
                            "plcc": metrics.plcc,
                            "krcc": metrics.krcc,
                            "rmse": metrics.rmse,
                        }
                    )
                    pd.DataFrame(metrics_history).to_csv(args.output_dir / "distributed_metrics.csv", index=False)
                    print(
                        f"Epoch {epoch:03d} loss={avg_loss:.6f} "
                        f"val_srocc={metrics.srocc:.4f} val_plcc={metrics.plcc:.4f}"
                    )
                else:
                    print(f"Epoch {epoch:03d} loss={avg_loss:.6f}")

                if args.test_dataset is not None and args.test_every > 0 and epoch % args.test_every == 0:
                    test_metrics, test_predictions_df = evaluate_optional_test_on_main_process(model, args, device)
                    metrics_history.append(
                        {
                            "epoch": epoch,
                            "train_loss": avg_loss,
                            "split": "test",
                            "dataset": canonical_dataset_name(args.test_dataset),
                            "srocc": test_metrics.srocc,
                            "plcc": test_metrics.plcc,
                            "krcc": test_metrics.krcc,
                            "rmse": test_metrics.rmse,
                        }
                    )
                    pd.DataFrame(metrics_history).to_csv(args.output_dir / "distributed_metrics.csv", index=False)
                    test_predictions_df.to_csv(
                        args.output_dir / "predictions" / f"{args.test_dataset}_epoch_{epoch:03d}_predictions.csv",
                        index=False,
                    )
                    print(
                        f"Epoch {epoch:03d} test_{args.test_dataset}_srocc={test_metrics.srocc:.4f} "
                        f"test_{args.test_dataset}_plcc={test_metrics.plcc:.4f}"
                    )

                latest_path = args.output_dir / "checkpoints" / f"{args.dataset}_distributed_latest.pth"
                save_checkpoint(latest_path, model, optimizer, epoch=epoch, best_srocc=best_srocc)
                should_save_periodic = args.save_every > 0 and epoch % args.save_every == 0
                if should_save_periodic:
                    save_checkpoint(
                        args.output_dir / "checkpoints" / f"{args.dataset}_distributed_epoch_{epoch:03d}.pth",
                        model,
                        optimizer,
                        epoch=epoch,
                        best_srocc=best_srocc,
                    )
                if metrics is not None and metrics.srocc > best_srocc:
                    best_srocc = metrics.srocc
                    save_checkpoint(
                        args.output_dir / "checkpoints" / f"{args.dataset}_distributed_best.pth",
                        model,
                        optimizer,
                        epoch=epoch,
                        best_srocc=best_srocc,
                    )
                    if predictions_df is not None:
                        predictions_df.to_csv(
                            args.output_dir / "predictions" / "saqt-iqa_val_best_predictions.csv",
                            index=False,
                        )

            if distributed_is_initialized():
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
