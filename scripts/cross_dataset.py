from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from glintiqa.configs.defaults import build_parser
from glintiqa.data import canonical_dataset_name, create_dataloader, dataset_indices
from glintiqa.engine.checkpoint import load_checkpoint, save_checkpoint
from glintiqa.engine.evaluate import evaluate_iqa
from glintiqa.engine.metrics import compute_iqa_metrics
from glintiqa.engine.train import train_one_epoch
from glintiqa.models import create_iqa_model


def parse_args():
    parser = build_parser()
    parser.description = "Train GlintIQA on one dataset and test on other datasets."
    parser.add_argument("--train-dataset", required=True, help="Source dataset used for training.")
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        default=None,
        help="Root of the source dataset. Defaults to --dataset-root.",
    )
    parser.add_argument("--test-datasets", nargs="+", required=True, help="Target datasets for cross testing.")
    parser.add_argument(
        "--test-dataset-roots",
        nargs="+",
        type=Path,
        required=True,
        help="Target dataset roots in the same order as --test-datasets.",
    )
    parser.add_argument(
        "--train-index-start",
        type=int,
        default=None,
        help="Optional start offset for source dataset reference indices.",
    )
    parser.add_argument(
        "--train-index-end",
        type=int,
        default=None,
        help="Optional end offset for source dataset reference indices.",
    )
    return parser.parse_args()


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


def build_test_root_map(test_datasets: List[str], test_roots: List[Path]) -> Dict[str, Path]:
    if len(test_datasets) != len(test_roots):
        raise ValueError("--test-datasets and --test-dataset-roots must have the same length.")
    return {
        canonical_dataset_name(dataset_name): Path(root)
        for dataset_name, root in zip(test_datasets, test_roots)
    }


def evaluate_targets(model, args, test_root_map: Dict[str, Path], device: torch.device, epoch: int):
    rows = []
    predictions_by_dataset = {}
    for dataset_name, root in test_root_map.items():
        eval_args = copy.copy(args)
        data_loader = create_dataloader(
            args=eval_args,
            dataset_name=dataset_name,
            root=root,
            indices=resolve_indices(dataset_name),
            patch_size=args.patch_size,
            patch_num=args.test_patch_num,
            is_train=False,
        )
        metrics, predictions, targets, filenames = evaluate_iqa(
            model=model,
            data_loader=data_loader,
            device=device,
            patch_num=args.test_patch_num,
        )
        rows.append(
            {
                "epoch": epoch,
                "dataset": dataset_name,
                "srocc": metrics.srocc,
                "plcc": metrics.plcc,
                "krcc": metrics.krcc,
                "rmse": metrics.rmse,
            }
        )
        predictions_by_dataset[dataset_name] = pd.DataFrame(
            {
                "filename": filenames,
                "predicted_score": predictions,
                "ground_truth": targets,
            }
        )
    return rows, predictions_by_dataset


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.train_dataset = canonical_dataset_name(args.train_dataset)
    train_root = args.train_dataset_root or args.dataset_root
    if train_root is None:
        raise ValueError("--train-dataset-root or --dataset-root is required.")

    test_root_map = build_test_root_map(args.test_datasets, args.test_dataset_roots)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    if args.resume is not None:
        load_checkpoint(args.resume, model, optimizer=None, strict=False)

    train_loader = create_dataloader(
        args=args,
        dataset_name=args.train_dataset,
        root=train_root,
        indices=resolve_indices(args.train_dataset, args.train_index_start, args.train_index_end),
        patch_size=args.patch_size,
        patch_num=args.train_patch_num,
        is_train=True,
    )

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

    metrics_history = []
    best_average_srocc = float("-inf")
    checkpoint_dir = args.output_dir / "checkpoints"
    prediction_dir = args.output_dir / "cross_dataset_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cross-dataset training source: {args.train_dataset}")
    print(f"Cross-dataset targets: {', '.join(test_root_map)}")
    for epoch in range(1, args.epochs + 1):
        losses, train_predictions, train_targets = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        train_metrics = compute_iqa_metrics(train_predictions, train_targets)
        target_rows, predictions_by_dataset = evaluate_targets(
            model=model,
            args=args,
            test_root_map=test_root_map,
            device=device,
            epoch=epoch,
        )
        average_srocc = sum(row["srocc"] for row in target_rows) / max(len(target_rows), 1)
        average_plcc = sum(row["plcc"] for row in target_rows) / max(len(target_rows), 1)
        avg_loss = sum(losses) / max(len(losses), 1)
        metrics_history.extend(target_rows)
        pd.DataFrame(metrics_history).to_csv(args.output_dir / "cross_dataset_metrics.csv", index=False)

        print(
            f"Epoch {epoch:03d} loss={avg_loss:.6f} train_srocc={train_metrics.srocc:.4f} "
            f"avg_test_srocc={average_srocc:.4f} avg_test_plcc={average_plcc:.4f}"
        )
        for row in target_rows:
            print(
                f"  {row['dataset']}: SROCC={row['srocc']:.4f}, "
                f"PLCC={row['plcc']:.4f}, KRCC={row['krcc']:.4f}, RMSE={row['rmse']:.4f}"
            )

        save_checkpoint(
            checkpoint_dir / f"{args.train_dataset}_cross_latest.pth",
            model,
            optimizer,
            epoch=epoch,
            best_srocc=best_average_srocc,
        )
        if average_srocc > best_average_srocc:
            best_average_srocc = average_srocc
            save_checkpoint(
                checkpoint_dir / f"{args.train_dataset}_cross_best.pth",
                model,
                optimizer,
                epoch=epoch,
                best_srocc=best_average_srocc,
                extra={"average_plcc": average_plcc, "test_datasets": list(test_root_map)},
            )
            for dataset_name, predictions_df in predictions_by_dataset.items():
                predictions_df.to_csv(prediction_dir / f"{dataset_name}_best_predictions.csv", index=False)


if __name__ == "__main__":
    main()
