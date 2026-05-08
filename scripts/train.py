from __future__ import annotations

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from glintiqa.configs.defaults import build_parser
from glintiqa.data import canonical_dataset_name, create_dataloader, train_test_split_indices
from glintiqa.engine.checkpoint import load_checkpoint, save_checkpoint
from glintiqa.engine.evaluate import evaluate_iqa
from glintiqa.engine.metrics import compute_iqa_metrics
from glintiqa.engine.train import train_one_epoch
from glintiqa.models import create_iqa_model


def setup_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(args):
    return create_iqa_model(
        args.model,
        arch=args.arch,
        img_size=args.img_size,
        patch_size=args.vit_patch_size,
        embed_dim=args.embed_dim,
    )


def train_one_round(args, round_num: int, device: torch.device):
    round_seed = round_num * args.seed
    setup_seed(round_seed)
    train_indices, test_indices = train_test_split_indices(args.dataset, seed=round_seed)
    model = create_model(args).to(device)

    train_loader = create_dataloader(
        args=args,
        dataset_name=args.dataset,
        root=args.dataset_root,
        indices=train_indices,
        patch_size=args.patch_size,
        patch_num=args.train_patch_num,
        is_train=True,
    )
    test_loader = create_dataloader(
        args=args,
        dataset_name=args.dataset,
        root=args.dataset_root,
        indices=test_indices,
        patch_size=args.patch_size,
        patch_num=args.test_patch_num,
        is_train=False,
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

    if args.resume is not None:
        load_checkpoint(args.resume, model, optimizer=None, strict=False)

    best_metrics = None
    best_epoch = 0
    best_srocc = float("-inf")
    checkpoint_dir = args.output_dir / "checkpoints"
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRound {round_num}/{args.train_test_round} seed={round_seed}")
    print(f"Train references: {len(train_indices)}, test references: {len(test_indices)}")
    for epoch in range(1, args.epochs + 1):
        losses, train_preds, train_targets = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        train_metrics = compute_iqa_metrics(train_preds, train_targets)
        test_metrics, predictions, targets, filenames = evaluate_iqa(
            model=model,
            data_loader=test_loader,
            device=device,
            patch_num=args.test_patch_num,
        )
        avg_loss = sum(losses) / max(len(losses), 1)
        print(
            f"Round {round_num:02d} Epoch {epoch:03d} loss={avg_loss:.6f} "
            f"train_srocc={train_metrics.srocc:.4f} test_srocc={test_metrics.srocc:.4f} "
            f"test_plcc={test_metrics.plcc:.4f}"
        )

        latest_path = checkpoint_dir / f"{args.dataset}_round_{round_num}_latest.pth"
        save_checkpoint(latest_path, model, optimizer, epoch=epoch, best_srocc=best_srocc)
        if test_metrics.srocc > best_srocc:
            best_srocc = test_metrics.srocc
            best_epoch = epoch
            best_metrics = test_metrics
            best_path = checkpoint_dir / f"{args.dataset}_round_{round_num}_best.pth"
            save_checkpoint(best_path, model, optimizer, epoch=epoch, best_srocc=best_srocc)
            pd.DataFrame(
                {
                    "filename": filenames,
                    "predicted_score": predictions,
                    "ground_truth": targets,
                }
            ).to_csv(prediction_dir / f"{args.dataset}_round_{round_num}_best_predictions.csv", index=False)

    if best_metrics is None:
        raise RuntimeError(f"Round {round_num} finished without evaluation metrics.")
    return {
        "round": round_num,
        "seed": round_seed,
        "best_epoch": best_epoch,
        "srocc": best_metrics.srocc,
        "plcc": best_metrics.plcc,
        "krcc": best_metrics.krcc,
        "rmse": best_metrics.rmse,
    }


def main() -> None:
    parser = build_parser()
    parser.add_argument("--train-test-round", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for training.")

    args.dataset = canonical_dataset_name(args.dataset)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.train_test_round <= 0:
        raise ValueError("--train-test-round must be greater than 0.")

    print(f"Training {args.model} on {args.dataset} for {args.train_test_round} rounds")
    round_results = []
    for round_num in range(1, args.train_test_round + 1):
        result = train_one_round(args, round_num=round_num, device=device)
        round_results.append(result)
        print(
            f"Round {round_num:02d} best epoch={result['best_epoch']} "
            f"SROCC={result['srocc']:.4f}, PLCC={result['plcc']:.4f}, "
            f"KRCC={result['krcc']:.4f}, RMSE={result['rmse']:.4f}"
        )

    results_df = pd.DataFrame(round_results)
    results_path = args.output_dir / f"{args.dataset}_round_results.csv"
    results_df.to_csv(results_path, index=False)

    median_result = {
        "srocc_median": float(np.median(results_df["srocc"])),
        "plcc_median": float(np.median(results_df["plcc"])),
        "krcc_median": float(np.median(results_df["krcc"])),
        "rmse_median": float(np.median(results_df["rmse"])),
    }
    pd.DataFrame([median_result]).to_csv(args.output_dir / f"{args.dataset}_median_results.csv", index=False)
    print("\nFinal median performance")
    print(
        f"SROCC={median_result['srocc_median']:.4f}, "
        f"PLCC={median_result['plcc_median']:.4f}, "
        f"KRCC={median_result['krcc_median']:.4f}, "
        f"RMSE={median_result['rmse_median']:.4f}"
    )
    print(f"Round results saved to {results_path}")


if __name__ == "__main__":
    main()
