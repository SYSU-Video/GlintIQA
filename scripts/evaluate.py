from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from glintiqa.configs.defaults import build_parser
from glintiqa.data import canonical_dataset_name, create_dataloader, dataset_indices
from glintiqa.engine.checkpoint import load_checkpoint
from glintiqa.engine.evaluate import evaluate_iqa
from glintiqa.models import create_iqa_model


def main() -> None:
    parser = build_parser()
    parser.description = "Evaluate a GlintIQA checkpoint"
    args = parser.parse_args()
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for evaluation.")
    if args.resume is None:
        raise ValueError("--resume is required for evaluation.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset = canonical_dataset_name(args.dataset)
    model = create_iqa_model(
        args.model,
        arch=args.arch,
        img_size=args.img_size,
        patch_size=args.vit_patch_size,
        embed_dim=args.embed_dim,
    )
    load_checkpoint(args.resume, model, strict=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_loader = create_dataloader(
        args=args,
        dataset_name=args.dataset,
        root=args.dataset_root,
        indices=dataset_indices(args.dataset),
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
    output_csv = args.output_dir / f"{args.dataset}_predictions.csv"
    pd.DataFrame(
        {
            "filename": filenames,
            "predicted_score": predictions,
            "ground_truth": targets,
        }
    ).to_csv(output_csv, index=False)
    print(
        f"SROCC={metrics.srocc:.4f}, PLCC={metrics.plcc:.4f}, "
        f"KRCC={metrics.krcc:.4f}, RMSE={metrics.rmse:.4f}"
    )
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    main()
