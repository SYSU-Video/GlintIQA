from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saqt.similarity_retrieval import retrieve_matches_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve semantically matched KADID references for source images.")
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--reference-feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("./outputs"))
    parser.add_argument("--pair-name", default="kadis700k_to_kadid-10k")
    parser.add_argument("--model-name", default="resnet101")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--source-list",
        type=Path,
        default=None,
        help=(
            "Optional selected source image list. Supports a TXT file with one image name per line, "
            "or a CSV file containing a 'kadis', 'source_image', 'image', or 'filename' column."
        ),
    )

    parser.add_argument(
        "--legacy-reference-decimals",
        type=int,
        default=None,
        help=(
            "Round loaded reference features before similarity. Use 6 to mimic old Step 1 "
            "np.savetxt(fmt='%%f') precision."
        ),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = (
        args.output_root
        / "retrieval"
        / args.pair_name
        / args.model_name
        / f"top{args.top_k}_semantic_matches.csv"
    )
    retrieve_matches_to_csv(
        source_image_root=args.source_image_root,
        reference_feature_dir=args.reference_feature_dir,
        output_csv=output_csv,
        model_name=args.model_name,
        weights=args.weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k,
        recursive=args.recursive,
        source_list=args.source_list,
        device=args.device,
        legacy_reference_decimals=args.legacy_reference_decimals,
    )
    print(f"Saved semantic matches to: {output_csv}")


if __name__ == "__main__":
    main()
