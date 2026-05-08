from __future__ import annotations

from pathlib import Path

import torch

from .datasets import create_iqa_dataset


def create_dataloader(
    args,
    dataset_name: str,
    root: Path,
    indices,
    patch_size: int,
    patch_num: int,
    is_train: bool,
):
    dataset = create_iqa_dataset(
        dataset_name=dataset_name,
        root=Path(root),
        indices=indices,
        patch_size=patch_size,
        patch_num=patch_num,
        is_train=is_train,
        args=args,
    )
    batch_size = args.batch_size if is_train else getattr(args, "eval_batch_size", args.batch_size)
    return create_torch_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        is_train=is_train,
    )


def create_torch_dataloader(dataset, batch_size: int, num_workers: int, is_train: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )
