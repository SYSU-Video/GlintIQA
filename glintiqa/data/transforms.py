from __future__ import annotations

import torchvision


STANDARD_DATASETS = {
    "live",
    "csiq",
    "tid2013",
    "kadid-10k",
    "livemd",

    "bid",
    "clive",
    "koniq-10k",
    "spaq",
    "fblive",

    "saqt-iqa",
    "generated_dataset",
    
}
RESIZE_512_384_DATASETS = {"livemd", "bid"}
RESIZE_512_512_DATASETS = {"fblive"}


def build_iqa_transform(dataset_name: str, patch_size: int, is_train: bool):
    normalize = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if dataset_name in RESIZE_512_512_DATASETS:
        # normalize = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transforms = [torchvision.transforms.Resize((512, 512))]
    else:
        
        transforms = []
        if dataset_name in RESIZE_512_384_DATASETS:
            transforms.append(torchvision.transforms.Resize((512, 384)))

    if is_train:
        transforms.extend(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
            ]
        )

    transforms.extend(
        [
            torchvision.transforms.RandomCrop(size=patch_size),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )
    return torchvision.transforms.Compose(transforms)

