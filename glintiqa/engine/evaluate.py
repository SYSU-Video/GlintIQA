from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_iqa_metrics
from .train import move_batch_to_device


def evaluate_iqa(model, data_loader, device, patch_num: int = 25):
    model.eval()
    predictions, targets, filenames = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images, labels, batch_filenames = move_batch_to_device(batch, device)
            preds = model(images).flatten()
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            if batch_filenames is None:
                batch_filenames = [f"sample_{idx}" for idx in range(len(labels))]
            filenames.extend(batch_filenames)

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if patch_num > 1 and len(predictions) % patch_num == 0:
        num_images = len(predictions) // patch_num
        predictions = predictions.reshape(num_images, patch_num).mean(axis=1)
        targets = targets.reshape(num_images, patch_num).mean(axis=1)
        filenames = [filenames[idx * patch_num] for idx in range(num_images)]

    return compute_iqa_metrics(predictions, targets), predictions, targets, filenames

