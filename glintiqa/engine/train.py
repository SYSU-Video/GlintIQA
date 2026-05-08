from __future__ import annotations

import torch
from tqdm import tqdm


def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True).flatten()
        filenames = batch.get("filename")
        return images, labels, filenames
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True).flatten(), None


def train_one_epoch(train_loader, model, optimizer, scheduler, criterion, device, epoch: int):
    model.train()
    losses, predictions, targets = [], [], []
    progress = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for batch in progress:
        images, labels, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(images).flatten()
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.detach().cpu()))
        predictions.extend(preds.detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())
        progress.set_postfix({"loss": f"{losses[-1]:.4f}"})
    return losses, predictions, targets

