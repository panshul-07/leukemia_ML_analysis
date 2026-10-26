"""Shared training helpers used by the two CNN+GNN experiment runners."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model, loader, criterion, optimizer, device, max_grad_norm: float | None = None
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = total = 0
    for inputs, labels in tqdm(loader, desc="train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    collect_embeddings: bool = False,
    measure_inference: bool = False,
    test_time_augmentation: bool = False,
):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    inference_times_ms: list[float] = []

    for inputs, labels in tqdm(loader, desc="eval", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        if measure_inference and device.type == "cuda":
            torch.cuda.synchronize()
        if measure_inference and device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        views = [inputs]
        if test_time_augmentation:
            views += [
                torch.flip(inputs, (-1,)),
                torch.flip(inputs, (-2,)),
                torch.flip(inputs, (-2, -1)),
            ]
        view_logits, view_embeddings = [], []
        for view in views:
            if collect_embeddings:
                logits, embeddings = model.forward_with_features(view)
                view_embeddings.append(embeddings)
            else:
                logits = model(view)
            view_logits.append(logits)
        logits = torch.stack(view_logits).mean(0)
        if collect_embeddings:
            all_embeddings.append(torch.stack(view_embeddings).mean(0).cpu().numpy())
        if measure_inference and device.type == "cuda":
            torch.cuda.synchronize()
        if measure_inference and device.type == "mps":
            torch.mps.synchronize()
        if measure_inference:
            elapsed = (time.perf_counter() - started) * 1000 / len(labels)
            inference_times_ms.extend([elapsed] * len(labels))
        loss = criterion(logits, labels)
        probabilities = logits.softmax(1)
        predictions = logits.argmax(1)
        running_loss += float(loss) * len(labels)
        correct += int((predictions == labels).sum())
        total += len(labels)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.append(probabilities.cpu().numpy())

    embeddings = np.concatenate(all_embeddings) if all_embeddings else None
    return (
        running_loss / total,
        correct / total,
        all_labels,
        all_predictions,
        np.concatenate(all_probabilities),
        embeddings,
        inference_times_ms,
    )
