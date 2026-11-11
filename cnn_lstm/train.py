"""Training with early stopping, one-cycle scheduling and gradient clipping."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .data import Augmenter, Normalizer, Split


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    patience: int = 6
    augment: bool = True
    seed: int = 42


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(
    model: nn.Module, x: np.ndarray, norm: Normalizer, batch_size: int = 128
) -> np.ndarray:
    """Class probabilities [N, num_classes] for a uint8 NHWC array."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        logits = model(norm(norm.to_unit(x[i : i + batch_size])))
        out.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(out)


def _evaluate_loss(model, split: Split, norm, criterion, batch_size=128) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(split.x), batch_size):
            xb = norm(norm.to_unit(split.x[i : i + batch_size]))
            yb = torch.from_numpy(split.y[i : i + batch_size])
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(split.x), correct / len(split.x)


def fit(
    model: nn.Module,
    train: Split,
    val: Split,
    norm: Normalizer,
    cfg: TrainConfig,
    log=print,
) -> History:
    """Train ``model`` in place and restore the weights from the best validation epoch."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    augment = Augmenter(seed=cfg.seed) if cfg.augment else None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train.x) / cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
    )

    hist = History()
    best_loss, best_state, bad_epochs = math.inf, None, 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(train.x))
        run_loss, run_correct, seen = 0.0, 0, 0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            xb = norm.to_unit(train.x[idx])
            if augment is not None:
                xb = augment(xb)
            xb = norm(xb)
            yb = torch.from_numpy(train.y[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)

        val_loss, val_acc = _evaluate_loss(model, val, norm, eval_criterion)
        hist.train_loss.append(run_loss / seen)
        hist.train_acc.append(run_correct / seen)
        hist.val_loss.append(val_loss)
        hist.val_acc.append(val_acc)
        hist.lr.append(optimizer.param_groups[0]["lr"])
        hist.epoch_seconds.append(time.perf_counter() - t0)
        log(
            f"  epoch {epoch:>2}/{cfg.epochs}  train loss {hist.train_loss[-1]:.4f} "
            f"acc {hist.train_acc[-1]:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} "
            f"| {hist.epoch_seconds[-1]:.1f}s"
        )

        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs, hist.best_epoch = val_loss, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                hist.stopped_early = True
                log(f"  early stop: no val-loss improvement for {cfg.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
