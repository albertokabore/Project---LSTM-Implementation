"""Model architectures.

* ``PixelLSTM``  - LSTM only: each image row of raw pixels is one time step.
* ``CNNClassifier`` - CNN only: the same convolutional backbone + global average pooling.
* ``CNNLSTM``    - CNN backbone, then each row of the feature map is one time step of a
  (bi)directional LSTM. This is the project's main model.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    """Two 3x3 conv + BatchNorm + ReLU layers, then 2x2 max-pool and spatial dropout."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class CNNBackbone(nn.Module):
    """64x64x3 -> 8x8x(channels[-1]) with three conv blocks."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128), dropout: float = 0.1):
        super().__init__()
        blocks, c_in = [], 3
        for c_out in channels:
            blocks.append(conv_block(c_in, c_out, dropout))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.out_channels = c_in
        self.downsample = 2 ** len(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def classifier_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """Dropout -> Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 128, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
    )


def _last_hidden(h_n: torch.Tensor, bidirectional: bool) -> torch.Tensor:
    """Final hidden state of the top LSTM layer, both directions concatenated."""
    if bidirectional:
        return torch.cat([h_n[-2], h_n[-1]], dim=1)
    return h_n[-1]


class PixelLSTM(nn.Module):
    """Reads the image top to bottom, one row of H x 3 pixels per time step."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 64,
        hidden: int = 128,
        layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_norm = nn.LayerNorm(image_size * 3)
        self.lstm = nn.LSTM(
            image_size * 3,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = classifier_head(hidden * (2 if bidirectional else 1), num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.permute(0, 2, 1, 3).flatten(2)  # [B, H, C*W]
        _, (h_n, _) = self.lstm(self.input_norm(seq))
        return self.head(_last_hidden(h_n, self.bidirectional))


class CNNClassifier(nn.Module):
    """CNN backbone + global average pooling; no recurrence (ablation baseline)."""

    def __init__(
        self, num_classes: int, image_size: int = 64, channels=(32, 64, 128), dropout: float = 0.3
    ):
        super().__init__()
        del image_size  # global pooling makes the head size-independent; kept for a uniform API
        self.backbone = CNNBackbone(channels, dropout=min(dropout, 0.1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = classifier_head(self.backbone.out_channels, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)).flatten(1))


class CNNLSTM(nn.Module):
    """CNN feature extractor followed by an LSTM over feature-map rows.

    For a 64x64 input the backbone yields an 8x8x128 map. Each of its 8 rows
    (8 columns x 128 channels = 1024 features) is one time step, so the LSTM models how
    spatial patterns are arranged from the top of the image to the bottom.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 64,
        channels=(32, 64, 128),
        hidden: int = 128,
        layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.backbone = CNNBackbone(channels, dropout=min(dropout, 0.1))
        feat_w = image_size // self.backbone.downsample
        self.seq_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            self.backbone.out_channels * feat_w,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = classifier_head(hidden * (2 if bidirectional else 1), num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)  # [B, C, H', W']
        seq = fmap.permute(0, 2, 1, 3).flatten(2)  # [B, H', C*W']
        _, (h_n, _) = self.lstm(self.seq_dropout(seq))
        return self.head(_last_hidden(h_n, self.bidirectional))


MODELS = {"pixel_lstm": PixelLSTM, "cnn": CNNClassifier, "cnn_lstm": CNNLSTM}


def build_model(arch: str, num_classes: int, **kwargs) -> nn.Module:
    if arch not in MODELS:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {sorted(MODELS)}")
    return MODELS[arch](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module, image_size: int = 64) -> str:
    """Layer-by-layer summary (output shape and parameter count) for the report."""
    rows: list[tuple[str, str, int]] = []
    hooks = []

    def hook(name):
        def fn(mod, _inp, out):
            shape = out[0].shape if isinstance(out, tuple) else out.shape
            own = sum(p.numel() for p in mod.parameters(recurse=False))
            rows.append((f"{name} ({type(mod).__name__})", str(list(shape)), own))

        return fn

    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            hooks.append(mod.register_forward_hook(hook(name)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, 3, image_size, image_size))
    model.train(was_training)
    for h in hooks:
        h.remove()

    width = max(len(r[0]) for r in rows)
    lines = [f"{'Layer'.ljust(width)}  {'Output shape':<18}  Params", "-" * (width + 30)]
    lines += [f"{n.ljust(width)}  {s:<18}  {p:,}" for n, s, p in rows]
    lines.append("-" * (width + 30))
    lines.append(f"Total trainable parameters: {count_parameters(model):,}")
    return "\n".join(lines)
