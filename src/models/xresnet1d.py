"""XResNet1D — deep pre-activation 1-D ResNet for 12-lead ECG classification.

Replicates the xresnet1d architecture from the PTB-XL benchmark
(Strodthoff et al., IEEE JBHI 2021) adapted for multi-label classification.

Supported configs:
    xresnet1d50   — [3, 4, 6, 3]  bottleneck stages  (~4M params)
    xresnet1d101  — [3, 4, 23, 3] bottleneck stages  (~8M params)

Design vs CNN1D:
    - Pre-activation bottleneck blocks (BN→ReLU→Conv × 3: 1×1, k×k, 1×1)
    - 3-layer stem with progressive channel growth (avoids large 7×7 stem)
    - Average pooling in downsampling shortcuts (reduces aliasing)
    - Multi-label output (BCEWithLogitsLoss compatible)
    - Configurable kernel size (k=5 default for ECG temporal coverage)
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for XResNet1D.  Install it with:  pip install torch"
    ) from exc


# ---------------------------------------------------------------------------
# Pre-activation bottleneck block
# ---------------------------------------------------------------------------

class _PreActBottleneck1D(nn.Module):
    """Pre-activation bottleneck residual block for 1-D sequences.

    Architecture: BN→ReLU→Conv(1×1) → BN→ReLU→Conv(k×k) → BN→ReLU→Conv(1×1)
    Shortcut uses average-pool + Conv(1×1) when downsampling.

    Args:
        in_channels:  Input channels.
        mid_channels: Bottleneck channels (typically out_channels // 4).
        out_channels: Output channels.
        stride:       Stride for the k×k convolution. Use 2 to halve length.
        kernel_size:  Kernel for the k×k (middle) convolution.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        # Pre-activation layers
        self.bn1   = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1, bias=False)

        self.bn2   = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(
            mid_channels, mid_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )

        self.bn3   = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, 1, bias=False)

        # Shortcut projection when shape changes
        self.shortcut: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            pool = nn.AvgPool1d(stride, stride, ceil_mode=True) if stride > 1 else nn.Identity()
            self.shortcut = nn.Sequential(
                pool,
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = x if self.shortcut is None else self.shortcut(x)

        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))

        return out + residual


# ---------------------------------------------------------------------------
# XResNet1D model
# ---------------------------------------------------------------------------

class XResNet1D(nn.Module):
    """Pre-activation 1-D ResNet for multi-label ECG classification.

    Architecture:
        Stem  : 3 × Conv1d (progressive channel growth) + MaxPool
        Stage1: ``depths[0]`` × PreActBottleneck1D (64 → 128)
        Stage2: ``depths[1]`` × PreActBottleneck1D (128 → 256, stride=2)
        Stage3: ``depths[2]`` × PreActBottleneck1D (256 → 512, stride=2)
        Stage4: ``depths[3]`` × PreActBottleneck1D (512 → 1024, stride=2)
        Head  : BN → ReLU → GlobalAvgPool → Dropout → FC(n_classes)

    Args:
        depths:     Number of bottleneck blocks per stage, e.g. [3, 4, 6, 3].
        n_leads:    Number of ECG leads (input channels). Default 12.
        n_classes:  Output logits (use sigmoid for multi-label). Default 5.
        kernel_size: Kernel for the middle (k×k) conv in each bottleneck.
        dropout:    Dropout probability before the final linear layer.

    Example:
        >>> model = xresnet1d101()
        >>> x = torch.randn(8, 12, 5000)
        >>> logits = model(x)       # shape: (8, 5)
        >>> probs = torch.sigmoid(logits)
    """

    # Channel config: (mid_channels, out_channels) per stage
    _STAGE_CHANNELS = [
        (32,  128),   # stage 1 — no stride
        (64,  256),   # stage 2 — stride 2
        (128, 512),   # stage 3 — stride 2
        (256, 1024),  # stage 4 — stride 2
    ]
    _STEM_CHANNELS = [32, 32, 64]  # progressive stem

    def __init__(
        self,
        depths: list[int],
        n_leads: int = 12,
        n_classes: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert len(depths) == 4, "depths must have exactly 4 elements"

        sc = self._STEM_CHANNELS

        # Stem: 3 × Conv(post-activation) + MaxPool
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads,  sc[0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(sc[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(sc[0], sc[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(sc[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(sc[1], sc[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(sc[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Build 4 stages
        in_ch = sc[2]  # 64 after stem
        stages = []
        strides = [1, 2, 2, 2]
        for i, (depth, stride) in enumerate(zip(depths, strides)):
            mid_ch, out_ch = self._STAGE_CHANNELS[i]
            stages.append(self._make_stage(in_ch, mid_ch, out_ch, depth, stride, kernel_size))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Head: final BN+ReLU (pre-act pattern), then GAP + FC
        self.head_bn  = nn.BatchNorm1d(in_ch)
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(in_ch, n_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        depth: int,
        stride: int,
        kernel_size: int,
    ) -> nn.Sequential:
        blocks = [_PreActBottleneck1D(in_ch, mid_ch, out_ch, stride=stride, kernel_size=kernel_size)]
        for _ in range(1, depth):
            blocks.append(_PreActBottleneck1D(out_ch, mid_ch, out_ch, kernel_size=kernel_size))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_leads, samples).

        Returns:
            Logit tensor of shape (batch, n_classes).
            Apply torch.sigmoid() to get per-class probabilities.
        """
        x = self.stem(x)
        x = self.stages(x)
        x = F.relu(self.head_bn(x), inplace=True)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def xresnet1d50(n_leads: int = 12, n_classes: int = 5, **kwargs) -> XResNet1D:
    """XResNet1D-50: depths=[3,4,6,3], ~4M parameters."""
    return XResNet1D([3, 4, 6, 3], n_leads=n_leads, n_classes=n_classes, **kwargs)


def xresnet1d101(n_leads: int = 12, n_classes: int = 5, **kwargs) -> XResNet1D:
    """XResNet1D-101: depths=[3,4,23,3], ~8M parameters."""
    return XResNet1D([3, 4, 23, 3], n_leads=n_leads, n_classes=n_classes, **kwargs)
