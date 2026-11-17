"""1D ResNet-style CNN for 12-lead ECG classification.

Requires PyTorch (not installed by default).  Install with:
    pip install torch

Architecture (input shape: batch × n_leads × samples):
    Stem  : Conv1d(n_leads→64, k=15, stride=2) → BN → ReLU → MaxPool
    Stage1: 2× ResBlock1D(64→64)
    Stage2: 2× ResBlock1D(64→128, stride=2)
    Stage3: 2× ResBlock1D(128→256, stride=2)
    Head  : GlobalAvgPool → FC(256→128) → ReLU → Dropout → FC(128→n_classes)

For a 500 Hz / 10 s signal (5000 samples) the receptive field covers ~260 ms,
sufficient to capture a full QRS complex and surrounding P/T waves.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for CNN1D.  Install it with:  pip install torch"
    ) from exc


class ResBlock1D(nn.Module):
    """Pre-activation residual block for 1-D sequences.

    Applies two Conv1d layers with BatchNorm and ReLU activations.
    A projection shortcut is added when the number of channels or the
    temporal stride changes.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolutional kernel size (odd integer recommended).
        stride: Stride for the first convolution (use 2 to halve sequence length).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return F.relu(out + residual, inplace=True)


class CNN1D(nn.Module):
    """Residual 1D CNN for multi-lead ECG binary classification.

    Accepts raw or preprocessed ECG signals and outputs class logits.
    For inference, apply softmax or sigmoid to the output.

    Args:
        n_leads: Number of ECG leads (input channels).  Defaults to 12.
        n_classes: Number of output classes.  Defaults to 2 (binary CVD vs NORM).
        dropout: Dropout probability applied before the final linear layer.

    Example:
        >>> model = CNN1D(n_leads=12, n_classes=2)
        >>> x = torch.randn(8, 12, 5000)   # batch of 8 records
        >>> logits = model(x)              # shape: (8, 2)
    """

    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stem: aggressive downsampling to reduce sequence length early
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages — each stage doubles channels, halves length (except stage1)
        self.stage1 = nn.Sequential(ResBlock1D(64, 64), ResBlock1D(64, 64))
        self.stage2 = nn.Sequential(ResBlock1D(64, 128, stride=2), ResBlock1D(128, 128))
        self.stage3 = nn.Sequential(ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256))

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for Conv layers, constant init for BN."""
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
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
