"""ECGFounder — Net1D backbone with LoRA adapters for PTB-XL fine-tuning.

Architecture: Net1D (RegNet-style 1-D CNN) from Hong 2020, as used in
ECGFounder (Li et al., NEJM AI 2025).  Pretrained on 10M+ ECG recordings.
GitHub: github.com/PKUDigitalHealth/ECGFounder
HuggingFace: PKUDigitalHealth/ECGFounder  (12_lead_ECGFounder.pth)

LoRA strategy:
    - Backbone Conv layers: FROZEN
    - SE module Linear layers (se_fc1, se_fc2): replaced by LoRALinear
    - Final classifier (dense): replaced with new nn.Linear(1024, n_classes)

Trainable params after LoRA injection (rank=8):
    ~300–500K  (vs 33.8M total) — safe for 17K samples.

Usage:
    >>> from src.models.ecgfounder import build_ecgfounder_lora
    >>> model = build_ecgfounder_lora(
    ...     n_classes=5,
    ...     rank=8,
    ...     alpha=16.0,
    ...     checkpoint="checkpoints/ecgfounder/12_lead_ECGFounder.pth",
    ... )
"""
import logging
import math
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for ECGFounder.  Install it with:  pip install torch"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Net1D — exact architecture from ECGFounder repo (Hong 2020 / Li 2025)
# ---------------------------------------------------------------------------

class _MyConv1dPadSame(nn.Module):
    """nn.Conv1d with TensorFlow-style SAME padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, groups=groups,
        )

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        x = F.pad(x, (p // 2, p - p // 2))
        return self.conv(x)


class _MyMaxPool1dPadSame(nn.Module):
    """nn.MaxPool1d with SAME padding."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x):
        p = max(0, self.kernel_size - 1)
        x = F.pad(x, (p // 2, p - p // 2))
        return self.pool(x)


class _Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    """Pre-activation bottleneck block with Squeeze-and-Excitation.

    Contains the two Linear layers targeted by LoRA: ``se_fc1`` and ``se_fc2``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        is_first_block: bool = False,
        use_bn: bool = True,
        use_do: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        mid = int(out_channels * ratio)

        # Conv path
        self.bn1          = nn.BatchNorm1d(in_channels)
        self.activation1  = _Swish()
        self.do1          = nn.Dropout(0.5)
        self.conv1        = _MyConv1dPadSame(in_channels, mid, 1, 1)

        self.bn2          = nn.BatchNorm1d(mid)
        self.activation2  = _Swish()
        self.do2          = nn.Dropout(0.5)
        self.conv2        = _MyConv1dPadSame(mid, mid, kernel_size, self.stride, groups)

        self.bn3          = nn.BatchNorm1d(mid)
        self.activation3  = _Swish()
        self.do3          = nn.Dropout(0.5)
        self.conv3        = _MyConv1dPadSame(mid, out_channels, 1, 1)

        # Squeeze-and-Excitation (LoRA targets)
        self.se_fc1       = nn.Linear(out_channels, out_channels // 2)
        self.se_fc2       = nn.Linear(out_channels // 2, out_channels)
        self.se_activation = _Swish()

        if downsample:
            self.max_pool = _MyMaxPool1dPadSame(self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)

        # SE
        se = out.mean(-1)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se)
        out = torch.einsum("abc,ab->abc", out, se)

        # Shortcut
        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2))
            identity = identity.transpose(-1, -2)

        return out + identity


class _BasicStage(nn.Module):
    def __init__(
        self,
        in_channels, out_channels, ratio, kernel_size, stride,
        groups, i_stage, m_blocks, use_bn=True, use_do=True,
    ) -> None:
        super().__init__()
        self.block_list = nn.ModuleList()
        for i_block in range(m_blocks):
            is_first = (i_stage == 0 and i_block == 0)
            do_down  = (i_block == 0)
            in_ch    = in_channels if do_down else out_channels
            self.block_list.append(BasicBlock(
                in_channels=in_ch,
                out_channels=out_channels,
                ratio=ratio,
                kernel_size=kernel_size,
                stride=stride if do_down else 1,
                groups=groups,
                downsample=do_down,
                is_first_block=is_first,
                use_bn=use_bn,
                use_do=use_do,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            x = block(x)
        return x


class Net1D(nn.Module):
    """ECGFounder backbone (Net1D-RegNet).

    Input:  (batch, in_channels, L)
    Output: (batch, n_classes)  logits
    """

    # ECGFounder 12-lead config (do NOT change — must match pretrained weights)
    _HF_REPO   = "PKUDigitalHealth/ECGFounder"
    _HF_FILE   = "12_lead_ECGFounder.pth"

    def __init__(
        self,
        in_channels: int = 12,
        base_filters: int = 64,
        ratio: float = 1.0,
        filter_list: list[int] | None = None,
        m_blocks_list: list[int] | None = None,
        kernel_size: int = 16,
        stride: int = 2,
        groups_width: int = 16,
        n_classes: int = 5,
        use_bn: bool = False,
        use_do: bool = False,
    ) -> None:
        super().__init__()
        filter_list    = filter_list    or [64, 160, 160, 400, 400, 1024, 1024]
        m_blocks_list  = m_blocks_list  or [2, 2, 2, 3, 3, 4, 4]

        self.first_conv       = _MyConv1dPadSame(in_channels, base_filters, kernel_size, 2)
        self.first_bn         = nn.BatchNorm1d(base_filters)
        self.first_activation = _Swish()

        self.stage_list = nn.ModuleList()
        in_ch = base_filters
        for i, (out_ch, n_blocks) in enumerate(zip(filter_list, m_blocks_list)):
            groups = out_ch // groups_width
            self.stage_list.append(_BasicStage(
                in_channels=in_ch, out_channels=out_ch,
                ratio=ratio, kernel_size=kernel_size, stride=stride,
                groups=groups, i_stage=i, m_blocks=n_blocks,
                use_bn=use_bn, use_do=use_do,
            ))
            in_ch = out_ch

        self.dense = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)
        if hasattr(self, "first_bn"):
            pass  # use_bn=False for ECGFounder pretrained — skip BN
        out = self.first_activation(out)
        for stage in self.stage_list:
            out = stage(out)
        return self.dense(out.mean(-1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with frozen base + trainable LoRA deltas.

    Implements W' = W + (B @ A) * (alpha / rank), where:
        W  = frozen pretrained weight (stored as buffer)
        A  = (rank, in_features)   — initialised with kaiming uniform
        B  = (out_features, rank)  — initialised to zero (delta = 0 at start)

    The zero initialisation of B guarantees that fine-tuning begins from
    exactly the pretrained checkpoint, with no initial perturbation.
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features  = original.in_features
        self.out_features = original.out_features
        self.rank         = rank
        self.scale        = alpha / rank

        # Pretrained weights stored as frozen buffers (appear in state_dict
        # but have no gradient — they will be skipped by the optimiser).
        self.register_buffer("weight", original.weight.data.clone())
        if original.bias is not None:
            self.register_buffer("bias", original.bias.data.clone())
        else:
            self.register_buffer("bias", None)

        # Trainable LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + delta

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3f}"
        )


def inject_lora(
    model: Net1D,
    rank: int = 8,
    alpha: float = 16.0,
    unfreeze_last_n_stages: int = 0,
) -> None:
    """Replace se_fc1 / se_fc2 in every BasicBlock with LoRALinear in-place.

    After injection:
    - Backbone Conv weights of frozen stages: requires_grad = False
    - LoRA adapters (lora_A, lora_B):         requires_grad = True
    - model.dense (new head):                  requires_grad = True
    - Last ``unfreeze_last_n_stages`` stages:  ALL params requires_grad = True
      (no LoRA injection — full fine-tuning of those stages)

    Args:
        model:                  Net1D instance with pretrained weights loaded.
        rank:                   LoRA rank (adapter dimensions).
        alpha:                  LoRA scaling factor.  scale = alpha / rank.
        unfreeze_last_n_stages: Number of trailing backbone stages to fully
                                unfreeze (0 = all stages LoRA-only, default).
    """
    n_stages = len(model.stage_list)
    lora_stage_indices   = set(range(n_stages - unfreeze_last_n_stages))
    full_ft_stage_indices = set(range(n_stages - unfreeze_last_n_stages, n_stages))

    # Inject LoRA only into stages that will be frozen
    for i, stage in enumerate(model.stage_list):
        if i in lora_stage_indices:
            for block in stage.block_list:
                block.se_fc1 = LoRALinear(block.se_fc1, rank, alpha)
                block.se_fc2 = LoRALinear(block.se_fc2, rank, alpha)

    if full_ft_stage_indices:
        logger.info(
            "Partial unfreeze: stages %s fully trainable  |  stages %s LoRA-only",
            sorted(full_ft_stage_indices), sorted(lora_stage_indices),
        )

    # Set requires_grad
    frozen = trained = 0
    for name, p in model.named_parameters():
        is_lora  = "lora_" in name
        is_dense = name.startswith("dense.")
        # Check if this param belongs to a fully-unfrozen stage
        is_full_ft = any(
            f"stage_list.{i}." in name for i in full_ft_stage_indices
        )
        p.requires_grad = is_lora or is_dense or is_full_ft
        if p.requires_grad:
            trained += p.numel()
        else:
            frozen += p.numel()

    logger.info(
        "LoRA injected — rank=%d  alpha=%.1f  "
        "trainable=%s  frozen=%s  (%.1f%% trainable)",
        rank, alpha,
        f"{trained:,}", f"{frozen:,}",
        100.0 * trained / (trained + frozen),
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _load_pretrained_state_dict(checkpoint_path: Path) -> dict:
    """Load checkpoint and extract the backbone state dict (no dense)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Handle both {state_dict: ...} and direct state-dict formats
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        sd = ckpt
    # Drop original dense layer — we replace it with a new head
    sd = {k: v for k, v in sd.items() if not k.startswith("dense.")}
    return sd


def _download_checkpoint(cache_dir: Path) -> Path:
    """Download 12_lead_ECGFounder.pth from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the ECGFounder checkpoint.\n"
            "Install it with:  pip install huggingface_hub"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Net1D._HF_FILE
    if local_path.exists():
        logger.info("ECGFounder checkpoint already cached: %s", local_path)
        return local_path

    logger.info("Downloading ECGFounder checkpoint from HuggingFace Hub...")
    downloaded = hf_hub_download(
        repo_id=Net1D._HF_REPO,
        filename=Net1D._HF_FILE,
        local_dir=str(cache_dir),
    )
    logger.info("Checkpoint saved -> %s", downloaded)
    return Path(downloaded)


def build_ecgfounder_lora(
    n_classes: int = 5,
    rank: int = 8,
    alpha: float = 16.0,
    unfreeze_last_n_stages: int = 0,
    checkpoint: Path | str | None = None,
    hf_cache_dir: Path | str = Path("checkpoints/ecgfounder"),
) -> Net1D:
    """Build ECGFounder with LoRA fine-tuning adapters.

    Downloads pretrained weights from HuggingFace Hub if ``checkpoint`` is
    not provided.  Injects LoRA into SE Linear layers of frozen stages and
    optionally fully unfreezes the last N backbone stages.  Replaces the
    classification head with a new ``Linear(1024, n_classes)``.

    Args:
        n_classes:              Number of output classes (5 for PTB-XL).
        rank:                   LoRA adapter rank.
        alpha:                  LoRA scaling factor.  scale = alpha / rank.
        unfreeze_last_n_stages: Fully unfreeze last N backbone stages (0 =
                                LoRA-only on all stages, default).
        checkpoint:             Path to local .pth checkpoint.  If None,
                                downloads from HuggingFace Hub.
        hf_cache_dir:           Directory for caching the HF download.

    Returns:
        Net1D model ready for fine-tuning on CPU (call .to(device) after).
    """
    hf_cache_dir = Path(hf_cache_dir)

    # 1. Resolve checkpoint path
    if checkpoint is None:
        checkpoint = _download_checkpoint(hf_cache_dir)
    else:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"ECGFounder checkpoint not found: {checkpoint}")

    # 2. Build model (original architecture, n_classes=5 for head replacement)
    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1.0,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=n_classes,
        use_bn=False,
        use_do=False,
    )

    # 3. Load pretrained backbone weights (strict=False, dense excluded)
    state_dict = _load_pretrained_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])
    backbone_missing = [k for k in missing if not k.startswith("dense.")]
    if backbone_missing:
        logger.warning("Missing backbone keys: %s", backbone_missing[:5])
    logger.info(
        "ECGFounder backbone loaded — %d keys  (dense excluded)",
        len(state_dict),
    )

    # 4. Fresh classification head
    model.dense = nn.Linear(1024, n_classes)
    nn.init.xavier_normal_(model.dense.weight)
    nn.init.zeros_(model.dense.bias)
    logger.info("Classification head replaced: Linear(1024, %d)", n_classes)

    # 5. Inject LoRA into SE modules (and optionally unfreeze last stages)
    inject_lora(model, rank=rank, alpha=alpha,
                unfreeze_last_n_stages=unfreeze_last_n_stages)

    total  = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "ECGFounder ready — total params: %s  trainable: %s",
        f"{total:,}", f"{trained:,}",
    )

    return model
