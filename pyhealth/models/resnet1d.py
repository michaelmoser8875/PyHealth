"""ResNet1D for binary waveform classification.

Contributor: Group 2 (individual attribution to be supplied).
Paper: Nonaka et al., Boosting Phonocardiogram Classification
Performance with Function Generated Data (ML4H 2025; PMLR 2026).
https://proceedings.mlr.press/v297/nonaka26a.html
"""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel
from .protoecgnet import ResNet1dBackbone


class ResNet1D(BaseModel):
    """Residual binary classifier over raw, channels-first waveforms.

    Uses ResNet1dBackbone with adaptive average pooling, optional dropout,
    and a one-logit head. Inputs are (B,T) or channels-first (B,C,T).

    Args:
        dataset: Dataset with fitted binary output processor.
        feature_key: Waveform field in the input schema.
        label_key: Binary target field in the output schema.
        layers: Positive block counts per stage; (2,2,2,2) is ResNet-18-style
            and (3,4,6,3) is ResNet-34-style.
        base_width: Channels in the first stage.
        kernel_size: Positive odd kernel size in residual blocks.
        in_channels: Number of waveform channels (normally one).
        dropout: Probability applied after global average pooling.
        pos_weight: Positive finite BCE weight; None uses the dataset
            processor's configured weighting.
        normalize: Apply per-sample z-score across channels and time with
            population standard deviation clamped to 1e-6.

    Examples:
        >>> model = ResNet1D(dataset)  # doctest: +SKIP
        >>> output = model(signal=torch.randn(2, 1, 80000))  # doctest: +SKIP
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "signal",
        label_key: str = "label",
        layers: Sequence[int] = (2, 2, 2, 2),
        base_width: int = 64,
        kernel_size: int = 3,
        in_channels: int = 1,
        dropout: float = 0.0,
        pos_weight: float | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(dataset)
        if feature_key not in self.feature_keys:
            raise ValueError(f"Unknown feature_key: {feature_key}")
        if self.label_keys != [label_key] or self.mode != "binary":
            raise ValueError("ResNet1D requires exactly one binary label_key")
        layers = tuple(layers)
        for name, values in (
            ("layers", layers),
            ("base_width", [base_width]),
            ("in_channels", [in_channels]),
            ("kernel_size", [kernel_size]),
        ):
            if any(type(v) is not int for v in values):
                raise TypeError(f"{name} must contain integers")
            if not values or any(v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for residual alignment")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if pos_weight is not None and (
            not math.isfinite(pos_weight) or pos_weight <= 0
        ):
            raise ValueError("pos_weight must be positive and finite")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be boolean")
        self.feature_key, self.label_key = feature_key, label_key
        self.in_channels, self.normalize = in_channels, normalize
        self.backbone = ResNet1dBackbone(
            in_channels=in_channels,
            base_width=base_width,
            layers=layers,
            kernel_size=kernel_size,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.out_channels, 1)
        self.register_buffer(
            "pos_weight",
            (
                None
                if pos_weight is None
                else torch.tensor([pos_weight], dtype=torch.float32)
            ),
        )

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score (B,T) or (B,C,T) waveforms.

        Args:
            **kwargs: Processed waveform and optional binary targets.

        Returns:
            ``logit`` and sigmoid ``y_prob``, each (B,1). With targets,
            also returns (B,1) ``y_true`` and scalar BCE ``loss``.
        """
        x = kwargs[self.feature_key].to(self.device, dtype=self.fc.weight.dtype)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected (B,{self.in_channels},T) or mono (B,T)")
        if x.shape[0] == 0 or x.shape[2] == 0:
            raise ValueError("Waveforms must have nonempty batch and time axes")
        if not torch.isfinite(x).all():
            raise ValueError("Waveforms must be finite")
        if self.normalize:
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), correction=0, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std
        logit = self.fc(self.dropout(self.pool(self.backbone(x)).flatten(1)))
        result = {"logit": logit, "y_prob": self.label_processor.activation(logit)}
        if self.label_key in kwargs:
            target = kwargs[self.label_key].to(logit.device, dtype=logit.dtype)
            if target.shape == (x.shape[0],):
                target = target.unsqueeze(1)
            if target.shape != logit.shape:
                raise ValueError("Binary targets must have shape (B,) or (B,1)")
            if (
                not torch.isfinite(target).all()
                or not ((target == 0) | (target == 1)).all()
            ):
                raise ValueError("Binary targets must contain only 0 and 1")
            options = {} if self.pos_weight is None else {"pos_weight": self.pos_weight}
            result.update(
                y_true=target, loss=self.label_processor.loss(logit, target, **options)
            )
        return result

    def load_pretrained(
        self, state_dict: Mapping[str, torch.Tensor], retain_head: bool = True
    ) -> None:
        """Load pretrained weights while preserving target-task loss weighting.

        Args:
            state_dict: Weights loaded with torch.load(weights_only=True).
            retain_head: Load classifier weights; False keeps the newly
                initialized classifier. Target-task loss weighting is retained.

        Raises:
            RuntimeError: If backbone keys or dimensions are incompatible.
        """
        excluded = {"pos_weight"}
        if not retain_head:
            excluded.update({"fc.weight", "fc.bias"})
        weights = {k: v for k, v in state_dict.items() if k not in excluded}
        current = self.state_dict()
        for key in excluded:
            if key in current:
                weights[key] = current[key]
        self.load_state_dict(weights, strict=True)
