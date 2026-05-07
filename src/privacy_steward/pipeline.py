"""Custom NER pipeline for openai/privacy-filter using native PyTorch implementation."""

from __future__ import annotations
import dataclasses
import functools
import json
import math
import os
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import tiktoken
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.nn import functional
from privacy_steward.models import EntitySpan


DEFAULT_MODEL = "openai/privacy-filter"
PRIVACY_FILTER_MODEL_TYPE: Final[str] = "privacy_filter"
REQUIRED_MODEL_CONFIG_KEYS: Final[tuple[str, ...]] = (
    "model_type",
    "encoding",
    "num_hidden_layers",
    "num_experts",
    "experts_per_token",
    "vocab_size",
    "num_labels",
    "hidden_size",
    "intermediate_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "sliding_window",
    "bidirectional_context",
    "bidirectional_left_context",
    "bidirectional_right_context",
    "default_n_ctx",
    "initial_context_length",
    "rope_theta",
    "rope_scaling_factor",
    "rope_ntk_alpha",
    "rope_ntk_beta",
    "param_dtype",
)
BACKGROUND_CLASS_LABEL: Final[str] = "O"
BOUNDARY_PREFIXES: Final[tuple[str, ...]] = ("B", "I", "E", "S")
SPAN_CLASS_NAMES: Final[tuple[str, ...]] = (
    BACKGROUND_CLASS_LABEL,
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)
NER_CLASS_NAMES: Final[tuple[str, ...]] = (BACKGROUND_CLASS_LABEL,) + tuple(
    f"{prefix}-{base_label}"
    for base_label in SPAN_CLASS_NAMES
    if base_label != BACKGROUND_CLASS_LABEL
    for prefix in BOUNDARY_PREFIXES
)
VITERBI_TRANSITION_BIAS_KEYS: Final[tuple[str, ...]] = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)
DEFAULT_VITERBI_CALIBRATION_PRESET: Final[str] = "default"


@functools.lru_cache(maxsize=8)
def _get_model_dir(model_id: str = DEFAULT_MODEL) -> Path:
    model_root = snapshot_download(model_id, allow_patterns=["original/*"])
    return Path(model_root) / "original"


def _require_int(
    value: object,
    *,
    context: str,
    field: str,
) -> int:
    """Return *value* as an int or raise a context-rich validation error."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} {field} must be an integer, got {value!r}")
    return value


def _require_float(
    value: object,
    *,
    context: str,
    field: str,
) -> float:
    """Return *value* as a float or raise a context-rich validation error."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} {field} must be a number, got {value!r}")
    return float(value)


def _require_nonempty_string(
    value: object,
    *,
    context: str,
    field: str,
) -> str:
    """Return *value* as a non-empty string or raise a validation error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} {field} must be a non-empty string")
    return value


def _close_open_span(
    spans: list[tuple[int, int, int]],
    current_label: int | None,
    start_idx: int | None,
    previous_idx: int | None,
) -> tuple[int | None, int | None]:
    """Close the active span when all required state is available."""
    if current_label is not None and start_idx is not None and previous_idx is not None:
        spans.append((current_label, start_idx, previous_idx + 1))
    return None, None


def _handle_span_boundary(
    spans: list[tuple[int, int, int]],
    *,
    boundary_tag: str | None,
    span_label: int,
    token_idx: int,
    current_label: int | None,
    start_idx: int | None,
    previous_idx: int | None,
) -> tuple[int | None, int | None]:
    """Update open-span state for a non-background token boundary tag."""
    next_label = current_label
    next_start = start_idx
    if boundary_tag == "S":
        next_label, next_start = _close_open_span(
            spans,
            next_label,
            next_start,
            previous_idx,
        )
        spans.append((span_label, token_idx, token_idx + 1))
        next_label, next_start = None, None
    elif boundary_tag == "B":
        next_label, next_start = _close_open_span(
            spans,
            next_label,
            next_start,
            previous_idx,
        )
        next_label, next_start = span_label, token_idx
    elif boundary_tag == "I":
        if current_label is None or current_label != span_label:
            next_label, next_start = _close_open_span(
                spans,
                next_label,
                next_start,
                previous_idx,
            )
            next_label, next_start = span_label, token_idx
    elif boundary_tag == "E":
        if current_label is None or current_label != span_label or start_idx is None:
            next_label, next_start = _close_open_span(
                spans,
                next_label,
                next_start,
                previous_idx,
            )
            spans.append((span_label, token_idx, token_idx + 1))
        else:
            spans.append((current_label, start_idx, token_idx + 1))
        next_label, next_start = None, None
    else:
        next_label, next_start = _close_open_span(
            spans,
            next_label,
            next_start,
            previous_idx,
        )
    return next_label, next_start


def validate_model_config_contract(
    checkpoint_config: dict[str, object],
    *,
    context: str,
) -> None:
    """Validate the raw checkpoint config before constructing model objects."""
    missing = [
        key for key in REQUIRED_MODEL_CONFIG_KEYS if key not in checkpoint_config
    ]
    if missing:
        raise ValueError(
            f"{context} is missing required model config keys: {', '.join(missing)}"
        )
    if checkpoint_config.get("model_type") != PRIVACY_FILTER_MODEL_TYPE:
        raise ValueError(
            f"{context} model_type must be {PRIVACY_FILTER_MODEL_TYPE!r}, got "
            f"{checkpoint_config.get('model_type')!r}"
        )
    if checkpoint_config.get("bidirectional_context") is not True:
        raise ValueError(f"{context} must use bidirectional_context=true")

    left_context = _require_int(
        checkpoint_config.get("bidirectional_left_context"),
        context=context,
        field="bidirectional_left_context",
    )
    right_context = _require_int(
        checkpoint_config.get("bidirectional_right_context"),
        context=context,
        field="bidirectional_right_context",
    )
    if left_context < 0 or right_context < 0:
        raise ValueError(
            f"{context} bidirectional context sizes must be >= 0 "
            f"(got {left_context}/{right_context})"
        )
    if left_context != right_context:
        raise ValueError(
            f"{context} bidirectional context must be symmetric "
            f"(got left={left_context}, right={right_context})"
        )

    sliding_window = _require_int(
        checkpoint_config.get("sliding_window"),
        context=context,
        field="sliding_window",
    )
    expected_sliding_window = 2 * left_context + 1
    if sliding_window != expected_sliding_window:
        raise ValueError(
            f"{context} sliding_window must equal 2 * bidirectional context + 1 "
            f"(got {sliding_window}, expected {expected_sliding_window})"
        )

    num_labels = _require_int(
        checkpoint_config["num_labels"],
        context=context,
        field="num_labels",
    )
    if num_labels != 33:
        raise ValueError(
            f"{context} must use num_labels=33 for the label space, got {num_labels}"
        )

    _require_nonempty_string(
        checkpoint_config["encoding"],
        context=context,
        field="encoding",
    )

    n_ctx = _require_int(
        checkpoint_config["default_n_ctx"],
        context=context,
        field="default_n_ctx",
    )
    if n_ctx <= 0:
        raise ValueError(f"{context} default_n_ctx must be positive, got {n_ctx}")

    raw_param_dtype = checkpoint_config["param_dtype"]
    if raw_param_dtype != "bfloat16":
        raise ValueError(
            f"{context} param_dtype must be bfloat16, got {raw_param_dtype!r}"
        )


def expert_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a batched expert linear projection."""
    num_rows, experts, k_dim = x.shape
    _, _, _, out_dim = weight.shape
    x_bmm = x.reshape(num_rows * experts, 1, k_dim)
    w_bmm = weight.reshape(num_rows * experts, k_dim, out_dim)
    out = torch.bmm(x_bmm, w_bmm).reshape(num_rows, experts, out_dim)
    if bias is not None:
        out = out + bias
    return out


@dataclass
class ModelConfig:
    """Structured model hyperparameters extracted from checkpoint config."""

    num_hidden_layers: int
    num_experts: int
    experts_per_token: int
    vocab_size: int
    num_labels: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    bidirectional_context_size: int
    initial_context_length: int
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float

    @classmethod
    def from_checkpoint_config(
        cls,
        checkpoint_config: dict[str, object],
        *,
        context: str,
    ) -> ModelConfig:
        """Build a validated config object from the raw checkpoint payload."""
        checkpoint_config = dict(checkpoint_config)
        checkpoint_config["bidirectional_context_size"] = checkpoint_config[
            "bidirectional_left_context"
        ]
        fields = {field.name: field for field in dataclasses.fields(cls)}
        config_values = {
            key: value for key, value in checkpoint_config.items() if key in fields
        }

        missing = [
            name
            for name, field in fields.items()
            if field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
            and name not in config_values
        ]
        if missing:
            raise ValueError(
                f"{context} is missing required model config fields: "
                f"{', '.join(missing)}"
            )

        num_hidden_layers = _require_int(
            config_values["num_hidden_layers"],
            context=context,
            field="num_hidden_layers",
        )
        num_experts = _require_int(
            config_values["num_experts"], context=context, field="num_experts"
        )
        experts_per_token = _require_int(
            config_values["experts_per_token"],
            context=context,
            field="experts_per_token",
        )
        vocab_size = _require_int(
            config_values["vocab_size"], context=context, field="vocab_size"
        )
        num_labels = _require_int(
            config_values["num_labels"], context=context, field="num_labels"
        )
        hidden_size = _require_int(
            config_values["hidden_size"], context=context, field="hidden_size"
        )
        intermediate_size = _require_int(
            config_values["intermediate_size"],
            context=context,
            field="intermediate_size",
        )
        head_dim = _require_int(
            config_values["head_dim"], context=context, field="head_dim"
        )
        num_attention_heads = _require_int(
            config_values["num_attention_heads"],
            context=context,
            field="num_attention_heads",
        )
        num_key_value_heads = _require_int(
            config_values["num_key_value_heads"],
            context=context,
            field="num_key_value_heads",
        )
        bidirectional_context_size = _require_int(
            config_values["bidirectional_context_size"],
            context=context,
            field="bidirectional_context_size",
        )
        initial_context_length = _require_int(
            config_values["initial_context_length"],
            context=context,
            field="initial_context_length",
        )
        rope_theta = _require_float(
            config_values["rope_theta"], context=context, field="rope_theta"
        )
        rope_scaling_factor = _require_float(
            config_values["rope_scaling_factor"],
            context=context,
            field="rope_scaling_factor",
        )
        rope_ntk_alpha = _require_float(
            config_values["rope_ntk_alpha"], context=context, field="rope_ntk_alpha"
        )
        rope_ntk_beta = _require_float(
            config_values["rope_ntk_beta"], context=context, field="rope_ntk_beta"
        )

        try:
            return cls(
                num_hidden_layers=num_hidden_layers,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                vocab_size=vocab_size,
                num_labels=num_labels,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                bidirectional_context_size=bidirectional_context_size,
                initial_context_length=initial_context_length,
                rope_theta=rope_theta,
                rope_scaling_factor=rope_scaling_factor,
                rope_ntk_alpha=rope_ntk_alpha,
                rope_ntk_beta=rope_ntk_beta,
            )
        except TypeError as exc:
            raise ValueError(
                f"Invalid model config payload at {context}: {exc}"
            ) from exc


class RMSNorm(torch.nn.Module):
    """Root-mean-square normalization with learned scale."""

    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ) -> None:
        """Create the normalization layer."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of *x*."""
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(x.dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to paired features."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.stack((out1, out2), dim=-1).reshape(x.shape)


class RotaryEmbedding(torch.nn.Module):
    """Cache rotary embedding coefficients for a given head dimension."""

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        *,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        """Create rotary caches sized for the requested context window."""
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        max_positions = int(self.initial_context_length * self.scaling_factor)
        max_positions = max(max_positions, self.initial_context_length)
        self.max_position_embeddings = max_positions
        cos, sin = self._compute_cos_sin(
            self.max_position_embeddings, device=torch.device("cpu")
        )
        target_device = device or torch.device("cpu")
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("cos_cache", cos.to(target_device), persistent=False)
        self.register_buffer("sin_cache", sin.to(target_device), persistent=False)

    def _compute_concentration_and_inv_freq(
        self, device: torch.device | None = None
    ) -> tuple[float, torch.Tensor]:
        device = device or self.device
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        return concentration, inv_freq

    def _compute_cos_sin(
        self, num_tokens: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concentration, inv_freq = self._compute_concentration_and_inv_freq(
            device=device
        )
        device = device or self.device
        t = torch.arange(num_tokens, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos.to(self.dtype), sin.to(self.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate query and key tensors in-place shape-preserving form."""
        num_tokens = query.shape[0]
        cos_cache: torch.Tensor
        sin_cache: torch.Tensor
        if num_tokens > self.cos_cache.shape[0]:
            cos, sin = self._compute_cos_sin(num_tokens, device=torch.device("cpu"))
            self.register_buffer("cos_cache", cos.to(query.device), persistent=False)
            self.register_buffer("sin_cache", sin.to(query.device), persistent=False)
        if self.cos_cache.device != query.device:
            cos_cache = self.cos_cache.to(query.device)
            sin_cache = self.sin_cache.to(query.device)
        else:
            cos_cache = self.cos_cache
            sin_cache = self.sin_cache
        cos = cos_cache[:num_tokens]
        sin = sin_cache[:num_tokens]

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = apply_rope(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = apply_rope(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    context_size: int,
) -> torch.Tensor:
    """Scaled dot-product attention with a bidirectional local window."""
    num_tokens, num_heads, q_mult, head_dim = q.shape
    window = 2 * context_size + 1
    padded_k = functional.pad(k, (0, 0, 0, 0, context_size, context_size))
    padded_v = functional.pad(v, (0, 0, 0, 0, context_size, context_size))
    k_window = padded_k.unfold(0, window, 1).permute(0, 3, 1, 2)
    v_window = padded_v.unfold(0, window, 1).permute(0, 3, 1, 2)
    idx = torch.arange(window, device=q.device) - context_size
    pos = torch.arange(num_tokens, device=q.device)[:, None] + idx[None, :]
    valid = (pos >= 0) & (pos < num_tokens)
    scores = torch.einsum("nhqd,nwhd->nhqw", q, k_window).float()
    scores *= sm_scale
    scores = scores.masked_fill(~valid[:, None, None, :], -float("inf"))
    sink_scores = (sinks * math.log(2.0)).reshape(num_heads, q_mult)
    sink_scores = sink_scores[None, :, :, None].expand(num_tokens, -1, -1, 1)
    scores = torch.cat([scores, sink_scores], dim=-1)
    weights = torch.softmax(scores, dim=-1)[..., :-1].to(v.dtype)
    attn = torch.einsum("nhqw,nwhd->nhqd", weights, v_window)
    return attn.reshape(num_tokens, -1)


class AttentionBlock(torch.nn.Module):
    """Attention block with rotary embeddings and local bidirectional context."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ) -> None:
        """Construct the attention projection and output layers."""
        super().__init__()
        param_dtype = torch.bfloat16
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.bidirectional_context_size = int(config.bidirectional_context_size)
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.float32)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=param_dtype
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=param_dtype,
        )
        self.qk_scale = 1 / math.sqrt(math.sqrt(config.head_dim))
        self.sm_scale = 1.0
        self.rope = RotaryEmbedding(
            config.head_dim,
            int(config.rope_theta),
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention and add the residual connection."""
        t = self.norm(x)
        if t.dtype != self.qkv.weight.dtype:
            t = t.to(self.qkv.weight.dtype)
        qkv = functional.linear(t, self.qkv.weight, self.qkv.bias)
        query = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        key = qkv[
            :,
            self.num_attention_heads * self.head_dim : (
                self.num_attention_heads + self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()
        value = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (
                self.num_attention_heads + 2 * self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()

        query, key = self.rope(query, key)
        query = query * self.qk_scale
        key = key * self.qk_scale
        sinks = self.sinks
        num_tokens = query.shape[0]
        query = query.view(
            num_tokens,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        key = key.view(num_tokens, self.num_key_value_heads, self.head_dim)
        value = value.view(num_tokens, self.num_key_value_heads, self.head_dim)
        attn_out = sdpa(
            query,
            key,
            value,
            sinks,
            self.sm_scale,
            self.bidirectional_context_size,
        )
        if attn_out.dtype != self.out.weight.dtype:
            attn_out = attn_out.to(self.out.weight.dtype)
        proj_bias = self.out.bias
        proj = functional.linear(attn_out, self.out.weight, proj_bias)
        return x + proj.to(x.dtype)


def swiglu(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Apply the SwiGLU activation used by the expert MLP."""
    x_glu, x_linear = x.chunk(2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    """Mixture-of-experts feed-forward block."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ) -> None:
        """Allocate gating and expert weights for the MLP block."""
        super().__init__()
        param_dtype = torch.bfloat16
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = 7.0
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=param_dtype
        )
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size, config.intermediate_size * 2),
                device=device,
                dtype=param_dtype,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=device,
                dtype=param_dtype,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size, config.hidden_size),
                device=device,
                dtype=param_dtype,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=param_dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated expert MLP and add the residual connection."""
        t = self.norm(x)
        gate_scores = functional.linear(
            t.float(), self.gate.weight.float(), self.gate.bias.float()
        )
        experts = torch.topk(gate_scores, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.softmax(experts.values, dim=-1) / self.experts_per_token

        expert_indices = experts.indices
        experts_per_token_eff = self.experts_per_token

        def _moe_chunk(
            t_chunk: torch.Tensor,
            expert_indices_chunk: torch.Tensor,
            expert_weights_chunk: torch.Tensor,
        ) -> torch.Tensor:
            mlp1_weight = self.mlp1_weight[expert_indices_chunk].float()
            mlp1_bias = self.mlp1_bias[expert_indices_chunk].float()
            t_expanded = (
                t_chunk.float()
                .unsqueeze(1)
                .expand(-1, expert_indices_chunk.shape[1], -1)
            )
            out = expert_linear(
                t_expanded,
                mlp1_weight,
                mlp1_bias,
            )
            out = swiglu(out, limit=self.swiglu_limit)
            mlp2_weight = self.mlp2_weight[expert_indices_chunk].float()
            mlp2_bias = self.mlp2_bias[expert_indices_chunk].float()
            out = expert_linear(
                out.float(),
                mlp2_weight,
                mlp2_bias,
            )
            if out.dtype != expert_weights_chunk.dtype:
                out = out.to(expert_weights_chunk.dtype)
            out = torch.einsum("bec,be->bc", out, expert_weights_chunk)
            out = out * experts_per_token_eff
            return out.to(x.dtype)

        torch_ops_chunk_size = 32
        if t.shape[0] > torch_ops_chunk_size:
            chunks = []
            for start in range(0, t.shape[0], torch_ops_chunk_size):
                end = start + torch_ops_chunk_size
                chunks.append(
                    _moe_chunk(
                        t[start:end],
                        expert_indices[start:end],
                        expert_weights[start:end],
                    )
                )
            t = torch.cat(chunks, dim=0)
        else:
            t = _moe_chunk(t, expert_indices, expert_weights)
        return x + t


class TransformerBlock(torch.nn.Module):
    """Single transformer block composed of attention and MLP sublayers."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ) -> None:
        """Construct the attention and MLP submodules."""
        super().__init__()
        self.attn = AttentionBlock(config, device=device)
        self.mlp = MLPBlock(config, device=device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention followed by the MLP."""
        x = self.attn(x)
        return self.mlp(x)


class Checkpoint:
    """Checkpoint shard reader for the model weights."""

    @staticmethod
    def build_param_name_map(
        num_hidden_layers: int,
    ) -> dict[str, str]:
        """Map model parameter names to checkpoint tensor names."""
        return (
            {
                f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.swiglu.bias"
                for n in range(num_hidden_layers)
            }
            | {
                f"block.{n}.mlp.mlp1_weight": f"block.{n}.mlp.swiglu.weight"
                for n in range(num_hidden_layers)
            }
            | {
                f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.out.bias"
                for n in range(num_hidden_layers)
            }
            | {
                f"block.{n}.mlp.mlp2_weight": f"block.{n}.mlp.out.weight"
                for n in range(num_hidden_layers)
            }
        )

    def __init__(self, path: str, device: torch.device, num_hidden_layers: int) -> None:
        """Index all checkpoint shards and verify tensor name uniqueness."""
        self.param_name_map = self.build_param_name_map(num_hidden_layers)
        self.device_str = (
            device.type if device.index is None else f"{device.type}:{device.index}"
        )
        safetensor_files = [
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".safetensors")
        ]
        tensor_name_to_file: dict[str, str] = {}
        for safetensor_file in safetensor_files:
            with safe_open(
                safetensor_file, framework="pt", device=self.device_str
            ) as handle:
                for key in handle.keys():
                    prior_file = tensor_name_to_file.get(key)
                    if prior_file is not None:
                        raise ValueError(
                            "Duplicate tensor name in checkpoint shards: "
                            f"{key!r} appears in {prior_file!r} and {safetensor_file!r}"
                        )
                    tensor_name_to_file[key] = safetensor_file
        self.tensor_name_to_file = tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        """Return the checkpoint tensor mapped to *name*."""
        mapped = self.param_name_map.get(name, name)
        return self._get_tensor(mapped)

    def _get_tensor(self, name: str) -> torch.Tensor:
        if name not in self.tensor_name_to_file:
            raise KeyError(f"Tensor {name!r} not found in checkpoint")
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as handle:
            return handle.get_tensor(name)


class Transformer(torch.nn.Module):
    """Full transformer model used for entity classification."""

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        """Construct the model from a validated configuration."""
        super().__init__()
        param_dtype = torch.bfloat16
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=param_dtype
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, device=device)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=False,
            device=device,
            dtype=param_dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-token logits for the provided token ids."""
        x = self.embedding(token_ids)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = functional.linear(x, self.unembedding.weight, None)
        return x

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        *,
        device: torch.device,
    ) -> Transformer:
        """Load a transformer from a checkpoint directory."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        config_path = Path(checkpoint_dir) / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            checkpoint_config = json.load(handle)
        if not isinstance(checkpoint_config, dict):
            raise ValueError(f"Invalid checkpoint config payload at {config_path}")
        validate_model_config_contract(
            checkpoint_config,
            context=str(config_path),
        )

        config = ModelConfig.from_checkpoint_config(
            checkpoint_config,
            context=str(config_path),
        )
        checkpoint = Checkpoint(
            checkpoint_dir,
            device,
            num_hidden_layers=config.num_hidden_layers,
        )

        model = cls(config=config, device=device)
        model.eval()

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)
            if param.data.shape != loaded_tensor.shape:
                raise ValueError(
                    f"Tensor shape mismatch for {name!r}: expected "
                    f"{tuple(param.data.shape)}, got {tuple(loaded_tensor.shape)}"
                )
            param.data.copy_(loaded_tensor)

        return model


@dataclass(frozen=True)
class LabelInfo:
    """Lookup tables describing how token labels map to spans."""

    boundary_label_lookup: dict[str, dict[str, int]]
    token_to_span_label: dict[int, int]
    token_boundary_tags: dict[int, str | None]
    span_class_names: tuple[str, ...]
    span_label_lookup: dict[str, int]
    background_token_label: int
    background_span_label: int


def labels_to_spans(
    labels_by_index: dict[int, int], label_info: LabelInfo
) -> list[tuple[int, int, int]]:
    """Convert token labels into contiguous span tuples."""
    spans: list[tuple[int, int, int]] = []
    current_label: int | None = None
    start_idx: int | None = None
    previous_idx: int | None = None
    background_span_label = label_info.background_span_label

    for token_idx in sorted(labels_by_index):
        label_id = labels_by_index[token_idx]
        span_label = label_info.token_to_span_label.get(label_id)
        boundary_tag = label_info.token_boundary_tags.get(label_id)

        if previous_idx is not None and token_idx != previous_idx + 1:
            current_label, start_idx = _close_open_span(
                spans,
                current_label,
                start_idx,
                previous_idx,
            )

        if span_label is None:
            previous_idx = token_idx
            continue

        if span_label == background_span_label:
            if current_label is not None and start_idx is not None:
                spans.append((current_label, start_idx, token_idx))
            current_label, start_idx = None, None
            previous_idx = token_idx
            continue

        current_label, start_idx = _handle_span_boundary(
            spans,
            boundary_tag=boundary_tag,
            span_label=span_label,
            token_idx=token_idx,
            current_label=current_label,
            start_idx=start_idx,
            previous_idx=previous_idx,
        )

        previous_idx = token_idx

    if current_label is not None and start_idx is not None and previous_idx is not None:
        spans.append((current_label, start_idx, previous_idx + 1))
    return spans


def token_spans_to_char_spans(
    spans: Sequence[tuple[int, int, int]],
    char_starts: Sequence[int],
    char_ends: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Map token spans to character spans using precomputed offsets."""
    converted: list[tuple[int, int, int]] = []
    for label_idx, token_start, token_end in spans:
        if not (0 <= token_start < token_end <= len(char_starts)):
            continue
        char_start = char_starts[token_start]
        char_end = char_ends[token_end - 1]
        if char_end <= char_start:
            continue
        converted.append((label_idx, char_start, char_end))
    return converted


def trim_char_spans_whitespace(
    spans: Sequence[tuple[int, int, int]],
    text: str,
) -> list[tuple[int, int, int]]:
    """Strip leading and trailing whitespace from character spans."""
    trimmed: list[tuple[int, int, int]] = []
    for label_idx, raw_start, raw_end in spans:
        if not (0 <= raw_start < raw_end <= len(text)):
            continue
        start = raw_start
        end = raw_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((label_idx, start, end))
    return trimmed


@dataclass(frozen=True)
class InferenceRuntime:
    """Runtime bundle containing the loaded model and tokenization state."""

    model: Transformer
    encoding: tiktoken.Encoding
    label_info: LabelInfo
    device: torch.device
    n_ctx: int
    bidirectional_context_size: int


@functools.lru_cache(maxsize=8)
def get_viterbi_transition_biases(model_id: str = DEFAULT_MODEL) -> dict[str, float]:
    """Load Viterbi transition biases from the calibration file if present."""
    calibration_path = _get_model_dir(model_id) / "viterbi_calibration.json"
    default_biases = {key: 0.0 for key in VITERBI_TRANSITION_BIAS_KEYS}
    if not calibration_path.is_file():
        return default_biases

    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Viterbi calibration payload at {calibration_path}")

    raw_biases: object = payload
    operating_points = payload.get("operating_points")
    if operating_points is not None:
        if not isinstance(operating_points, dict):
            raise ValueError(f"Invalid operating_points payload at {calibration_path}")
        preset_entry = operating_points.get(DEFAULT_VITERBI_CALIBRATION_PRESET)
        if not isinstance(preset_entry, dict):
            raise ValueError(
                f"Missing operating_points.{DEFAULT_VITERBI_CALIBRATION_PRESET!s} "
                f"in {calibration_path}"
            )
        raw_biases = preset_entry.get("biases")

    if not isinstance(raw_biases, dict):
        raise ValueError(f"Invalid Viterbi bias payload at {calibration_path}")

    resolved_biases: dict[str, float] = {}
    for key in VITERBI_TRANSITION_BIAS_KEYS:
        raw_value = raw_biases.get(key)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(f"Missing or invalid {key!r} in {calibration_path}")
        resolved_biases[key] = float(raw_value)
    return resolved_biases


@functools.lru_cache(maxsize=8)
def get_runtime(model_id: str = DEFAULT_MODEL) -> InferenceRuntime:
    """Load and cache the model, tokenizer, and label metadata."""
    checkpoint = _get_model_dir(model_id)
    if not checkpoint.exists() or not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    if not any(checkpoint.glob("*.safetensors")):
        raise FileNotFoundError(
            f"Checkpoint directory has no .safetensors files: {checkpoint}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = checkpoint / "config.json"
    checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Invalid checkpoint config payload at {config_path}")
    validate_model_config_contract(
        checkpoint_config,
        context=str(config_path),
    )
    ner_class_names = NER_CLASS_NAMES
    n_ctx = int(checkpoint_config["default_n_ctx"])
    bidirectional_context_size = int(checkpoint_config["bidirectional_left_context"])

    encoding = tiktoken.get_encoding(str(checkpoint_config["encoding"]).strip())
    span_class_names: list[str] = [BACKGROUND_CLASS_LABEL]
    span_label_lookup: dict[str, int] = {BACKGROUND_CLASS_LABEL: 0}
    boundary_label_lookup: dict[str, dict[str, int]] = {}
    token_to_span_label: dict[int, int] = {}
    token_boundary_tags: dict[int, str | None] = {}
    background_idx: int | None = None
    for idx, name in enumerate(ner_class_names):
        if name == BACKGROUND_CLASS_LABEL:
            background_idx = idx
            token_to_span_label[idx] = span_label_lookup[BACKGROUND_CLASS_LABEL]
            token_boundary_tags[idx] = None
            continue
        boundary, base_label = name.split("-", 1)
        span_idx = span_label_lookup.get(base_label)
        if span_idx is None:
            span_idx = len(span_class_names)
            span_class_names.append(base_label)
            span_label_lookup[base_label] = span_idx
        token_to_span_label[idx] = span_idx
        token_boundary_tags[idx] = boundary
        boundary_label_lookup.setdefault(base_label, {})[boundary] = idx
    if background_idx is None:
        raise ValueError("Class names must include background label 'O'")
    for base_label, mapping in boundary_label_lookup.items():
        missing = set(BOUNDARY_PREFIXES) - set(mapping)
        if missing:
            raise ValueError(
                f"Missing boundary classes {sorted(missing)} "
                f"for base label {base_label}"
            )
    label_info = LabelInfo(
        boundary_label_lookup={
            key: dict(value) for key, value in boundary_label_lookup.items()
        },
        token_to_span_label=dict(token_to_span_label),
        token_boundary_tags=dict(token_boundary_tags),
        span_class_names=tuple(span_class_names),
        span_label_lookup=dict(span_label_lookup),
        background_token_label=background_idx,
        background_span_label=span_label_lookup[BACKGROUND_CLASS_LABEL],
    )
    model = Transformer.from_checkpoint(str(checkpoint), device=device)
    return InferenceRuntime(
        model=model,
        encoding=encoding,
        label_info=label_info,
        device=device,
        n_ctx=n_ctx,
        bidirectional_context_size=bidirectional_context_size,
    )


class Decoder:
    """Viterbi decoder for token-classification logits."""

    def __init__(
        self,
        label_info: LabelInfo,
        model_id: str = DEFAULT_MODEL,
    ) -> None:
        """Precompute transition scores for the valid label lattice."""
        self.label_info = label_info
        num_classes = len(label_info.token_to_span_label)
        self._start_scores = torch.full((num_classes,), -1e9, dtype=torch.float32)
        self._end_scores = torch.full((num_classes,), -1e9, dtype=torch.float32)
        self._transition_scores = torch.full(
            (num_classes, num_classes), -1e9, dtype=torch.float32
        )
        transition_biases = get_viterbi_transition_biases(model_id)

        background_token_idx = label_info.background_token_label
        background_span_idx = label_info.background_span_label
        token_boundary_tags = label_info.token_boundary_tags
        token_to_span_label = label_info.token_to_span_label

        for idx in range(num_classes):
            tag = token_boundary_tags.get(idx)
            span_label = token_to_span_label.get(idx)
            if tag in {"B", "S"} or idx == background_token_idx:
                self._start_scores[idx] = 0.0
            if tag in {"E", "S"} or idx == background_token_idx:
                self._end_scores[idx] = 0.0

            for next_idx in range(num_classes):
                next_tag = token_boundary_tags.get(next_idx)
                next_span_label = token_to_span_label.get(next_idx)
                if self._is_valid_transition(
                    prev_tag=tag,
                    prev_span=span_label,
                    next_tag=next_tag,
                    next_span=next_span_label,
                    background_token_idx=background_token_idx,
                    background_span_idx=background_span_idx,
                    next_idx=next_idx,
                ):
                    self._transition_scores[idx, next_idx] = self._transition_bias(
                        prev_tag=tag,
                        prev_span=span_label,
                        next_tag=next_tag,
                        next_span=next_span_label,
                        background_span_idx=background_span_idx,
                        biases=transition_biases,
                    )

    @staticmethod
    def _is_valid_transition(
        *,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        background_token_idx: int,
        background_span_idx: int,
        next_idx: int,
    ) -> bool:
        next_is_background = (
            next_span == background_span_idx or next_idx == background_token_idx
        )
        if (next_span is None or next_tag is None) and not next_is_background:
            return False

        if prev_span is None or prev_tag is None:
            return next_is_background or next_tag in {"B", "S"}

        prev_is_background = prev_span == background_span_idx
        if prev_is_background or prev_tag in {"E", "S"}:
            return next_is_background or next_tag in {"B", "S"}
        if prev_tag in {"B", "I"}:
            return prev_span == next_span and next_tag in {"I", "E"}
        return False

    @staticmethod
    def _transition_bias(
        *,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        background_span_idx: int,
        biases: dict[str, float],
    ) -> float:
        next_is_background = next_span == background_span_idx
        prev_is_background = prev_span == background_span_idx
        if prev_is_background:
            return (
                biases["transition_bias_background_stay"]
                if next_is_background
                else biases["transition_bias_background_to_start"]
            )
        if prev_tag in {"B", "I"}:
            return (
                biases["transition_bias_inside_to_continue"]
                if next_tag == "I"
                else biases["transition_bias_inside_to_end"]
            )
        return (
            biases["transition_bias_end_to_background"]
            if next_is_background
            else biases["transition_bias_end_to_start"]
        )

    def decode(self, token_logprobs: torch.Tensor) -> list[int]:
        """Decode the best label path for the provided token log-probabilities."""
        if token_logprobs.ndim != 2:
            raise ValueError("token_logprobs must have shape [seq_len, num_classes]")
        seq_len, num_classes = token_logprobs.shape
        if seq_len == 0:
            return []

        start_scores = self._start_scores.to(
            device=token_logprobs.device,
            dtype=token_logprobs.dtype,
        )
        end_scores = self._end_scores.to(
            device=token_logprobs.device,
            dtype=token_logprobs.dtype,
        )
        transition_scores = self._transition_scores.to(
            device=token_logprobs.device,
            dtype=token_logprobs.dtype,
        )
        scores = token_logprobs[0] + start_scores
        backpointers = torch.empty(
            (seq_len - 1, num_classes),
            device=token_logprobs.device,
            dtype=torch.int64,
        )

        for idx in range(1, seq_len):
            transitions = scores.unsqueeze(1) + transition_scores
            best_scores, best_paths = transitions.max(dim=0)
            scores = best_scores + token_logprobs[idx]
            backpointers[idx - 1] = best_paths

        if not torch.isfinite(scores).any():
            return token_logprobs.argmax(dim=1).tolist()

        scores = scores + end_scores
        last_label = scores.argmax()
        path = torch.empty((seq_len,), device=token_logprobs.device, dtype=torch.int64)
        path[-1] = last_label
        for idx in range(seq_len - 2, -1, -1):
            last_label = backpointers[idx, last_label]
            path[idx] = last_label
        return path.tolist()


def _collect_token_score_vectors(
    runtime: InferenceRuntime,
    token_ids: tuple[int, ...],
) -> list[torch.Tensor]:
    """Compute per-token log-probability vectors for the full text.

    Long inputs are evaluated with overlapping windows so tokens near a chunk
    boundary retain bidirectional context from neighbouring tokens. Tokens
    covered by multiple windows receive the mean of their log-probability
    vectors.
    """
    total_tokens = len(token_ids)
    if total_tokens == 0:
        return []

    context_size = max(0, int(runtime.bidirectional_context_size))
    stride = runtime.n_ctx - context_size
    if stride <= 0:
        stride = runtime.n_ctx

    score_sums: list[torch.Tensor | None] = [None] * total_tokens
    counts = [0] * total_tokens
    start = 0
    while start < total_tokens:
        end = min(start + runtime.n_ctx, total_tokens)
        window_tokens = torch.tensor(
            token_ids[start:end], device=runtime.device, dtype=torch.int32
        )
        logits = runtime.model(window_tokens)
        log_probs = functional.log_softmax(logits.float(), dim=-1)
        if log_probs.shape[0] != window_tokens.shape[0]:
            raise ValueError("Logprob output length does not match window length")

        for offset, log_prob in enumerate(log_probs.unbind(0)):
            token_idx = start + offset
            current = score_sums[token_idx]
            score_sums[token_idx] = log_prob if current is None else current + log_prob
            counts[token_idx] += 1

        if end == total_tokens:
            break
        start += stride

    token_score_vectors: list[torch.Tensor] = []
    for score_sum, count in zip(score_sums, counts, strict=True):
        if score_sum is None or count == 0:
            raise ValueError("Missing logprob output for token")
        token_score_vectors.append(score_sum / count)
    return token_score_vectors


def _decode_token_bytes(
    runtime: InferenceRuntime,
    token_ids: tuple[int, ...],
) -> tuple[list[bytes], str]:
    """Decode token bytes and reconstruct the UTF-8 text stream."""
    token_bytes = [
        runtime.encoding.decode_single_token_bytes(token_id) for token_id in token_ids
    ]
    decoded_text = b"".join(token_bytes).decode("utf-8", errors="replace")
    return token_bytes, decoded_text


def _compute_char_byte_offsets(decoded_text: str) -> tuple[list[int], list[int]]:
    """Compute per-character byte offsets for a decoded UTF-8 string."""
    char_byte_starts: list[int] = []
    char_byte_ends: list[int] = []
    byte_cursor = 0
    for ch in decoded_text:
        char_byte_starts.append(byte_cursor)
        byte_cursor += len(ch.encode("utf-8"))
        char_byte_ends.append(byte_cursor)
    return char_byte_starts, char_byte_ends


def _token_byte_spans_to_char_spans(
    token_bytes: Sequence[bytes],
    char_byte_starts: Sequence[int],
    char_byte_ends: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Translate token byte spans to inclusive/exclusive character spans."""
    char_starts: list[int] = []
    char_ends: list[int] = []
    token_byte_cursor = 0
    for raw_bytes in token_bytes:
        token_byte_start = token_byte_cursor
        token_byte_end = token_byte_start + len(raw_bytes)
        token_byte_cursor = token_byte_end
        start_idx = bisect_right(char_byte_ends, token_byte_start)
        end_idx = bisect_left(char_byte_starts, token_byte_end)
        end_idx = max(end_idx, start_idx)
        char_starts.append(start_idx)
        char_ends.append(end_idx)
    return char_starts, char_ends


def _build_detected_entities(
    runtime: InferenceRuntime,
    source_text: str,
    predicted_char_spans: Sequence[tuple[int, int, int]],
    span_scores: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Convert span tuples into the public entity payload."""
    detected: list[dict[str, object]] = []
    for idx, (label_idx, start, end) in enumerate(predicted_char_spans):
        if not (0 <= start < end <= len(source_text)):
            continue
        label = (
            runtime.label_info.span_class_names[label_idx]
            if 0 <= label_idx < len(runtime.label_info.span_class_names)
            else f"label_{label_idx}"
        )
        score = (
            span_scores[idx]
            if span_scores is not None and idx < len(span_scores)
            else 1.0
        )
        detected.append(
            {
                "entity": label,
                "start": int(start),
                "end": int(end),
                "score": float(score),
            }
        )
    return detected


def _score_token_spans(
    token_logprobs: torch.Tensor,
    decoded_labels: Sequence[int],
    token_spans: Sequence[tuple[int, int, int]],
) -> list[float]:
    """Return one confidence score per decoded token span."""
    span_scores: list[float] = []
    for _label_idx, start, end in token_spans:
        selected_logprobs: list[torch.Tensor] = []
        for token_idx in range(start, end):
            if not 0 <= token_idx < token_logprobs.shape[0]:
                continue
            label_idx = int(decoded_labels[token_idx])
            if not 0 <= label_idx < token_logprobs.shape[1]:
                continue
            selected_logprobs.append(token_logprobs[token_idx, label_idx])
        if not selected_logprobs:
            span_scores.append(0.0)
            continue
        mean_logprob = torch.stack(selected_logprobs).mean()
        span_scores.append(float(mean_logprob.exp().clamp(0.0, 1.0).item()))
    return span_scores


@torch.inference_mode()
def predict_text(
    runtime: InferenceRuntime,
    text: str,
    decoder: Decoder,
) -> tuple[str, list[dict[str, object]]]:
    """Predict entity spans for *text* and return a normalized entity payload."""
    token_ids = tuple(
        int(token) for token in runtime.encoding.encode(text, allowed_special="all")
    )
    if not token_ids:
        return text, []

    if runtime.n_ctx <= 0:
        raise ValueError("runtime.n_ctx must be positive")

    token_score_vectors = _collect_token_score_vectors(runtime, token_ids)

    if not token_score_vectors:
        return text, []

    stacked_scores = torch.stack(token_score_vectors, dim=0)
    decoded_labels = decoder.decode(stacked_scores)
    if len(decoded_labels) != len(token_ids):
        decoded_labels = stacked_scores.argmax(dim=1).tolist()

    predicted_labels_by_index = {
        token_idx: int(label) for token_idx, label in enumerate(decoded_labels)
    }
    predicted_token_spans = labels_to_spans(
        predicted_labels_by_index, runtime.label_info
    )
    token_bytes, decoded_text = _decode_token_bytes(runtime, token_ids)
    char_byte_starts, char_byte_ends = _compute_char_byte_offsets(decoded_text)
    char_starts, char_ends = _token_byte_spans_to_char_spans(
        token_bytes,
        char_byte_starts,
        char_byte_ends,
    )
    if char_ends and char_ends[-1] != len(decoded_text):
        raise ValueError(
            "Character length mismatch for decoded text "
            f"(tokens={char_ends[-1]}, text={len(decoded_text)})"
        )
    if decoded_text != text:
        raise ValueError(
            "Decoded token text differs from input text; cannot safely map spans "
            "back to original input"
        )
    source_text = text
    span_scores = _score_token_spans(
        stacked_scores,
        decoded_labels,
        predicted_token_spans,
    )
    predicted_char_spans = token_spans_to_char_spans(
        predicted_token_spans,
        char_starts,
        char_ends,
    )
    predicted_char_spans = trim_char_spans_whitespace(predicted_char_spans, source_text)
    detected = _build_detected_entities(
        runtime, source_text, predicted_char_spans, span_scores
    )
    return source_text, detected


class NERPipeline:
    """Token-classification pipeline loaded once and reused across files."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Initialize the cached runtime and decoder."""
        self._runtime = get_runtime(model_id=model_id)
        self._decoder = Decoder(label_info=self._runtime.label_info, model_id=model_id)

    def predict(self, text: str) -> list[EntitySpan]:
        """Return PII entity spans detected in *text*."""
        if not text.strip():
            return []
        source_text, detected = predict_text(self._runtime, text, self._decoder)
        spans = []
        for entity in detected:
            try:
                start = _require_int(
                    entity.get("start"), context="detected entity", field="start"
                )
                end = _require_int(
                    entity.get("end"), context="detected entity", field="end"
                )
            except ValueError:
                continue
            entity_type_raw = entity.get("entity")
            if not isinstance(entity_type_raw, str):
                continue
            score_raw = entity.get("score")
            score = (
                float(score_raw)
                if isinstance(score_raw, int | float)
                and not isinstance(score_raw, bool)
                else 1.0
            )
            spans.append(
                EntitySpan(
                    start=start,
                    end=end,
                    entity_type=entity_type_raw,
                    score=score,
                    word=source_text[start:end],
                )
            )
        return spans
