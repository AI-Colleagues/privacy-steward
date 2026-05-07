"""Targeted tests for the pipeline and model helpers."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
from privacy_steward import pipeline
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import (
    AttentionBlock,
    Checkpoint,
    Decoder,
    InferenceRuntime,
    LabelInfo,
    MLPBlock,
    ModelConfig,
    NERPipeline,
    RotaryEmbedding,
    Transformer,
    build_redacted_text,
    expert_linear,
    get_runtime,
    get_viterbi_transition_biases,
    labels_to_spans,
    predict_text,
    sdpa,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
    validate_model_config_contract,
)


class FakeEncoding:
    """Tiny ASCII-only tokenizer for deterministic tests."""

    def __init__(self, transform=lambda text: text) -> None:
        self._transform = transform

    def encode(self, text: str, allowed_special: str = "all") -> list[int]:
        return list(self._transform(text).encode("utf-8"))

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id])


class FakeModel:
    """Return predictable logits for each token position."""

    def __init__(self, label_path: list[int], num_labels: int = 5) -> None:
        self.label_path = label_path
        self.num_labels = num_labels

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(token_ids.shape[0])
        logits = torch.full((seq_len, self.num_labels), -10.0)
        for idx in range(seq_len):
            label = self.label_path[idx] if idx < len(self.label_path) else 0
            logits[idx, label] = 10.0
        return logits


class FakeDecoder:
    """Return a predetermined label path regardless of logits."""

    def __init__(
        self,
        label_info: LabelInfo,
        decoded_path: list[int] | None = None,
    ) -> None:
        self.label_info = label_info
        self.decoded_path = decoded_path or []

    def decode(self, token_logprobs: torch.Tensor) -> list[int]:
        return list(self.decoded_path)


class FakeCache:
    """Wrap a tensor so we can force a device mismatch branch."""

    def __init__(self, tensor: torch.Tensor, device: torch.device) -> None:
        self.tensor = tensor
        self.device = device
        self.shape = tensor.shape

    def to(self, device: torch.device) -> torch.Tensor:
        return self.tensor.to(device)

    def __getitem__(self, item):
        return self.tensor[item]


def _valid_config() -> dict[str, object]:
    return {
        "model_type": "privacy_filter",
        "encoding": "gpt2",
        "num_hidden_layers": 1,
        "num_experts": 2,
        "experts_per_token": 1,
        "vocab_size": 16,
        "num_labels": 33,
        "hidden_size": 4,
        "intermediate_size": 2,
        "head_dim": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "sliding_window": 3,
        "bidirectional_context": True,
        "bidirectional_left_context": 1,
        "bidirectional_right_context": 1,
        "default_n_ctx": 4,
        "initial_context_length": 4,
        "rope_theta": 10000.0,
        "rope_scaling_factor": 1.0,
        "rope_ntk_alpha": 1.0,
        "rope_ntk_beta": 32.0,
        "param_dtype": "bfloat16",
    }


def _label_info() -> LabelInfo:
    return LabelInfo(
        boundary_label_lookup={"private_person": {"B": 1, "I": 2, "E": 3, "S": 4}},
        token_to_span_label={0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
        token_boundary_tags={0: None, 1: "B", 2: "I", 3: "E", 4: "S"},
        span_class_names=("O", "private_person"),
        span_label_lookup={"O": 0, "private_person": 1},
        background_token_label=0,
        background_span_label=0,
    )


def _runtime(
    *,
    label_path: list[int],
    transform=lambda text: text,
    n_ctx: int = 8,
) -> InferenceRuntime:
    return InferenceRuntime(
        model=FakeModel(label_path=label_path),
        encoding=FakeEncoding(transform=transform),
        label_info=_label_info(),
        device=torch.device("cpu"),
        n_ctx=n_ctx,
    )


def test_require_helpers_validate_inputs() -> None:
    assert pipeline._require_int(4, context="cfg", field="value") == 4
    assert (
        pipeline._require_nonempty_string(
            "text",
            context="cfg",
            field="encoding",
        )
        == "text"
    )


def test_require_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        pipeline._require_int(True, context="cfg", field="value")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        pipeline._require_nonempty_string("", context="cfg", field="encoding")


def test_close_open_span_appends_and_resets() -> None:
    spans: list[tuple[int, int, int]] = []
    next_label, next_start = pipeline._close_open_span(spans, 3, 5, 7)
    assert spans == [(3, 5, 8)]
    assert next_label is None
    assert next_start is None


def test_close_open_span_ignores_incomplete_state() -> None:
    spans: list[tuple[int, int, int]] = []
    next_label, next_start = pipeline._close_open_span(spans, None, 5, 7)
    assert spans == []
    assert next_label is None
    assert next_start is None


@pytest.mark.parametrize(
    (
        "boundary_tag",
        "current_label",
        "start_idx",
        "previous_idx",
        "expected_spans",
        "expected_state",
    ),
    [
        ("S", 9, 2, 4, [(9, 2, 5), (7, 8, 9)], (None, None)),
        ("B", 9, 2, 4, [(9, 2, 5)], (7, 8)),
        ("I", 7, 2, 4, [], (7, 2)),
        ("I", 8, 2, 4, [(8, 2, 5)], (7, 8)),
        ("E", 7, 2, 4, [(7, 2, 9)], (None, None)),
        ("E", None, None, 4, [(7, 8, 9)], (None, None)),
        (None, 9, 2, 4, [(9, 2, 5)], (None, None)),
    ],
)
def test_handle_span_boundary_cases(
    boundary_tag: str | None,
    current_label: int | None,
    start_idx: int | None,
    previous_idx: int | None,
    expected_spans: list[tuple[int, int, int]],
    expected_state: tuple[int | None, int | None],
) -> None:
    spans: list[tuple[int, int, int]] = []
    next_label, next_start = pipeline._handle_span_boundary(
        spans,
        boundary_tag=boundary_tag,
        span_label=7,
        token_idx=8,
        current_label=current_label,
        start_idx=start_idx,
        previous_idx=previous_idx,
    )
    assert spans == expected_spans
    assert (next_label, next_start) == expected_state


def test_validate_model_config_contract_accepts_valid_payload() -> None:
    validate_model_config_contract(_valid_config(), context="config.json")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda cfg: cfg.pop("encoding"), "missing required model config keys"),
        (lambda cfg: cfg.__setitem__("model_type", "wrong"), "model_type must be"),
        (
            lambda cfg: cfg.__setitem__("bidirectional_context", False),
            "bidirectional_context=true",
        ),
        (
            lambda cfg: cfg.__setitem__("bidirectional_left_context", -1),
            "must be >= 0",
        ),
        (
            lambda cfg: (
                cfg.__setitem__("bidirectional_left_context", 2),
                cfg.__setitem__("bidirectional_right_context", 1),
            ),
            "must be symmetric",
        ),
        (
            lambda cfg: cfg.__setitem__("sliding_window", 99),
            "sliding_window must equal",
        ),
        (lambda cfg: cfg.__setitem__("num_labels", 7), "num_labels=33"),
        (lambda cfg: cfg.__setitem__("default_n_ctx", 0), "must be positive"),
        (lambda cfg: cfg.__setitem__("param_dtype", "float16"), "must be bfloat16"),
    ],
)
def test_validate_model_config_contract_rejects_invalid_payload(
    mutator,
    message: str,
) -> None:
    cfg = _valid_config()
    mutator(cfg)
    with pytest.raises(ValueError, match=message):
        validate_model_config_contract(cfg, context="config.json")


def test_model_config_from_checkpoint_config_success() -> None:
    cfg = _valid_config()
    config = ModelConfig.from_checkpoint_config(cfg, context="config.json")
    assert config.bidirectional_context_size == 1
    assert config.num_hidden_layers == 1


def test_model_config_from_checkpoint_config_missing_fields() -> None:
    cfg = _valid_config()
    cfg.pop("hidden_size")
    with pytest.raises(ValueError, match="missing required model config fields"):
        ModelConfig.from_checkpoint_config(cfg, context="config.json")


def test_model_config_from_checkpoint_config_wraps_type_errors(monkeypatch) -> None:
    cfg = _valid_config()

    def boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise TypeError("boom")

    monkeypatch.setattr(ModelConfig, "__init__", boom)
    with pytest.raises(ValueError, match="Invalid model config payload"):
        ModelConfig.from_checkpoint_config(cfg, context="config.json")


def test_expert_linear_adds_bias() -> None:
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    weight = torch.ones((1, 2, 2, 1))
    bias = torch.tensor([[[0.5], [1.5]]])

    out = expert_linear(x, weight, bias)

    assert out.shape == (1, 2, 1)
    assert torch.allclose(out, torch.tensor([[[3.5], [8.5]]]))


def test_expert_linear_without_bias() -> None:
    x = torch.tensor([[[1.0, 2.0]]])
    weight = torch.ones((1, 1, 2, 1))

    out = expert_linear(x, weight, None)

    assert torch.allclose(out, torch.tensor([[[3.0]]]))


def test_rotary_embedding_concentration_and_cache_growth() -> None:
    emb = RotaryEmbedding(2, 10000, torch.float32, initial_context_length=1)
    concentration, inv_freq = emb._compute_concentration_and_inv_freq()
    assert concentration == 1.0
    assert inv_freq.shape == (1,)

    scaled = RotaryEmbedding(
        2,
        10000,
        torch.float32,
        initial_context_length=2,
        scaling_factor=2.0,
    )
    scaled_concentration, scaled_inv_freq = scaled._compute_concentration_and_inv_freq()
    assert scaled_concentration > 1.0
    assert scaled_inv_freq.shape == (1,)

    query = torch.zeros((3, 2), dtype=torch.float32)
    key = torch.zeros((3, 2), dtype=torch.float32)
    rotated_query, rotated_key = emb(query, key)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_rotary_embedding_device_mismatch_branch() -> None:
    emb = RotaryEmbedding(2, 10000, torch.float32, initial_context_length=2)
    object.__setattr__(emb, "cos_cache", FakeCache(emb.cos_cache, torch.device("meta")))
    object.__setattr__(emb, "sin_cache", FakeCache(emb.sin_cache, torch.device("meta")))
    query = torch.zeros((1, 2), dtype=torch.float32)
    key = torch.zeros((1, 2), dtype=torch.float32)

    rotated_query, rotated_key = emb(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_decoder_transition_predicates() -> None:
    assert not Decoder._is_valid_transition(
        prev_tag=None,
        prev_span=None,
        next_tag=None,
        next_span=1,
        background_token_idx=0,
        background_span_idx=0,
        next_idx=1,
    )
    assert not Decoder._is_valid_transition(
        prev_tag="B",
        prev_span=1,
        next_tag="S",
        next_span=2,
        background_token_idx=0,
        background_span_idx=0,
        next_idx=2,
    )
    assert not Decoder._is_valid_transition(
        prev_tag="X",
        prev_span=1,
        next_tag="B",
        next_span=1,
        background_token_idx=0,
        background_span_idx=0,
        next_idx=1,
    )


def test_sdpa_returns_expected_shape() -> None:
    q = torch.ones((3, 1, 2, 2), dtype=torch.float32)
    k = torch.ones((3, 1, 2), dtype=torch.float32)
    v = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
    sinks = torch.zeros((1, 2), dtype=torch.float32)

    out = sdpa(q, k, v, sinks, sm_scale=1.0, context_size=1)

    assert out.shape == (3, 4)


def test_attention_block_forward_hits_dtype_conversions() -> None:
    config = ModelConfig.from_checkpoint_config(_valid_config(), context="config.json")
    block = AttentionBlock(config, device=torch.device("cpu"))
    original_sdpa = pipeline.sdpa
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        pipeline,
        "sdpa",
        lambda q, k, v, sinks, sm_scale, context_size: original_sdpa(
            q, k, v, sinks, sm_scale, context_size
        ).to(torch.float32),
    )
    try:
        out = block(torch.randn(3, config.hidden_size, dtype=torch.float32))
    finally:
        monkeypatch.undo()
    assert out.shape == (3, config.hidden_size)


def test_mlp_block_forward_hits_chunk_and_cast_paths(monkeypatch) -> None:
    config = ModelConfig.from_checkpoint_config(_valid_config(), context="config.json")
    block = MLPBlock(config, device=torch.device("cpu"))
    original_softmax = torch.softmax

    def softmax_bfloat16(input, dim=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        return original_softmax(input, dim, *args, **kwargs).to(torch.bfloat16)

    monkeypatch.setattr(torch, "softmax", softmax_bfloat16)
    out = block(torch.randn(33, config.hidden_size, dtype=torch.float32))
    assert out.shape == (33, config.hidden_size)


def test_mlp_block_forward_without_cast_branch() -> None:
    config = ModelConfig.from_checkpoint_config(_valid_config(), context="config.json")
    block = MLPBlock(config, device=torch.device("cpu"))
    out = block(torch.randn(2, config.hidden_size, dtype=torch.float32))
    assert out.shape == (2, config.hidden_size)


def test_get_viterbi_transition_biases_default_branch(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: tmp_path
    )
    assert get_viterbi_transition_biases() == {
        key: 0.0 for key in pipeline.VITERBI_TRANSITION_BIAS_KEYS
    }


def test_get_model_dir_appends_original_suffix(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        pipeline, "snapshot_download", lambda *args, **kwargs: str(tmp_path)
    )
    model_dir = pipeline._get_model_dir()
    assert model_dir == tmp_path / "original"


def test_get_viterbi_transition_biases_reads_explicit_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    payload = {
        "operating_points": {
            "default": {
                "biases": {
                    key: float(idx)
                    for idx, key in enumerate(pipeline.VITERBI_TRANSITION_BIAS_KEYS)
                }
            }
        }
    }
    (model_dir / "viterbi_calibration.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )

    biases = get_viterbi_transition_biases()
    assert biases["transition_bias_background_stay"] == 0.0
    assert biases["transition_bias_end_to_start"] == float(
        len(pipeline.VITERBI_TRANSITION_BIAS_KEYS) - 1
    )


def test_get_viterbi_transition_biases_reads_top_level_biases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    payload = {
        key: float(idx) for idx, key in enumerate(pipeline.VITERBI_TRANSITION_BIAS_KEYS)
    }
    (model_dir / "viterbi_calibration.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )

    biases = get_viterbi_transition_biases()
    assert biases["transition_bias_background_stay"] == 0.0
    assert biases["transition_bias_end_to_start"] == float(
        len(pipeline.VITERBI_TRANSITION_BIAS_KEYS) - 1
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "Invalid Viterbi calibration payload"),
        ('{"operating_points": []}', "Invalid operating_points payload"),
        (
            '{"operating_points": {"default": []}}',
            "Missing operating_points.default",
        ),
        ('{"operating_points": {"default": {}}}', "Invalid Viterbi bias payload"),
        (
            json.dumps(
                {
                    "operating_points": {
                        "default": {
                            "biases": {
                                key: (False if i == 0 else 0.0)
                                for i, key in enumerate(
                                    pipeline.VITERBI_TRANSITION_BIAS_KEYS
                                )
                            }
                        }
                    }
                }
            ),
            "Missing or invalid",
        ),
    ],
)
def test_get_viterbi_transition_biases_rejects_invalid_payloads(
    monkeypatch,
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    (model_dir / "viterbi_calibration.json").write_text(payload)
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )

    with pytest.raises(ValueError, match=message):
        get_viterbi_transition_biases()


def test_decoder_initialization_builds_transition_scores(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "get_viterbi_transition_biases",
        lambda model_id=pipeline.DEFAULT_MODEL: {
            key: 0.0 for key in pipeline.VITERBI_TRANSITION_BIAS_KEYS
        },
    )
    decoder = Decoder(_label_info())

    assert decoder._transition_scores.shape == (5, 5)
    assert decoder._start_scores.shape == (5,)
    assert decoder._end_scores.shape == (5,)


def test_decoder_decode_handles_normal_and_fallback_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "get_viterbi_transition_biases",
        lambda model_id=pipeline.DEFAULT_MODEL: {
            key: 0.0 for key in pipeline.VITERBI_TRANSITION_BIAS_KEYS
        },
    )
    decoder = Decoder(_label_info())

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
        ]
    )
    path = decoder.decode(logits)
    assert len(path) == 3

    fallback = decoder.decode(torch.full((2, 5), float("-inf")))
    assert fallback == [0, 0]


def test_decoder_decode_rejects_bad_shapes(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "get_viterbi_transition_biases",
        lambda model_id=pipeline.DEFAULT_MODEL: {
            key: 0.0 for key in pipeline.VITERBI_TRANSITION_BIAS_KEYS
        },
    )
    decoder = Decoder(_label_info())

    with pytest.raises(ValueError, match="shape \\[seq_len, num_classes\\]"):
        decoder.decode(torch.ones(5))
    assert decoder.decode(torch.empty((0, 5))) == []


def test_labels_to_spans_handles_gaps_background_unknown_and_final_close() -> None:
    labels = {
        0: 1,
        1: 2,
        3: 0,
        4: 99,
        5: 4,
        6: 1,
        7: 2,
    }
    spans = labels_to_spans(labels, _label_info())
    assert spans == [(1, 0, 2), (1, 5, 6), (1, 6, 8)]


def test_labels_to_spans_closes_on_background() -> None:
    spans = labels_to_spans({0: 1, 1: 0}, _label_info())
    assert spans == [(1, 0, 1)]


def test_token_spans_to_char_spans_basic() -> None:
    assert token_spans_to_char_spans([(1, 0, 2)], [0, 5, 11], [5, 11, 12]) == [
        (1, 0, 11)
    ]


def test_token_spans_to_char_spans_skips_invalid() -> None:
    assert token_spans_to_char_spans([(1, 5, 10)], [0, 5], [5, 10]) == []


def test_token_spans_to_char_spans_skips_zero_width_char_spans() -> None:
    assert token_spans_to_char_spans([(1, 0, 1)], [0], [0]) == []


def test_trim_char_spans_whitespace_variants() -> None:
    text = "  Alice  "
    assert trim_char_spans_whitespace([(1, 0, len(text))], text) == [(1, 2, 7)]
    assert trim_char_spans_whitespace([(1, 0, 5)], "Alice") == [(1, 0, 5)]
    assert trim_char_spans_whitespace([(1, 0, 10)], "Alice") == []
    assert trim_char_spans_whitespace([(1, 0, 3)], "   ") == []


def test_get_runtime_success(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_valid_config()))
    (model_dir / "model.safetensors").write_text("stub")
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )

    fake_model = object()
    monkeypatch.setattr(
        pipeline.Transformer,
        "from_checkpoint",
        classmethod(lambda cls, checkpoint_dir, *, device: fake_model),
    )

    runtime = get_runtime()

    assert runtime.model is fake_model
    assert runtime.n_ctx == 4
    assert runtime.label_info.span_class_names[0] == "O"
    assert "private_person" in runtime.label_info.span_class_names


def test_get_runtime_rejects_invalid_checkpoint_layout(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_valid_config()))
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )

    with pytest.raises(FileNotFoundError, match="no \\.safetensors files"):
        get_runtime()


def test_get_runtime_rejects_missing_checkpoint_directory(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_get_model_dir",
        lambda model_id=pipeline.DEFAULT_MODEL: tmp_path / "missing",
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        get_runtime()


def test_get_runtime_rejects_invalid_config_payload(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("[]")
    (model_dir / "model.safetensors").write_text("stub")
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )

    with pytest.raises(ValueError, match="Invalid checkpoint config payload"):
        get_runtime()


def test_get_runtime_rejects_missing_background_label(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_valid_config()))
    (model_dir / "model.safetensors").write_text("stub")
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )
    monkeypatch.setattr(
        pipeline, "NER_CLASS_NAMES", ("B-private_person", "E-private_person")
    )

    with pytest.raises(ValueError, match="background label 'O'"):
        get_runtime()


def test_get_runtime_rejects_incomplete_boundary_sets(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "original"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_valid_config()))
    (model_dir / "model.safetensors").write_text("stub")
    monkeypatch.setattr(
        pipeline, "_get_model_dir", lambda model_id=pipeline.DEFAULT_MODEL: model_dir
    )
    monkeypatch.setattr(
        pipeline,
        "NER_CLASS_NAMES",
        ("O", "B-private_person", "I-private_person", "E-private_person"),
    )

    with pytest.raises(ValueError, match="Missing boundary classes"):
        get_runtime()


def test_checkpoint_build_param_name_map() -> None:
    mapping = Checkpoint.build_param_name_map(2)
    assert mapping["block.0.mlp.mlp1_weight"] == "block.0.mlp.swiglu.weight"
    assert mapping["block.1.mlp.mlp2_bias"] == "block.1.mlp.out.bias"


def test_checkpoint_init_detects_duplicates(monkeypatch, tmp_path: Path) -> None:
    shard1 = tmp_path / "a.safetensors"
    shard2 = tmp_path / "b.safetensors"
    shard1.write_text("stub")
    shard2.write_text("stub")

    class FakeHandle:
        def __init__(self, keys: list[str]) -> None:
            self._keys = keys

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return self._keys

    def fake_safe_open(path, framework: str, device: str):  # type: ignore[no-untyped-def]
        return FakeHandle(["tensor"])

    monkeypatch.setattr(pipeline, "safe_open", fake_safe_open)

    with pytest.raises(ValueError, match="Duplicate tensor name"):
        Checkpoint(str(tmp_path), torch.device("cpu"), num_hidden_layers=1)


def test_checkpoint_init_records_unique_tensors(monkeypatch, tmp_path: Path) -> None:
    shard = tmp_path / "a.safetensors"
    shard.write_text("stub")

    class FakeHandle:
        def __init__(self, keys: list[str]) -> None:
            self._keys = keys

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return self._keys

    monkeypatch.setattr(
        pipeline,
        "safe_open",
        lambda path, framework, device: FakeHandle(["tensor_a", "tensor_b"]),
    )

    checkpoint = Checkpoint(str(tmp_path), torch.device("cpu"), num_hidden_layers=1)
    assert checkpoint.tensor_name_to_file == {
        "tensor_a": str(shard),
        "tensor_b": str(shard),
    }


def test_checkpoint_get_uses_param_name_map(monkeypatch) -> None:
    class FakeHandle:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name: str) -> torch.Tensor:
            return self.tensor

    fake = Checkpoint.__new__(Checkpoint)
    fake.param_name_map = {"alias": "real"}
    fake.tensor_name_to_file = {"real": "/tmp/model.safetensors"}
    fake.device_str = "cpu"
    monkeypatch.setattr(
        pipeline,
        "safe_open",
        lambda path, framework, device: FakeHandle(torch.tensor([1.0, 2.0])),
    )

    tensor = fake.get("alias")
    assert torch.equal(tensor, torch.tensor([1.0, 2.0]))


def test_checkpoint_get_raises_for_missing_tensor() -> None:
    fake = Checkpoint.__new__(Checkpoint)
    fake.param_name_map = {}
    fake.tensor_name_to_file = {}
    fake.device_str = "cpu"

    with pytest.raises(KeyError, match="not found in checkpoint"):
        fake.get("missing")


def test_transformer_from_checkpoint_success_and_shape_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _valid_config()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(json.dumps(config))

    model = Transformer(
        ModelConfig.from_checkpoint_config(config, context="config.json"),
        device=torch.device("cpu"),
    )
    tensors = {
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }

    class FakeCheckpoint:
        def __init__(
            self, path: str, device: torch.device, num_hidden_layers: int
        ) -> None:
            self.tensors = dict(tensors)

        def get(self, name: str) -> torch.Tensor:
            return self.tensors[name]

    monkeypatch.setattr(pipeline, "Checkpoint", FakeCheckpoint)

    loaded = Transformer.from_checkpoint(
        str(checkpoint_dir), device=torch.device("cpu")
    )
    assert isinstance(loaded, Transformer)

    class BadCheckpoint(FakeCheckpoint):
        def get(self, name: str) -> torch.Tensor:
            if name == "embedding.weight":
                return torch.zeros((1, 1))
            return super().get(name)

    monkeypatch.setattr(pipeline, "Checkpoint", BadCheckpoint)
    with pytest.raises(ValueError, match="Tensor shape mismatch"):
        Transformer.from_checkpoint(str(checkpoint_dir), device=torch.device("cpu"))


def test_transformer_from_checkpoint_rejects_invalid_config(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("[]")
    (checkpoint_dir / "model.safetensors").write_text("stub")

    with pytest.raises(ValueError, match="Invalid checkpoint config payload"):
        Transformer.from_checkpoint(str(checkpoint_dir), device=torch.device("cpu"))


def test_transformer_forward_small_and_chunked(monkeypatch) -> None:
    config = ModelConfig.from_checkpoint_config(_valid_config(), context="config.json")
    original_softmax = torch.softmax

    def softmax_bfloat16(input, dim=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        return original_softmax(input, dim, *args, **kwargs).to(torch.bfloat16)

    monkeypatch.setattr(torch, "softmax", softmax_bfloat16)
    model = Transformer(config, device=torch.device("cpu"))
    logits = model(torch.tensor([1, 2, 3], dtype=torch.int64))
    assert logits.shape == (3, 33)

    long_logits = model(torch.arange(33, dtype=torch.int64) % config.vocab_size)
    assert long_logits.shape == (33, 33)


def test_predict_text_success_and_mismatch_source() -> None:
    runtime = _runtime(label_path=[1, 2, 3], transform=lambda text: text.lower())
    decoder = FakeDecoder(runtime.label_info, decoded_path=[1, 2, 3])

    source_text, detected = predict_text(runtime, "Abc", decoder)
    assert source_text == "abc"
    assert detected == [{"entity": "private_person", "start": 0, "end": 3}]


def test_predict_text_empty_and_invalid_runtime() -> None:
    empty_runtime = _runtime(label_path=[], n_ctx=8)
    decoder = FakeDecoder(empty_runtime.label_info, decoded_path=[1])
    assert predict_text(empty_runtime, "", decoder) == ("", [])

    bad_runtime = _runtime(label_path=[1], n_ctx=0)
    with pytest.raises(ValueError, match="must be positive"):
        predict_text(bad_runtime, "a", decoder)


def test_predict_text_handles_empty_score_vectors(monkeypatch) -> None:
    runtime = _runtime(label_path=[1, 2, 3], transform=lambda text: text)
    decoder = FakeDecoder(runtime.label_info, decoded_path=[1, 2, 3])
    monkeypatch.setattr(
        pipeline,
        "_collect_token_score_vectors",
        lambda runtime, token_ids: [],
    )
    assert predict_text(runtime, "Abc", decoder) == ("Abc", [])


def test_predict_text_falls_back_when_decoder_length_mismatches() -> None:
    runtime = _runtime(label_path=[1, 2, 3], transform=lambda text: text.lower())
    decoder = FakeDecoder(runtime.label_info, decoded_path=[1])
    source_text, detected = predict_text(runtime, "Abc", decoder)
    assert source_text == "abc"
    assert detected == [{"entity": "private_person", "start": 0, "end": 3}]


def test_predict_text_rejects_char_length_mismatch(monkeypatch) -> None:
    runtime = _runtime(label_path=[1], transform=lambda text: text)
    decoder = FakeDecoder(runtime.label_info, decoded_path=[1])

    monkeypatch.setattr(
        pipeline,
        "_token_byte_spans_to_char_spans",
        lambda token_bytes, char_starts, char_ends: ([0], [2]),
    )
    with pytest.raises(ValueError, match="Character length mismatch"):
        predict_text(runtime, "a", decoder)


def test_collect_token_score_vectors_rejects_length_mismatch(monkeypatch) -> None:
    runtime = _runtime(label_path=[1], transform=lambda text: text)

    class BadModel:
        def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros((token_ids.shape[0] + 1, 5))

    runtime = InferenceRuntime(
        model=BadModel(),
        encoding=runtime.encoding,
        label_info=runtime.label_info,
        device=runtime.device,
        n_ctx=runtime.n_ctx,
    )

    with pytest.raises(ValueError, match="Logprob output length"):
        pipeline._collect_token_score_vectors(runtime, (1, 2, 3))


def test_build_detected_entities_skips_invalid_spans() -> None:
    runtime = _runtime(label_path=[1], transform=lambda text: text)
    detected = pipeline._build_detected_entities(
        runtime,
        "abc",
        [(1, 0, 2), (1, 2, 5)],
    )
    assert detected == [{"entity": "private_person", "start": 0, "end": 2}]


def test_build_redacted_text_handles_entities_and_skips_invalid_entries() -> None:
    text = "Alice and Bob met Carol at dusk."
    entities = [
        {"start": 10, "end": 13, "entity": "private_person"},
        {"start": 0, "end": 5, "entity": "private_person"},
        {"start": 12, "end": 14, "entity": "private_person"},
        {"start": 15.0, "end": 19.0, "entity": "private_person"},
        {"start": 20, "end": 25, "entity": 123},
        {"start": 26, "end": 30, "entity": "mystery"},
        {"start": 100, "end": 110, "entity": "mystery"},
    ]

    result = build_redacted_text(text, entities)
    assert result.count("[PERSON]") == 2
    assert "[REDACTED]" in result
    assert "Alice" not in result


def test_build_redacted_text_empty_inputs_return_original() -> None:
    assert build_redacted_text("", []) == ""
    assert build_redacted_text("plain text", []) == "plain text"


def test_ner_pipeline_passes_model_id_to_runtime_and_decoder(monkeypatch) -> None:
    fake_runtime = _runtime(label_path=[1, 2, 3], transform=lambda text: text)
    seen: dict[str, str] = {}

    def fake_get_runtime(model_id: str = pipeline.DEFAULT_MODEL) -> InferenceRuntime:
        seen["runtime"] = model_id
        return fake_runtime

    class FakeDecoderFactory:
        def __init__(
            self,
            label_info: LabelInfo,
            model_id: str = pipeline.DEFAULT_MODEL,
        ) -> None:
            seen["decoder"] = model_id
            self.label_info = label_info

        def decode(self, token_logprobs: torch.Tensor) -> list[int]:
            return [1, 2, 3]

    monkeypatch.setattr(pipeline, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(pipeline, "Decoder", FakeDecoderFactory)

    NERPipeline(model_id="custom/model")

    assert seen == {"runtime": "custom/model", "decoder": "custom/model"}


def test_ner_pipeline_predict_uses_model_output(monkeypatch) -> None:
    fake_runtime = _runtime(label_path=[1, 2, 3], transform=lambda text: text)

    class FakeDecoderFactory:
        def __init__(
            self,
            label_info: LabelInfo,
            model_id: str = pipeline.DEFAULT_MODEL,
        ) -> None:
            self.label_info = label_info

        def decode(self, token_logprobs: torch.Tensor) -> list[int]:
            return [1, 2, 3]

    monkeypatch.setattr(
        pipeline,
        "get_runtime",
        lambda model_id=pipeline.DEFAULT_MODEL: fake_runtime,
    )
    monkeypatch.setattr(pipeline, "Decoder", FakeDecoderFactory)
    pipe = NERPipeline()
    spans = pipe.predict("abc")
    assert spans == [
        EntitySpan(start=0, end=3, entity_type="private_person", score=1.0, word="abc")
    ]


def test_ner_pipeline_predict_skips_empty_text(monkeypatch) -> None:
    fake_runtime = _runtime(label_path=[], transform=lambda text: text)

    class FakeDecoderFactory:
        def __init__(
            self,
            label_info: LabelInfo,
            model_id: str = pipeline.DEFAULT_MODEL,
        ) -> None:
            self.label_info = label_info

        def decode(self, token_logprobs: torch.Tensor) -> list[int]:
            return []

    monkeypatch.setattr(
        pipeline,
        "get_runtime",
        lambda model_id=pipeline.DEFAULT_MODEL: fake_runtime,
    )
    monkeypatch.setattr(pipeline, "Decoder", FakeDecoderFactory)
    pipe = NERPipeline()
    assert pipe.predict("") == []
