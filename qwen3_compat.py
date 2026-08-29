"""Compat patches for qwen_asr 0.0.6 on transformers 5.12.

Import this BEFORE `from qwen_asr import Qwen3ASRModel`:

  1. Qwen3ASRConfig init-order fix: transformers 5.12 base __init__ runs a
     validate_token_ids hook that calls the overridden get_text_config(),
     which touches self.thinker_config before the subclass sets it. We
     pre-seed the attributes via object.__setattr__ so the hook finds them.
"""

import sys

sys.path.insert(0, "/data/其他模型/ASR模型/Qwen3-ASR")

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

if "default" not in ROPE_INIT_FUNCTIONS:
    def _default_rope(config, device=None):
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default_rope

from qwen_asr.core.transformers_backend import configuration_qwen3_asr as _qcfg

_thinker = _qcfg.Qwen3ASRThinkerConfig
for _attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
    if not hasattr(_thinker, _attr):
        setattr(_thinker, _attr, None)

from qwen_asr.core.transformers_backend import modeling_qwen3_asr as _qm

if not hasattr(_qm.Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):
    _qm.Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = staticmethod(_default_rope)


def repair_positional_embedding(tower):
    """from_pretrained on transformers 5.12 corrupts the sinusoids buffer
    (re-init pass writes garbage); recompute the closed-form values."""
    import numpy as np

    pe = tower.positional_embedding.positional_embedding
    length, channels = pe.shape
    log_inc = np.log(10000.0) / (channels // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(channels // 2).float())
    scaled = torch.arange(length)[:, None] * inv[None, :]
    pe.copy_(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1))


def valid_frames(feature_length):
    """ONNX-safe mel->output length (matches modeling_ctc.qwen3_output_lengths;
    avoids negative floor-division which ONNX Div truncates toward zero)."""
    full = feature_length // 100
    leave = feature_length % 100
    tail = torch.where(leave == 0, torch.zeros_like(leave), (leave - 1) // 8 + 1)
    return full * 13 + tail

_orig_init = _qcfg.Qwen3ASRConfig.__init__


def _patched_init(self, thinker_config=None, support_languages=None, **kwargs):
    object.__setattr__(self, "thinker_config", _qcfg.Qwen3ASRThinkerConfig(**(thinker_config or {})))
    object.__setattr__(self, "support_languages", support_languages)
    _orig_init(self, thinker_config=thinker_config, support_languages=support_languages, **kwargs)


if getattr(_qcfg.Qwen3ASRConfig.__init__, "__name__", "") != "_patched_init":
    _qcfg.Qwen3ASRConfig.__init__ = _patched_init
