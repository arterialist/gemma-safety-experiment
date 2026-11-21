"""Unit tests for Gemma architecture unwrapping and model helper utilities."""

import torch
import torch.nn as nn

from gemma_safety.mech_interp import get_text_backbone, get_lm_head


class _DummyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.norm = nn.LayerNorm(8)


class _DummyMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _DummyTextModel()


class _DummyGemma3Conditional(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyMultimodalModel()
        self.lm_head = nn.Linear(8, 32, bias=False)


class _DummyPeftWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self._base = base

    def get_base_model(self):
        return self._base


def test_unwrap_peft_and_multimodal_gemma3():
    cond = _DummyGemma3Conditional()
    backbone, layers = get_text_backbone(cond)
    assert backbone is cond.model.language_model
    assert len(layers) == 2
    assert get_lm_head(cond) is cond.lm_head

    peft_wrapped = _DummyPeftWrapper(cond)
    backbone2, layers2 = get_text_backbone(peft_wrapped)
    assert backbone2 is cond.model.language_model
    assert len(layers2) == 2
    assert get_lm_head(peft_wrapped) is cond.lm_head
