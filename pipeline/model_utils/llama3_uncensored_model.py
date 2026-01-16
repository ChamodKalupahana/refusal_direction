
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.llama3_model import (
    orthogonalize_llama3_weights,
    act_add_llama3_weights,
    tokenize_instructions_llama3_chat,
    LLAMA3_CHAT_TEMPLATE,
)

# Llama 3 Lexi Uncensored uses the same chat template as Llama 3
# Reference: https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored

# Refusal tokens for Llama 3 tokenizer
LLAMA3_UNCENSORED_REFUSAL_TOKS = [40]  # ['I']


class Llama3UncensoredModel(ModelBase):
    """Model utility class for Llama 3 uncensored models (e.g., Orenguteng/Llama-3-8B-Lexi-Uncensored)."""

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_llama3_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            LLAMA3_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
        )

    def _get_refusal_toks(self):
        return LLAMA3_UNCENSORED_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_llama3_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_llama3_weights, direction=direction, coeff=coeff, layer=layer
        )
