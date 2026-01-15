
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.yi_model import orthogonalize_yi_weights, act_add_yi_weights

# Template for Yi Uncensored (assuming standard alpaca/vicuna style often used in uncensored finetunes)
YI_UNCENSORED_CHAT_TEMPLATE = "### Human: {instruction}\n### Assistant:"

def format_instruction_yi_uncensored(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    formatted_instruction = YI_UNCENSORED_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_yi_uncensored(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_yi_uncensored(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_yi_uncensored(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

class YiUncensoredModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        # The uncensored model repo doesn't have a tokenizer, use the base one
        base_model_path = "01-ai/yi-6b-chat"
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        tokenizer.padding_side = 'left'
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_yi_uncensored, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(YI_UNCENSORED_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        # Using same refusal tokens as base Yi model for now
        return [59597] # ['I']

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_yi_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_yi_weights, direction=direction, coeff=coeff, layer=layer)
