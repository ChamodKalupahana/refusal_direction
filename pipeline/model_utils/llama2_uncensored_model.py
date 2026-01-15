
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.llama2_model import orthogonalize_llama2_weights, act_add_llama2_weights

# Template for Llama 2 Uncensored
LLAMA2_UNCENSORED_CHAT_TEMPLATE = "### Human: {instruction}\n### Assistant:"

def format_instruction_llama2_uncensored(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    # System prompts not typically used in this simple format
    formatted_instruction = LLAMA2_UNCENSORED_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_llama2_uncensored(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_llama2_uncensored(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama2_uncensored(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

class Llama2UncensoredModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):

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
            use_fast=False
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_llama2_uncensored, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # The end of instruction token logic needs to be careful with the new template.
        # Template: "### Human: {instruction}\n### Assistant:"
        # We want the tokens corresponding to "\n### Assistant:"
        return self.tokenizer.encode(LLAMA2_UNCENSORED_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        # "I" token is 306 in Llama 2 tokenizer
        return [306] 

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_llama2_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_llama2_weights, direction=direction, coeff=coeff, layer=layer)
