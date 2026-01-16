from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:

    # Check for Qwen2/Qwen2.5 models first (more specific match)
    if 'qwen2' in model_path.lower():
        from pipeline.model_utils.qwen2_model import Qwen2Model
        return Qwen2Model(model_path)
    if 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    # Check for Llama 3 uncensored models first (more specific match)
    if 'orenguteng/llama-3' in model_path.lower() and 'uncensored' in model_path.lower():
        from pipeline.model_utils.llama3_uncensored_model import Llama3UncensoredModel
        return Llama3UncensoredModel(model_path)
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    if 'georgesung/llama2_7b_chat_uncensored' in model_path:
        from pipeline.model_utils.llama2_uncensored_model import Llama2UncensoredModel
        return Llama2UncensoredModel(model_path)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)
    if 'spkgyk/Yi-6B-Chat-uncensored' in model_path:
        from pipeline.model_utils.yi_uncensored_model import YiUncensoredModel
        return YiUncensoredModel(model_path)
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
