import os
from autogen_ext.models.openai import OpenAIChatCompletionClient

def get_llm_client(api_key: str, model: str = "gpt-4o", base_url: str = None, temperature: float = 0.0, vision: bool = False):
    """
    统一创建 LLM 客户端，避免硬编码。
    """
    model_info = {
        'vision': vision,
        'function_calling': True,
        'json_output': True,
        'family': 'unknown'
    }
    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info,
        temperature=temperature
    )

def load_prompt(filename: str) -> str:
    """
    从 prompts 目录加载提示词模板。
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        import logging
        logging.getLogger(__name__).warning(f"Prompt file not found: {prompt_path}")
        return ""
