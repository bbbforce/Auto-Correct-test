import os
import logging
from autogen_ext.models.openai import OpenAIChatCompletionClient
from core import PROJECT_ROOT


def get_llm_client(
    api_key: str,
    model: str = "gpt-4o",
    base_url: str = None,
    temperature: float = 0.6,
    top_p: float = None,
    max_tokens: int = None,
    reasoning_effort: str = None,
    vision: bool = False,
    function_calling: bool = True,
    json_output: bool = True,
):
    """
    统一创建 LLM 客户端。

    Args:
        api_key:            API 密钥
        model:              模型名称
        base_url:           服务商接口地址（OpenAI 官方可留空）
        temperature:        采样温度
        top_p:              核采样参数（None = 使用 API 默认值）
        max_tokens:         最大输出 token 数（None = 使用 API 默认值）
        reasoning_effort:   思考深度 'minimal'|'low'|'medium'|'high'（None = 不启用）
        vision:             是否声明视觉能力
        function_calling:   是否声明函数调用能力
        json_output:        是否声明 JSON 输出能力
    """
    model_info = {
        'vision': vision,
        'function_calling': function_calling,
        'json_output': json_output,
        'structured_output': json_output,
        'family': 'unknown'
    }

    # 动态构建可选参数，只传入非 None 值，避免向不支持的 API 发送无效字段
    kwargs = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info,
        temperature=temperature,
    )
    if top_p is not None:
        kwargs['top_p'] = top_p
    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens
    if reasoning_effort is not None:
        kwargs['reasoning_effort'] = reasoning_effort

    return OpenAIChatCompletionClient(**kwargs)


def load_prompt(filename: str) -> str:
    """
    从 prompts 目录加载提示词模板。
    """
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.getLogger(__name__).warning(f"Prompt file not found: {prompt_path}")
        return ""
