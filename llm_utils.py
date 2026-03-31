"""
LLM 输出解析工具 —— 从 LLM 响应中提取 JSON 并转换为 Pydantic 模型。
"""

from __future__ import annotations
import json
import re
import logging
import contextvars
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# 流式输出回调上下文变量（async-safe，不需要修改任何 Agent 代码）
stream_callback_var: contextvars.ContextVar = contextvars.ContextVar('stream_callback', default=None)

T = TypeVar("T", bound=BaseModel)


async def process_stream_and_filter_think(stream_gen, print_output=True) -> str:
    """
    处理 AssistantAgent 的 run_stream 生成器，动态过滤掉 <think>...</think> 标签内容，
    不在终端打印也不包含在最终返回的字符串中。
    """
    from autogen_agentchat.messages import ModelClientStreamingChunkEvent, ThoughtEvent
    import sys

    in_think = False
    buffer = ""
    final_content = []

    try:
        async for msg in stream_gen:
            if isinstance(msg, ThoughtEvent):
                continue
                
            if hasattr(msg, "content") and isinstance(msg.content, str):
                if isinstance(msg, ModelClientStreamingChunkEvent):
                    chunk = msg.content
                    buffer += chunk

                    while True:
                        if not in_think:
                            think_start = buffer.find("<think>")
                            if think_start != -1:
                                before_think = buffer[:think_start]
                                if before_think:
                                    if print_output:
                                        sys.stdout.write(before_think)
                                        sys.stdout.flush()
                                    final_content.append(before_think)
                                    _cb = stream_callback_var.get(None)
                                    if _cb:
                                        await _cb(before_think)

                                buffer = buffer[think_start + len("<think>"):]
                                in_think = True
                            else:
                                if len(buffer) < 7:
                                    break
                                
                                safe_part = buffer[:-6]
                                if safe_part:
                                    if print_output:
                                        sys.stdout.write(safe_part)
                                        sys.stdout.flush()
                                    final_content.append(safe_part)
                                    _cb = stream_callback_var.get(None)
                                    if _cb:
                                        await _cb(safe_part)
                                buffer = buffer[-6:]
                                break
                        else:
                            think_end = buffer.find("</think>")
                            if think_end != -1:
                                buffer = buffer[think_end + len("</think>"):]
                                in_think = False
                            else:
                                if len(buffer) < 8:
                                    break
                                buffer = buffer[-7:]
                                break

        if not in_think and buffer:
            if print_output:
                sys.stdout.write(buffer)
                sys.stdout.flush()
            final_content.append(buffer)
            _cb = stream_callback_var.get(None)
            if _cb:
                await _cb(buffer)

    except Exception as e:
        logger.error(f"Error while processing stream: {e}")

    return "".join(final_content).strip()


def extract_json_from_llm_response(text: str) -> dict:
    """从 LLM 响应文本中提取 JSON 对象。

    处理策略（按顺序尝试）：
    1. 直接 json.loads
    2. 去除 markdown 代码块包裹后 json.loads
    3. 用正则提取最外层 {} 中的内容
    4. 尝试修复常见截断问题（补全大括号）

    Raises:
        ValueError: 所有策略均失败时抛出
    """
    if not text or not text.strip():
        raise ValueError("LLM 返回了空内容")

    cleaned = text.strip()

    # 策略 0：响应开头被流式截断（缺少 '{'），尝试补全后解析
    if not cleaned.startswith("{") and not cleaned.startswith("[") and not cleaned.startswith("```"):
        candidate = "{" + cleaned
        # 补全缺失的闭合括号
        open_count = candidate.count("{")
        close_count = candidate.count("}")
        if open_count > close_count:
            candidate += "}" * (open_count - close_count)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 策略 1：直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 策略 2：去除 markdown 代码块
    md_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    md_match = md_pattern.search(cleaned)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 策略 3：正则提取 {} 中的内容
    brace_pattern = re.compile(r"\{.*\}", re.DOTALL)
    brace_match = brace_pattern.search(cleaned)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 策略 4：尝试修复截断（补全大括号）
            open_count = candidate.count("{")
            close_count = candidate.count("}")
            if open_count > close_count:
                candidate += "}" * (open_count - close_count)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # 策略 5：贪婪匹配失败，逐个尝试所有 {...} 子匹配（从最后一个开始，通常是最终输出）
    all_braces = re.findall(r"\{[^{}]*\}", cleaned)
    for candidate in reversed(all_braces):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"无法从 LLM 响应中提取有效 JSON。原始响应前200字符：{cleaned[:200]}"
    )


def parse_llm_response(text: str, model_class: Type[T]) -> T:
    """从 LLM 响应中提取 JSON 并转换为指定的 Pydantic 模型。

    Args:
        text: LLM 的原始输出文本
        model_class: 目标 Pydantic BaseModel 子类

    Returns:
        解析并校验后的 Pydantic 模型实例

    Raises:
        ValueError: JSON 提取或 Pydantic 校验失败
    """
    raw_dict = extract_json_from_llm_response(text)
    try:
        return model_class.model_validate(raw_dict)
    except ValidationError as e:
        raise ValueError(
            f"LLM 输出通过了 JSON 解析但未通过 {model_class.__name__} 校验：{e}"
        ) from e
