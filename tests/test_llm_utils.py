"""tests/test_llm_utils.py — LLM JSON 解析工具的单元测试。"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.llm_utils import extract_json_from_llm_response, parse_llm_response
from core.models import DiagnosisResult, EvaluationResult


# ─── extract_json_from_llm_response ───

class TestExtractJson:
    def test_clean_json(self):
        text = '{"key": "value", "num": 42}'
        result = extract_json_from_llm_response(text)
        assert result == {"key": "value", "num": 42}

    def test_markdown_wrapped_json(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_llm_response(text)
        assert result == {"key": "value"}

    def test_markdown_python_wrapped(self):
        """应该也能处理 ```python 包裹（虽然不常见）。"""
        text = 'Here is the result:\n```json\n{"fix_type": "code", "hint": "test"}\n```\nDone.'
        result = extract_json_from_llm_response(text)
        assert result["fix_type"] == "code"

    def test_json_with_surrounding_text(self):
        text = 'Here is my analysis:\n{"is_correct": true, "confidence": 0.95, "feedback": "Good"}\nEnd of analysis.'
        result = extract_json_from_llm_response(text)
        assert result["is_correct"] is True

    def test_truncated_json_repair(self):
        """截断 JSON 应通过补全大括号修复。"""
        text = '{"key": "value", "nested": {"inner": "data"}'
        result = extract_json_from_llm_response(text)
        assert result["key"] == "value"
        assert result["nested"]["inner"] == "data"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="空内容"):
            extract_json_from_llm_response("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="空内容"):
            extract_json_from_llm_response("   \n  ")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="无法从 LLM"):
            extract_json_from_llm_response("This is just plain text with no JSON at all.")

    def test_multiple_json_blocks_takes_last(self):
        """多个 JSON 块时取最后一个（通常是最终输出）。"""
        text = 'Ignore this {"inner": 1}. The real one: {"outer": 2}'
        result = extract_json_from_llm_response(text)
        assert result == {"outer": 2}

    def test_json_with_newlines(self):
        text = '{\n  "fix_type": "code",\n  "hint": "Fixed indentation",\n  "after_code": "print(1)",\n  "confidence": 0.8\n}'
        result = extract_json_from_llm_response(text)
        assert result["confidence"] == 0.8


# ─── parse_llm_response ───

class TestParseLlmResponse:
    def test_valid_diagnosis(self):
        text = '{"fix_type": "code", "hint": "Fixed import", "after_code": "import os", "confidence": 0.9}'
        result = parse_llm_response(text, DiagnosisResult)
        assert isinstance(result, DiagnosisResult)
        assert result.confidence == 0.9

    def test_valid_evaluation(self):
        text = '```json\n{"is_correct": true, "confidence": 0.95, "feedback": "Looks correct"}\n```'
        result = parse_llm_response(text, EvaluationResult)
        assert isinstance(result, EvaluationResult)
        assert result.is_correct is True

    def test_missing_required_field_uses_defaults(self):
        """DiagnosisResult 所有字段都有默认值，所以空 JSON 也应成功。"""
        text = '{}'
        result = parse_llm_response(text, DiagnosisResult)
        assert result.fix_type == "code"

    def test_extra_fields_ignored(self):
        text = '{"is_correct": true, "confidence": 0.9, "feedback": "ok", "extra": "ignored"}'
        result = parse_llm_response(text, EvaluationResult)
        assert result.is_correct is True

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_llm_response("Not valid JSON at all", DiagnosisResult)

    def test_incompatible_value_raises(self):
        """fix_type 必须是 'parsing' 或 'code'。"""
        text = '{"fix_type": "magic"}'
        with pytest.raises(ValueError):
            parse_llm_response(text, DiagnosisResult)
