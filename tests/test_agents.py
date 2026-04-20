"""tests/test_agents.py — Agent 模块的单元测试（mock LLM 调用）。"""
import pytest
import sys
import os
import logging
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import DiagnosisResult, EvaluationResult


def _make_stub_logger():
    """创建一个静默 logger 供测试用。"""
    logger = logging.getLogger(f"test_stub_{id(object())}")
    logger.addHandler(logging.NullHandler())
    return logger


class TestErrorDiagnosisAgent:
    """测试 ErrorDiagnosisAgent 的 prompt 构造和返回值解析（mock LLM）。"""

    @pytest.mark.asyncio
    @patch("agents.error_diagnosis.save_error_log")
    async def test_successful_diagnosis(self, mock_save_log):
        from agents.error_diagnosis import ErrorDiagnosisAgent

        agent = ErrorDiagnosisAgent.__new__(ErrorDiagnosisAgent)
        agent.agent = MagicMock()
        agent.logger = _make_stub_logger()
        agent.log_dir = None

        # 模拟 LLM 返回
        mock_result = MagicMock()
        mock_result.messages = [MagicMock(content='{"fix_type": "code", "hint": "Fixed import", "after_code": "import os", "confidence": 0.85}')]
        agent.agent.run = AsyncMock(return_value=mock_result)

        result = await agent.diagnose_and_fix(
            error_message="NameError: name 'os' is not defined",
            code="print(os.getcwd())",
            simulation_output="NameError...",
        )

        assert isinstance(result, DiagnosisResult)
        assert result.fix_type == "code"
        assert result.confidence == 0.85
        assert "import os" in result.after_code

    @pytest.mark.asyncio
    @patch("agents.error_diagnosis.save_error_log")
    async def test_diagnosis_with_repair_history(self, mock_save_log):
        from agents.error_diagnosis import ErrorDiagnosisAgent

        agent = ErrorDiagnosisAgent.__new__(ErrorDiagnosisAgent)
        agent.agent = MagicMock()
        agent.logger = _make_stub_logger()
        agent.log_dir = None

        # 捕获传给 agent.run 的 prompt
        captured_prompts = []

        async def capture_run(task):
            captured_prompts.append(task)
            mock_result = MagicMock()
            mock_result.messages = [MagicMock(content='{"fix_type": "code", "hint": "different fix", "after_code": "fixed", "confidence": 0.9}')]
            return mock_result

        agent.agent.run = capture_run

        history = [
            {"hint": "Added missing import", "confidence": 0.7},
            {"hint": "Fixed boundary condition", "confidence": 0.6},
        ]

        result = await agent.diagnose_and_fix(
            error_message="ValueError",
            code="code_here",
            simulation_output="error output",
            repair_history=history,
        )

        # 确认 prompt 中包含历史记录
        assert len(captured_prompts) == 1
        assert "Previous Repair History" in captured_prompts[0]
        assert "Added missing import" in captured_prompts[0]
        assert "Fixed boundary condition" in captured_prompts[0]

    @pytest.mark.asyncio
    @patch("agents.error_diagnosis.save_error_log")
    async def test_diagnosis_fallback_on_failure(self, mock_save_log):
        from agents.error_diagnosis import ErrorDiagnosisAgent

        agent = ErrorDiagnosisAgent.__new__(ErrorDiagnosisAgent)
        agent.agent = MagicMock()
        agent.logger = _make_stub_logger()
        agent.log_dir = None
        agent.agent.run = AsyncMock(side_effect=Exception("LLM timeout"))

        result = await agent.diagnose_and_fix(
            error_message="error",
            code="original_code",
            simulation_output="output",
        )

        assert isinstance(result, DiagnosisResult)
        assert result.confidence == 0.0
        assert result.after_code == "original_code"


class TestResultEvaluationAgent:
    """测试 ResultEvaluationAgent 的返回值解析（mock LLM）。"""

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        from agents.result_evaluation import ResultEvaluationAgent

        agent = ResultEvaluationAgent.__new__(ResultEvaluationAgent)
        agent.agent = MagicMock()
        agent.logger = _make_stub_logger()

        mock_result = MagicMock()
        mock_result.messages = [MagicMock(
            content='{"is_correct": true, "confidence": 0.92, "feedback": "Temperature distribution is physically correct."}'
        )]
        agent.agent.run = AsyncMock(return_value=mock_result)

        result = await agent.evaluate_results(
            prompt="Simulate heat transfer",
            code="fenics code",
            simulation_output="MAX_TEMP: 100",
            image_paths=[],
        )

        assert isinstance(result, EvaluationResult)
        assert result.is_correct is True
        assert result.confidence == 0.92
        # 确认没有 after_code 字段
        assert not hasattr(result, "after_code")

    @pytest.mark.asyncio
    async def test_evaluation_fallback_on_failure(self):
        from agents.result_evaluation import ResultEvaluationAgent

        agent = ResultEvaluationAgent.__new__(ResultEvaluationAgent)
        agent.agent = MagicMock()
        agent.logger = _make_stub_logger()
        agent.agent.run = AsyncMock(side_effect=Exception("Vision API failed"))

        result = await agent.evaluate_results(
            prompt="test",
            code="code",
            simulation_output="output",
            image_paths=[],
        )

        assert isinstance(result, EvaluationResult)
        assert result.is_correct is False
        assert result.confidence == 0.0
        assert "error" in result.feedback.lower() or "Error" in result.feedback
