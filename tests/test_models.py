"""tests/test_models.py — Pydantic 数据模型的单元测试。"""
import pytest
import sys
import os

# 把项目根目录加入 sys.path，使 import 正常工作
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import DiagnosisResult, EvaluationResult, ExecutionResult, ParsedSimulation


# ─── DiagnosisResult ───

class TestDiagnosisResult:
    def test_defaults(self):
        r = DiagnosisResult()
        assert r.fix_type == "code"
        assert r.confidence == 0.5
        assert r.after_code == ""

    def test_full_construction(self):
        r = DiagnosisResult(
            fix_type="parsing",
            hint="Missing import",
            before_code="import os",
            after_code="import os\nimport sys",
            confidence=0.9,
        )
        assert r.fix_type == "parsing"
        assert r.confidence == 0.9

    def test_invalid_fix_type_rejected(self):
        with pytest.raises(Exception):
            DiagnosisResult(fix_type="unknown")

    def test_confidence_clamped(self):
        with pytest.raises(Exception):
            DiagnosisResult(confidence=1.5)


# ─── EvaluationResult ───

class TestEvaluationResult:
    def test_defaults(self):
        r = EvaluationResult()
        assert r.is_correct is False
        assert r.confidence == 0.0
        assert r.feedback == ""

    def test_correct_result(self):
        r = EvaluationResult(is_correct=True, confidence=0.95, feedback="All good")
        assert r.is_correct is True

    def test_no_after_code_field(self):
        """EvaluationResult 不应有 after_code 字段（责任链清晰化）。"""
        r = EvaluationResult(is_correct=False, feedback="bad")
        assert not hasattr(r, "after_code")


# ─── ExecutionResult ───

class TestExecutionResult:
    def test_defaults(self):
        r = ExecutionResult()
        assert r.status == "error"
        assert r.files == []

    def test_timeout_status(self):
        r = ExecutionResult(status="timeout", output="timed out")
        assert r.status == "timeout"

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            ExecutionResult(status="crashed")

    def test_success_with_files(self):
        r = ExecutionResult(status="success", output="OK", files=["result/a.png"])
        assert len(r.files) == 1


# ─── ParsedSimulation ───

class TestParsedSimulation:
    def test_defaults(self):
        r = ParsedSimulation()
        assert r.parsed == {}
        assert r.full_text == ""

    def test_with_data(self):
        r = ParsedSimulation(
            parsed={"problem_type": "heat", "dimension": 2},
            full_text="Simulate heat...",
        )
        assert r.parsed["problem_type"] == "heat"


# ─── JSON 序列化往返测试 ───

class TestSerialization:
    def test_diagnosis_roundtrip(self):
        original = DiagnosisResult(fix_type="code", hint="fix", confidence=0.8)
        json_str = original.model_dump_json()
        restored = DiagnosisResult.model_validate_json(json_str)
        assert original == restored

    def test_evaluation_roundtrip(self):
        original = EvaluationResult(is_correct=True, confidence=0.9, feedback="ok")
        json_str = original.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)
        assert original == restored
