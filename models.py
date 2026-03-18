"""
Pydantic 数据模型 —— 定义所有 Agent 间通信的结构化契约。
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class DiagnosisResult(BaseModel):
    """ErrorDiagnosisAgent 的返回值契约。"""
    fix_type: Literal["parsing", "code"] = "code"
    hint: str = "Automatically diagnosed."
    before_code: str = ""
    after_code: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """ResultEvaluationAgent 的返回值契约。
    
    注意：评估 Agent 只负责诊断，不负责修复代码。
    feedback 字段应详细描述物理/逻辑错误。
    """
    is_correct: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback: str = ""


class ExecutionResult(BaseModel):
    """SimulationExecutorAgent 的返回值契约。"""
    status: Literal["success", "error", "timeout"] = "error"
    output: str = ""
    files: list[str] = Field(default_factory=list)


class ParsedSimulation(BaseModel):
    """ParsingAgent 的返回值契约。"""
    parsed: dict = Field(default_factory=dict)
    full_text: str = ""
