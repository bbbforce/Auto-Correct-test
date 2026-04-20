"""
Pydantic 数据模型 —— 定义所有 Agent 间通信的结构化契约。
"""

from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Literal, Optional
from pydantic import BaseModel, Field


class DiagnosisResult(BaseModel):
    """ErrorDiagnosisAgent 的返回值契约。"""
    fix_type: Literal["parsing", "code"] = "code"
    hint: str = ""
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


class SimulationContext(BaseModel):
    """协调主循环中各轮次传递和追踪的状态。"""
    current_code: str = ""
    best_code: Optional[str] = None
    best_confidence: float = -1.0
    final_output: str = ""
    repair_history: list[dict] = Field(default_factory=list)
    simulation_success: bool = False

    def add_repair_record(self, hint: str, confidence: float,
                          error_message: str = "", code_before: str = "",
                          code_after: str = ""):
        self.repair_history.append({
            "hint": hint,
            "confidence": confidence,
            "error_message": error_message,
            "code_before": code_before,
            "code_after": code_after,
        })

    def update_best(self, code: str, confidence: float, output: str):
        self.best_code = code
        self.best_confidence = confidence
        self.final_output = output


@dataclass
class PipelineResult:
    """仿真管道的最终执行结果。"""
    success: bool = False
    run_dir: str = ""
    report: str = ""
    final_output: str = ""
    generated_code: str = ""
    result_images: list[str] = dc_field(default_factory=list)
    error_message: str = ""
