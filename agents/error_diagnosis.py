"""Agent: 错误诊断 —— 分析执行错误并尝试修复代码。"""

import json
import os
import datetime
from agents.base import BaseAgent
from core.models import DiagnosisResult
from core.llm_utils import parse_llm_response, process_stream_and_filter_think

ERROR_LOG_FILE = "error_logs.txt"


def save_error_log(error_message: str, code: str, simulation_output: str, log_dir: str = None, logger=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"Timestamp: {timestamp}\n"
        f"Error Message: {error_message}\n"
        + ("-" * 80 + "\n") * 4 +
        f"Code:\n{code}\n"
        + ("-" * 80 + "\n") * 4 +
        f"Simulation Output:\n{simulation_output}\n"
        + ("-" * 80 + "\n") * 4
    )
    log_path = os.path.join(log_dir, ERROR_LOG_FILE) if log_dir else ERROR_LOG_FILE
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
    if logger:
        logger.info("Error log saved to " + log_path)


class ErrorDiagnosisAgent(BaseAgent):
    agent_name = "error_diagnosis_agent"
    prompt_file = "error_diagnosis.txt"

    # ── 生成参数 ──
    # temperature = 0.6         # 继承基类默认
    # top_p = None              # 使用 API 默认值
    # max_tokens = None         # 使用 API 默认值
    # reasoning_effort = None   # None = 不启用; 可选 'minimal'|'low'|'medium'|'high'

    # ── 模型能力声明 ──
    # enable_vision = False     # 继承基类默认
    # function_calling = True   # 继承基类默认
    # json_output = True        # 继承基类默认

    async def diagnose_and_fix(
        self,
        error_message: str,
        code: str,
        simulation_output: str,
        iteration: int = -1,
        repair_history: list[dict] = None,
        log_dir: str = None,
    ) -> DiagnosisResult:
        """诊断错误并修复代码。

        Args:
            error_message: 错误信息
            code: 当前代码
            simulation_output: 仿真输出
            iteration: 当前重试轮次
            repair_history: 历史修复记录，避免重蹈覆辙
            log_dir: 日志输出目录（覆盖实例级 self.log_dir）

        Returns:
            DiagnosisResult: 结构化的诊断结果
        """
        effective_log_dir = log_dir or self.log_dir
        self.logger.info("Starting error diagnosis...")
        save_error_log(error_message, code, simulation_output, log_dir=effective_log_dir, logger=self.logger)

        if not simulation_output.strip():
            simulation_output = "[No output detected. The code may have failed to execute properly.]"
            self.logger.warning("Simulation output is empty; inserted placeholder message.")

        # 构建历史修复记录部分
        history_section = ""
        if repair_history:
            history_section = "\n[Previous Repair History — DO NOT repeat the same fixes]\n"
            for i, record in enumerate(repair_history):
                history_section += (
                    f"  Attempt {i+1}: {record.get('hint', 'N/A')} "
                    f"(confidence: {record.get('confidence', 'N/A')})\n"
                )

        prompt = f"""
🧾 Input Data:
[Error Message]
{error_message}

[Simulation Output Log]
{simulation_output}

[Original Code]
{code}
{history_section}"""

        # 从知识库检索已知修复方案
        known_fixes_section = ""
        from services.error_memory import ErrorMemory
        try:
            memory = ErrorMemory()
            similar = memory.search(error_message, top_k=3)
            known_fixes_section = memory.format_for_prompt(similar)
            if known_fixes_section:
                self.logger.info(f"Injected {len(similar)} known fixes from error memory")
        except Exception as e:
            self.logger.warning(f"Error memory lookup failed (non-fatal): {e}")

        if known_fixes_section:
            prompt += f"\n{known_fixes_section}"

        # 从本地 Demo 库检索相关参考代码
        from services.demo_retriever import retrieve_demos_for_error
        try:
            demo_ref = retrieve_demos_for_error(error_message, code)
            if demo_ref:
                prompt += f"\n{demo_ref}"
                self.logger.info("Injected demo reference for error diagnosis")
        except Exception as e:
            self.logger.warning(f"Demo retrieval for error failed (non-fatal): {e}")

        # 保存 prompt 到日志目录
        prompt_log_path = os.path.join(effective_log_dir, "last_prompt.txt") if effective_log_dir else "last_prompt.txt"
        with open(prompt_log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(prompt + "\n")

        try:
            agent = self._get_agent()
            content = await process_stream_and_filter_think(agent.run_stream(task=prompt), print_output=True)

            diagnosis = parse_llm_response(content, DiagnosisResult)
            diagnosis.before_code = code
            return diagnosis

        except Exception as e:
            self.logger.error(f"Error diagnosis failed: {e}")
            return DiagnosisResult(
                fix_type="code",
                hint=f"Diagnosis failed: {e}",
                before_code=code,
                after_code=code,
                confidence=0.0,
            )
