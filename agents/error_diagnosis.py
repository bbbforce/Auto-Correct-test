"""Agent: 错误诊断 —— 分析执行错误并尝试修复代码。"""

import json
import os
import datetime
from agents.base import BaseAgent
from core.models import DiagnosisResult
from core.llm_utils import parse_llm_response, process_stream_and_filter_think
from services.code_validator import validate_code, validate_api_patterns

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
            history_section = "\n[Previous Repair History — 以下修改必须保留，在此基础上继续修复]\n"
            for i, record in enumerate(repair_history):
                history_section += (
                    f"\n--- Attempt {i+1} ---\n"
                    f"错误: {record.get('error_message', 'N/A')[:200]}\n"
                    f"修复说明: {record.get('hint', 'N/A')}\n"
                    f"置信度: {record.get('confidence', 'N/A')}\n"
                )
                if record.get('code_after'):
                    history_section += "修复后代码已作为当前代码传入（请在此基础上修改，不要回退）\n"

        prompt = f"""
🧾 Input Data:
[Error Message]
{error_message}

[Simulation Output Log]
{simulation_output}

⚠️ 关键规则：你收到的 [Original Code] 是经过前几轮修复的最新版本。
你必须在此基础上修改，严禁回退之前已修复的内容。

[Original Code]
{code}
{history_section}"""

        # ── P0: 注入 API 数据库上下文 ──
        from services.api_context import get_api_context_for_error
        try:
            api_ref = get_api_context_for_error(error_message, code)
            if api_ref:
                prompt += f"\n{api_ref}"
                self.logger.info("Injected API database context for error diagnosis")
        except Exception as e:
            self.logger.warning(f"API context injection failed (non-fatal): {e}")

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

            # P1: 综合静态检查（替代原硬编码的 _validate_known_patterns）
            if diagnosis.after_code:
                is_valid, all_warnings = validate_code(diagnosis.after_code)
                if not is_valid or all_warnings:
                    self.logger.warning(f"静态检查发现 {len(all_warnings)} 个问题，触发二次修复")
                    retry_prompt = (
                        f"你上一次修复的代码仍包含以下问题，请修正：\n"
                        + "\n".join(f"  - {w}" for w in all_warnings)
                        + f"\n\n[需要修正的代码]\n{diagnosis.after_code}"
                    )
                    try:
                        retry_content = await process_stream_and_filter_think(
                            agent.run_stream(task=retry_prompt), print_output=True
                        )
                        retry_diagnosis = parse_llm_response(retry_content, DiagnosisResult)
                        if retry_diagnosis.after_code:
                            diagnosis.after_code = retry_diagnosis.after_code
                            diagnosis.hint += f" [二次修复: {', '.join(all_warnings)}]"
                            self.logger.info("静态检查二次修复成功")
                    except Exception as retry_e:
                        self.logger.warning(f"静态检查二次修复失败 (non-fatal): {retry_e}")

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
