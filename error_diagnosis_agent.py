import json
import os
import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, ThoughtEvent
from config import get_llm_client, load_prompt
from utils import setup_logger
from models import DiagnosisResult
from llm_utils import parse_llm_response, process_stream_and_filter_think

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


class ErrorDiagnosisAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('error_diagnosis_agent', 'error_diagnosis_agent.log', log_dir=log_dir)
        self.log_dir = log_dir
        self.model_client = get_llm_client(api_key, model, base_url, temperature=0.0)
        
    def _get_agent(self) -> AssistantAgent:
        system_message = load_prompt("error_diagnosis.txt")
        return AssistantAgent(
            name="error_diagnosis_agent",
            model_client=self.model_client,
            system_message=system_message,
            model_client_stream=True
        )

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
            simulation_output = "[No output detected. The code may have failed to execute properly.]" # [未检测到输出。代码可能未能正确执行。]
            self.logger.warning("Simulation output is empty; inserted placeholder message.")

        # 构建历史修复记录部分
        history_section = ""
        if repair_history:
            history_section = "\n[Previous Repair History — DO NOT repeat the same fixes]\n" # [\n[先前修复历史 —— 不要重复相同的修复]\n]
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

        # 保存 prompt 到日志目录
        prompt_log_path = os.path.join(effective_log_dir, "last_prompt.txt") if effective_log_dir else "last_prompt.txt"
        with open(prompt_log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(prompt + "\n")

        try:
            print(f"\n--- ErrorDiagnosisAgent Streaming Output ---")
            agent = self._get_agent()
            content = await process_stream_and_filter_think(agent.run_stream(task=prompt), print_output=True)
            print("\n--------------------------------------------")
            

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
