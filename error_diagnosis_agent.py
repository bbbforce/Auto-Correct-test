import json
import os
import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from utils import setup_logger
from models import DiagnosisResult
from llm_utils import parse_llm_response

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
        self.model_client = OpenAIChatCompletionClient(
            model=model, 
            api_key=api_key, 
            base_url=base_url,
            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},
            temperature=0
        )
        system_message = """
你是一个专门诊断和解决 FEniCS 仿真 Python 代码错误的诊断代理。

你的任务是：
1. 分析仿真输出和错误日志。
2. 准确识别原始代码失败的根本原因。
3. 返回修正后的完整代码（纯 Python，无需 markdown）。

📦 输出格式（必须严格遵循此 JSON 模式）：
{
  "fix_type": "parsing" 或 "code",
  "hint": "错误说明及修复方法",
  "after_code": "修改后的完整 Python 代码",
  "confidence": 0.0 到 1.0 之间的浮点数
}
"""
        self.agent = AssistantAgent(
            name="error_diagnosis_agent",
            model_client=self.model_client,
            system_message=system_message
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

        # 保存 prompt 到日志目录
        prompt_log_path = os.path.join(effective_log_dir, "last_prompt.txt") if effective_log_dir else "last_prompt.txt"
        with open(prompt_log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(prompt + "\n")

        try:
            result = await self.agent.run(task=prompt)
            content = result.messages[-1].content.strip()

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
