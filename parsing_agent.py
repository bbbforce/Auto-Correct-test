import json
import os
import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from utils import setup_logger
from llm_utils import extract_json_from_llm_response


class ParsingAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('parsing_agent', 'parsing_agent.log', log_dir=log_dir)
        self.log_dir = log_dir
        self.model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},
            temperature=0
        )
        system_message = """
你将收到一个已澄清的、适用于 FEniCS 的仿真描述。
你的任务是将这段话解析为一个结构化的 JSON 对象，捕捉代码生成所需的所有关键属性。

📌 输出必须是一个包含以下字段的有效 JSON：
- problem_type（如 heat、fluid、elasticity、hyperelasticity、fracture）
- pde_description（一段简短的描述性句子）
- dimension（1、2 或 3）
- domain（形状描述）
- domain_geometry_file（可选，如不需要则使用 null）
- mesh（包含字段的对象：nx、ny[, nz]）—— 仅在 3D 时包含 nz
- variables（如 ["u"]、["u", "p"]、["u", "d"]）
- time_dependent（true/false）
- nonlinear（true/false）
- coupled（true/false）
- boundary_conditions（Dirichlet 或 Neumann 边界条件列表）
- initial_conditions（每个变量的初始值）
- source_terms（源项列表，或空数组 []）
- material_properties（物理参数字典）
- notes（可选字段，用于特殊考虑事项）

📦 只输出 JSON 对象。不要包含任何解释、markdown 或代码块。
"""
        self.agent = AssistantAgent(
            name="parsing_agent",
            model_client=self.model_client,
            system_message=system_message
        )

    async def parse(self, clarified_input: str) -> dict:
        self.logger.info(f"Parsing clarified input: {clarified_input}")
        response_content = None
        try:
            prompt = f"Input:\n{clarified_input}\n\nOutput:\n"
            
            # Send to LLM for parsing
            result = await self.agent.run(task=prompt)
            content = result.messages[-1].content.strip()
            response_content = content

            parsed_fields = extract_json_from_llm_response(content)

            # Final output includes parsed fields + original text
            parsed_data = {
                "parsed": parsed_fields,
                "full_text": clarified_input
            }

            self.save_parsed_json(parsed_data)
            self.save_parsed_text(parsed_data)

        except Exception as e:
            self.logger.error(f"Parsing failed: {e}")
            fallback = response_content if response_content else str(clarified_input)
            parsed_data = {
                "parsed": {"fallback_text": fallback},
                "full_text": clarified_input
            }

        self.logger.info(f"Parsed data: {parsed_data}")
        return parsed_data

    def _resolve_path(self, filename: str) -> str:
        """将文件名解析到 log_dir 下（如有）。"""
        if self.log_dir:
            return os.path.join(self.log_dir, filename)
        return filename

    def save_parsed_json(self, parsed_data: dict, path: str = "parsed_results.jsonl"):
        try:
            path = self._resolve_path(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(parsed_data, ensure_ascii=False) + "\n")
            self.logger.info(f"✅ JSON saved to {path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save JSON: {str(e)}")

    def save_parsed_text(self, parsed_data: dict, path: str = "parsed_results.txt"):
        try:
            path = self._resolve_path(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write("[PARSED RESULT]\n")
                for key, value in parsed_data.items():
                    f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
                f.write("\n" + "-" * 80 + "\n\n")
            self.logger.info(f"✅ Text log saved to {path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save text log: {str(e)}")
