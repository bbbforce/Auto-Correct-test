"""Agent: 结构化解析 —— 将清晰化后的自然语言转换为结构化 JSON 参数。"""

import json
import os
from agents.base import BaseAgent
from core.llm_utils import extract_json_from_llm_response, process_stream_and_filter_think


class ParsingAgent(BaseAgent):
    agent_name = "parsing_agent"
    prompt_file = "parsing.txt"

    async def parse(self, clarified_input: str) -> dict:
        self.logger.info(f"Parsing clarified input: {clarified_input}")
        response_content = None
        try:
            prompt = f"Input:\n{clarified_input}\n\nOutput:\n"

            agent = self._get_agent()
            content = await process_stream_and_filter_think(agent.run_stream(task=prompt), print_output=True)

            response_content = content
            parsed_fields = extract_json_from_llm_response(content)

            # 最终输出包括解析后的字段 + 原始文本
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
