import json
import os
import datetime
from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client, load_prompt
from utils import setup_logger
from llm_utils import extract_json_from_llm_response


class ParsingAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('parsing_agent', 'parsing_agent.log', log_dir=log_dir)
        self.log_dir = log_dir
        self.model_client = get_llm_client(api_key, model, base_url, temperature=0.0)
        system_message = load_prompt("parsing.txt")
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
