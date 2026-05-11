"""Agent: 代码构建 —— 根据结构化参数生成 FEniCS 仿真代码。"""

import json
from agents.base import BaseAgent
from core.llm_utils import process_stream_and_filter_think


class CodeBuilderAgent(BaseAgent):
    agent_name = "code_builder_agent"
    prompt_file = "code_builder_fenicsx.txt"  # 可选 code_builder_fenicsx.txt 用于新版本

    async def build_code(self, parsed_data: dict) -> str:
        self.logger.info(f"Building code from parsed data: {parsed_data}")
        data_str = json.dumps(parsed_data, ensure_ascii=False)
        try:
            prompt = f"Input Parameters:\n{data_str}"

            # 从知识库检索相关历史教训
            from services.error_memory import ErrorMemory
            try:
                memory = ErrorMemory()
                problem_type = parsed_data.get("problem_type", "")
                tags = [problem_type] if problem_type else []
                relevant = memory.search_by_tags(tags) if tags else []
                lessons = memory.format_for_prompt(relevant)
                if lessons:
                    prompt += f"\n\n{lessons}"
                    self.logger.info(f"Injected {len(relevant)} lessons from error memory")
            except Exception as e:
                self.logger.warning(f"Error memory lookup failed (non-fatal): {e}")

            # 从本地 Demo 库检索相关参考代码
            from services.demo_retriever import retrieve_demos
            try:
                demo_ref = retrieve_demos(parsed_data)
                if demo_ref:
                    prompt += f"\n\n{demo_ref}"
                    self.logger.info("Injected demo reference into prompt")
            except Exception as e:
                self.logger.warning(f"Demo retrieval failed (non-fatal): {e}")

            agent = self._get_agent()
            code = await process_stream_and_filter_think(agent.run_stream(task=prompt), print_output=True)

            if code.startswith("```"):
                code = code.replace("```python", "").replace("```", "").strip()

        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            return ""
        self.logger.info(f"Generated code: {code}")
        return code
