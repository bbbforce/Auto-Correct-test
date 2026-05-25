"""Agent: 代码构建 —— 根据结构化参数生成 FEniCS 仿真代码。"""

import json
from agents.base import BaseAgent
from core.llm_utils import process_stream_and_filter_think


# AST 预检最大重试次数（语法错误时自动重试）
_MAX_SYNTAX_RETRIES = 1


class CodeBuilderAgent(BaseAgent):
    agent_name = "code_builder_agent"
    prompt_file = "code_builder_fenicsx.txt"  # 可选 code_builder_fenicsx.txt 用于新版本

    # ── 生成参数 ──
    # temperature = 0.6         # 继承基类默认
    # top_p = None              # 使用 API 默认值
    # max_tokens = None         # 使用 API 默认值
    # reasoning_effort = None   # None = 不启用; 可选 'minimal'|'low'|'medium'|'high'

    # ── 模型能力声明 ──
    # enable_vision = False     # 继承基类默认
    # function_calling = True   # 继承基类默认
    # json_output = True        # 继承基类默认

    async def build_code(self, parsed_data: dict, env_info: str = "") -> str:
        self.logger.info(f"Building code from parsed data: {parsed_data}")
        data_str = json.dumps(parsed_data, ensure_ascii=False)
        try:
            prompt = f"Input Parameters:\n{data_str}"

            # ── P0: 注入 API 数据库上下文 ──
            from services.api_context import get_api_context_for_builder
            try:
                api_ref = get_api_context_for_builder(parsed_data)
                if api_ref:
                    prompt += f"\n\n{api_ref}"
                    self.logger.info("Injected API database context into prompt")
            except Exception as e:
                self.logger.warning(f"API context injection failed (non-fatal): {e}")

            # ── P0: 注入代码骨架模板 ──
            from templates import format_skeleton_for_prompt
            try:
                problem_type = parsed_data.get("problem_type", "")
                skeleton = format_skeleton_for_prompt(problem_type)
                if skeleton:
                    prompt += f"\n\n{skeleton}"
                    self.logger.info(f"Injected skeleton template for '{problem_type}'")
            except Exception as e:
                self.logger.warning(f"Skeleton template injection failed (non-fatal): {e}")

            # ── P1: 注入环境版本信息 ──
            if env_info:
                prompt += f"\n\n【目标运行环境】\n{env_info}"

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

            # ── P1: AST 语法预检 ──
            code = await self._validate_and_retry(code, prompt, agent)

        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            return ""
        self.logger.info(f"Generated code: {code}")
        return code

    async def _validate_and_retry(self, code: str, original_prompt: str, agent) -> str:
        """AST 语法预检：如果有语法错误，带上错误信息重试一次。"""
        from services.code_validator import validate_code

        is_valid, warnings = validate_code(code)
        if is_valid:
            self.logger.info("Code passed AST validation ✅")
            return code

        self.logger.warning(f"Code validation found {len(warnings)} issue(s): {warnings}")

        # 只重试一次
        retry_prompt = (
            f"你上次生成的代码存在以下问题，请修正后重新输出完整代码（纯 Python，无 Markdown）：\n"
            + "\n".join(f"  - {w}" for w in warnings)
            + f"\n\n[需要修正的代码]\n{code}"
        )

        try:
            retry_code = await process_stream_and_filter_think(
                agent.run_stream(task=retry_prompt), print_output=True
            )
            if retry_code.startswith("```"):
                retry_code = retry_code.replace("```python", "").replace("```", "").strip()

            # 再次验证
            is_valid2, warnings2 = validate_code(retry_code)
            if is_valid2:
                self.logger.info("Code passed AST validation after retry ✅")
                return retry_code
            else:
                self.logger.warning(f"Retry code still has issues: {warnings2}, using retry result anyway")
                return retry_code
        except Exception as e:
            self.logger.warning(f"AST validation retry failed (non-fatal): {e}")
            return code
