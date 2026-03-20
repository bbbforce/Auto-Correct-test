from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from utils import setup_logger


class InputClarifierAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('input_clarifier_agent', 'input_clarifier_agent.log', log_dir=log_dir)
        self.model_client = OpenAIChatCompletionClient(
            model=model, 
            api_key=api_key, 
            base_url=base_url,
            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},
            temperature=0.2
        )
        system_message = """
你将收到用户用自然语言描述的仿真需求。根据以下指南，生成一段完整的、可以直接用于 FEniCS 的仿真描述。

📌 在澄清时请仔细考虑以下几个方面：
──────────────────────────────────────────────────────

🔷 问题类型与结构
- 确定偏微分方程类型：热传导、流体（Navier-Stokes）、弹性、断裂、反應扩散等。
- 明确问题是稳态还是瞬态。
- 判断是否为多物理场耦合（如热弹性、流固耦合、电力学）。
- 确定空间维度：2D 或 3D。
- 描述区域形状：矩形、圆形、圆柱、是否存在缺口等。

🔷 场变量与边界条件
- 定义场变量，如 u、p、T、d、k 等。
- 明确推断并描述边界条件和初始条件。
- 估算所需材料属性：E、nu、k、rho、cp、mu、Gc 等。

🔷 数值设置
- 建议适当的时间步长 dt（若是瞬态问题）。
- 推荐求解器结构（如非线性 Newton 求解器、交错格式等）。
- 说明输出格式（如是否将结果存储为 .xdmf）。

📦 请将输出格式化为一段完整的技术中文段落。
📦 不要包含章节标题或项目符号列表。
"""
        self.agent = AssistantAgent(
            name="input_clarifier_agent",
            model_client=self.model_client,
            system_message=system_message
        )

    async def clarify(self, raw_input: str) -> str:
        try:
            prompt = f"[User Request]\n{raw_input}\n\n[Refined Simulation Specification]\n"
            self.logger.info("Clarifying input: %s", raw_input)
            
            result = await self.agent.run(task=prompt)
            refined = result.messages[-1].content.strip()
            
            self.logger.info("Clarified result: %s", refined)
            return refined
        except Exception as e:
            self.logger.error("Clarification failed: %s", str(e))
            return raw_input  # fallback
