from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from utils import setup_logger


class MechanicalInsightAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None,
                 report_filename: str = "simulation_report.txt", log_dir: str = None):
        self.logger = setup_logger('mechanical_insight_agent', 'mechanical_insight_agent.log', log_dir=log_dir)
        self.model_client = OpenAIChatCompletionClient(
            model=model, 
            api_key=api_key, 
            base_url=base_url,
            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},
            temperature=0
        )
        self.report_filename = report_filename
        system_message = """
你是一位机械仿真专家。分析以下 FEniCS 代码，并生成一份结构化的、对学生友好的报告，解释其科学意图和力学意义。

📌 你的解释应包括：
1. **仿真目标**：代码解决了什么物理问题？建模了哪些方程和条件？
2. **物理概念**：描述所使用的力学或物理原理（如应力、扩散、弹性、Navier-Stokes）。
3. **偏微分方程解释**：解释控制方程、其各项及各自的物理作用。
4. **代码分析**：逐行或分块解释代码实现的内容（如网格、边界条件、求解器）。
5. **关键因素**：突出敏感部分，如网格分辨率、时间步长、边界设置及其对精度的影响。
6. **数值与性能考虑**：讨论数值稳定性和优化策略（如后向 Euler、Newton 求解器参数）。
7. **结论**：总结该仿真提供的物理洞察，以及如何扩展或验证。
8. **建议**：提出改进或测试变体的建议（如材料属性变化、不同的荷载条件）。
"""
        self.agent = AssistantAgent(
            name="mechanical_insight_agent",
            model_client=self.model_client,
            system_message=system_message
        )

    async def generate_report(self, simulation_code: str, language: str = "English") -> str:
        self.logger.info("Generating report from the simulation code...")
        try:
            prompt = f"🌐 Respond in the specified language: {language}. Keep the tone accessible and instructional, suitable for advanced undergraduate or graduate students.\n\n[Simulation Code]\n{simulation_code}"
            
            result = await self.agent.run(task=prompt)
            report = result.messages[-1].content.strip()
            
            self.save_report_to_file(report)
            return report
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return "Error: Failed to generate mechanical insight report."

    def save_report_to_file(self, report: str):
        try:
            with open(self.report_filename, "w", encoding="utf-8") as file:
                file.write(report)
            self.logger.info(f"Report saved to {self.report_filename}.")
        except Exception as e:
            self.logger.error(f"Failed to save report to file: {e}")
