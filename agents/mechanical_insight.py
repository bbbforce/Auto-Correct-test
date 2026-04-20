"""Agent: 力学洞察报告 —— 从仿真代码生成力学分析报告。"""

from agents.base import BaseAgent
from core.llm_utils import process_stream_and_filter_think


class MechanicalInsightAgent(BaseAgent):
    agent_name = "mechanical_insight_agent"
    prompt_file = "mechanical_insight.txt"

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None,
                 report_filename: str = "simulation_report.txt", log_dir: str = None):
        super().__init__(api_key=api_key, model=model, base_url=base_url, log_dir=log_dir)
        self.report_filename = report_filename

    async def generate_report(self, simulation_code: str, language: str = "English") -> str:
        self.logger.info("Generating report from the simulation code...")
        try:
            prompt = f"🌐 Respond in the specified language: {language}. Keep the tone accessible and instructional, suitable for advanced undergraduate or graduate students.\n\n[Simulation Code]\n{simulation_code}"
            agent = self._get_agent()
            report = await process_stream_and_filter_think(agent.run_stream(task=prompt), print_output=True)

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
