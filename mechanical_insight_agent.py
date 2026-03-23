from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client, load_prompt
import os
from utils import setup_logger


class MechanicalInsightAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None,
                 report_filename: str = "simulation_report.txt", log_dir: str = None):
        self.logger = setup_logger('mechanical_insight_agent', 'mechanical_insight_agent.log', log_dir=log_dir)
        self.model_client = get_llm_client(api_key, model, base_url, temperature=0.0)
        self.report_filename = report_filename
        system_message = load_prompt("mechanical_insight.txt")
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
