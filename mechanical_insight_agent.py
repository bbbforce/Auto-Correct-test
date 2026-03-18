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
You are a mechanical simulation expert. Analyze the following FEniCS code and produce a structured, student-friendly report that explains its scientific intent and mechanical significance.

📌 Your explanation should include:
1. **Simulation Goal**: What physical problem does the code solve? Which equations and conditions are modeled?
2. **Physical Concepts**: Describe the mechanical or physical principles used (e.g., stress, diffusion, elasticity, Navier-Stokes).
3. **PDE Explanation**: Explain the governing PDE, its terms, and their physical roles.
4. **Code Analysis**: Line-by-line or block-level explanation of what the code implements (e.g., mesh, boundary conditions, solvers).
5. **Critical Factors**: Highlight sensitive parts like mesh resolution, time step, boundary settings, and their impact on accuracy.
6. **Numerical and Performance Considerations**: Discuss numerical stability and optimization strategies (e.g., backward Euler, Newton solver parameters).
7. **Conclusion**: Summarize the physical insights provided by this simulation and how it could be extended or validated.
8. **Recommendations**: Suggest improvements or test variations (e.g., material property changes, different loading conditions).
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
