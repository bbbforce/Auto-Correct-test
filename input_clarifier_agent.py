from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client, load_prompt
from utils import setup_logger


class InputClarifierAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('input_clarifier_agent', 'input_clarifier_agent.log', log_dir=log_dir)
        self.model_client = get_llm_client(api_key, model, base_url, temperature=0.2)
        system_message = load_prompt("input_clarifier.txt")
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
