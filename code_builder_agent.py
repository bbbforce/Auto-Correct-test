import json
from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client, load_prompt
from utils import setup_logger

class CodeBuilderAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('code_builder_agent', 'code_builder_agent.log', log_dir=log_dir)
        self.model_client = get_llm_client(api_key, model, base_url, temperature=0.0)
        system_message = load_prompt("code_builder.txt")
        self.agent = AssistantAgent(
            name="code_builder_agent",
            model_client=self.model_client,
            system_message=system_message
        )

    async def build_code(self, parsed_data: dict) -> str:
        self.logger.info(f"Building code from parsed data: {parsed_data}")
        data_str = json.dumps(parsed_data, ensure_ascii=False)
        try:
            prompt = f"Input Parameters:\n{data_str}"
            result = await self.agent.run(task=prompt)
            code = result.messages[-1].content.strip()
            
            if code.startswith("```"):
                code = code.replace("```python", "").replace("```", "").strip()
                
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            return ""
        self.logger.info(f"Generated code: {code}")
        return code
