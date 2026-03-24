import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, ThoughtEvent
from config import get_llm_client, load_prompt
from utils import setup_logger
from llm_utils import process_stream_and_filter_think

class CodeBuilderAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('code_builder_agent', 'code_builder_agent.log', log_dir=log_dir)
        self.model_client = get_llm_client(api_key, model, base_url, temperature=0.0)
        
    def _get_agent(self) -> AssistantAgent:
        system_message = load_prompt("code_builder.txt")  #可选code_builder.txt用于老版本fenics
        return AssistantAgent(
            name="code_builder_agent",
            model_client=self.model_client,
            system_message=system_message,
            model_client_stream=True
        )

    async def build_code(self, parsed_data: dict) -> str:
        self.logger.info(f"Building code from parsed data: {parsed_data}")
        data_str = json.dumps(parsed_data, ensure_ascii=False)
        try:
            prompt = f"Input Parameters:\n{data_str}"
            
            print(f"\n--- CodeBuilderAgent Streaming Output ---")
            
            agent = self._get_agent()
            code = await process_stream_and_filter_think(agent.run_stream(task=prompt), print_output=True)
            
            print("\n-----------------------------------------")
            
            if code.startswith("```"):
                code = code.replace("```python", "").replace("```", "").strip()
                
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            return ""
        self.logger.info(f"Generated code: {code}")
        return code
