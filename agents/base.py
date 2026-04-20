"""
所有 LLM Agent 的公共基类 —— 消除各 Agent 中重复的初始化模板代码。

子类只需声明 agent_name、prompt_file 等类属性，
然后实现自身的业务方法（如 clarify、parse、build_code 等）。
"""

from autogen_agentchat.agents import AssistantAgent
from core.config import get_llm_client, load_prompt
from core.utils import setup_logger


class BaseAgent:
    """LLM Agent 基类。

    类属性（子类必须设置）：
        agent_name:   Agent 标识名（用于日志和 autogen 注册）
        prompt_file:  对应的 prompt 模板文件名（位于 prompts/ 目录下）

    可选类属性：
        enable_vision: 是否需要视觉能力（默认 False）
        temperature:   LLM 采样温度（默认 0.0）
    """

    agent_name: str = ""
    prompt_file: str = ""
    enable_vision: bool = False
    temperature: float = 0.0

    def __init__(self, api_key: str, model: str = "gpt-4o",
                 base_url: str = None, log_dir: str = None):
        self.log_dir = log_dir
        self.logger = setup_logger(
            self.agent_name, f'{self.agent_name}.log', log_dir=log_dir
        )
        self.model_client = get_llm_client(
            api_key, model, base_url,
            temperature=self.temperature,
            vision=self.enable_vision,
        )

    def _get_agent(self) -> AssistantAgent:
        """创建一个全新的 AssistantAgent 实例（无历史对话状态）。"""
        system_message = load_prompt(self.prompt_file)
        return AssistantAgent(
            name=self.agent_name,
            model_client=self.model_client,
            system_message=system_message,
            model_client_stream=True,
        )
