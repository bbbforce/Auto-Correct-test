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

    可选类属性（生成参数 —— 子类可覆盖以定制行为）：
        temperature:        LLM 采样温度（默认 0.6）
        top_p:              核采样参数（None = 使用 API 默认值）
        max_tokens:         最大输出 token 数（None = 使用 API 默认值）
        reasoning_effort:   思考深度 'minimal'|'low'|'medium'|'high'（None = 不启用）

    可选类属性（模型能力声明）：
        enable_vision:      是否声明视觉能力（默认 False；False 时发送图片会抛异常）
        function_calling:   是否声明函数调用能力（默认 True）
        json_output:        是否声明 JSON 输出能力（默认 True）
    """

    agent_name: str = ""
    prompt_file: str = ""

    # ── 生成参数 ──
    temperature: float = 0.6
    top_p: float = None
    max_tokens: int = None
    reasoning_effort: str = None

    # ── 模型能力声明 ──
    enable_vision: bool = False
    function_calling: bool = True
    json_output: bool = True

    def __init__(self, api_key: str, model: str = "gpt-4o",
                 base_url: str = None, log_dir: str = None):
        self.log_dir = log_dir
        self.logger = setup_logger(
            self.agent_name, f'{self.agent_name}.log', log_dir=log_dir
        )
        self.model_client = get_llm_client(
            api_key, model, base_url,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            vision=self.enable_vision,
            function_calling=self.function_calling,
            json_output=self.json_output,
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
