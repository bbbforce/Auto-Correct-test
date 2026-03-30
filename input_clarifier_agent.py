import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, ThoughtEvent, TextMessage, MultiModalMessage
from autogen_core import Image
from config import get_llm_client, load_prompt
from utils import setup_logger
from llm_utils import process_stream_and_filter_think


class InputClarifierAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('input_clarifier_agent', 'input_clarifier_agent.log', log_dir=log_dir)
        # 启用 vision 以支持图片输入
        self.model_client = get_llm_client(api_key, model, base_url, temperature=0.2, vision=True)
        
    def _get_agent(self) -> AssistantAgent:
        system_message = load_prompt("input_clarifier.txt")
        return AssistantAgent(
            name="input_clarifier_agent",
            model_client=self.model_client,
            system_message=system_message,
            model_client_stream=True
        )

    async def clarify(self, raw_input: str, images: list[str] = None) -> str:
        """澄清用户输入，支持附带图片的多模态输入。

        Args:
            raw_input: 用户的文本描述（可为空字符串，此时依赖图片/文件内容）
            images: 附件中提取的图片文件路径列表
        """
        try:
            prompt_text = f"[User Request]\n{raw_input}\n\n[Refined Simulation Specification]\n"
            self.logger.info("Clarifying input: %s", raw_input)
            if images:
                self.logger.info("附带 %d 张图片", len(images))
            
            print(f"\n--- InputClarifierAgent Streaming Output ---")
            
            agent = self._get_agent()

            # 构造 task：有图片时使用 MultiModalMessage，否则使用纯文本
            if images:
                ag_images = []
                for img_path in images:
                    if os.path.exists(img_path):
                        try:
                            ag_images.append(Image.from_file(img_path))
                            self.logger.info(f"加载图片: {img_path}")
                        except Exception as e:
                            self.logger.warning(f"加载图片失败 {img_path}: {e}")

                if ag_images:
                    mm_content = [prompt_text] + ag_images
                    task = MultiModalMessage(content=mm_content, source="user")
                else:
                    task = prompt_text
            else:
                task = prompt_text

            refined = await process_stream_and_filter_think(agent.run_stream(task=task), print_output=True)

            print("\n-------------------------------------------")
            
            self.logger.info("Clarified result: %s", refined)
            return refined
        except Exception as e:
            self.logger.error("Clarification failed: %s", str(e))
            return raw_input  # fallback (后备方案)
