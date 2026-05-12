"""Agent: 输入清晰化 —— 将用户的模糊输入转化为清晰的仿真规格描述。"""

import os
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image

from agents.base import BaseAgent
from core.llm_utils import process_stream_and_filter_think


class InputClarifierAgent(BaseAgent):
    agent_name = "input_clarifier"
    prompt_file = "input_clarifier.txt"

    # ── 生成参数 ──
    temperature = 0.2
    # top_p = None              # 使用 API 默认值
    # max_tokens = None         # 使用 API 默认值
    # reasoning_effort = None   # None = 不启用; 可选 'minimal'|'low'|'medium'|'high'

    # ── 模型能力声明 ──
    enable_vision = True
    # function_calling = True   # 继承基类默认
    # json_output = True        # 继承基类默认

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

            self.logger.info("Clarified result: %s", refined)
            return refined
        except Exception as e:
            self.logger.error("Clarification failed: %s", str(e))
            return raw_input  # fallback (后备方案)
