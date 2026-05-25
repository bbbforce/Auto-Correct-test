"""
代码骨架模板加载器 —— 根据问题类型加载对应的 DOLFINx 代码骨架。

骨架模板存储在 templates/ 目录下，按问题类型命名。
在 CodeBuilder 的 task prompt 中注入，让 LLM 在此基础上填充物理部分。
"""

import os
import logging

logger = logging.getLogger(__name__)

TEMPLATE_DIR = os.path.dirname(__file__)

# problem_type → 模板文件名
TEMPLATE_MAP: dict[str, str] = {
    "heat": "skeleton_heat.py",
    "elasticity": "skeleton_elasticity.py",
    "hyperelasticity": "skeleton_hyperelasticity.py",
    "fluid": "skeleton_fluid.py",
    "phase_field_fracture": "skeleton_phase_field.py",
}

# 默认 fallback 模板
DEFAULT_TEMPLATE = "skeleton_heat.py"


def load_skeleton(problem_type: str) -> str:
    """根据问题类型加载对应的代码骨架模板。

    Args:
        problem_type: 问题类型（heat, elasticity, hyperelasticity, fluid, phase_field_fracture）

    Returns:
        骨架模板的代码文本。未找到时返回空字符串。
    """
    key = problem_type.lower().strip()
    filename = TEMPLATE_MAP.get(key)

    if not filename:
        logger.info(f"No skeleton template for problem_type='{key}', using default")
        filename = DEFAULT_TEMPLATE

    filepath = os.path.join(TEMPLATE_DIR, filename)
    if not os.path.isfile(filepath):
        logger.warning(f"Skeleton template file not found: {filepath}")
        return ""

    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    logger.info(f"Loaded skeleton template: {filename} ({len(code)} chars)")
    return code


def format_skeleton_for_prompt(problem_type: str) -> str:
    """加载骨架并格式化为可注入 prompt 的文本块。"""
    code = load_skeleton(problem_type)
    if not code:
        return ""

    return (
        f"【代码骨架模板 — 请以此骨架为基础生成代码，修改标注了「根据实际修改」的部分】\n"
        f"```python\n{code}\n```"
    )
