"""
Demo 检索注入服务 —— 根据问题类型从本地 Demo 库中检索相关参考代码。

将 DOLFINx 官方 Demo 源码注入到 CodeBuilder 和 ErrorDiagnosis 的 Prompt 中，
为 LLM 提供正确的 API 用法参考，提升代码生成和错误修复的成功率。
"""

import os
import re
import logging

from core import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Demo 文件根目录
DEMO_DIR = os.path.join(PROJECT_ROOT, "dolfinx-python-demo", "demo")

# ═══════════════════════════════════════════════════════
# 映射表：problem_type → demo 文件列表（按优先级排序）
# ═══════════════════════════════════════════════════════
DEMO_MAP: dict[str, list[str]] = {
    # 热传导 → Poisson（标量扩散，数学形式一致）
    "heat": ["demo_poisson.py"],
    # 线弹性
    "elasticity": ["demo_elasticity.py", "demo_static-condensation.py"],
    # 流体力学
    "fluid": ["demo_navier-stokes.py", "demo_stokes.py"],
    # 超弹性 → Cahn-Hilliard（非线性求解器模式参考）
    "hyperelasticity": ["demo_cahn-hilliard.py"],
    # 相场断裂 → Cahn-Hilliard（耦合非线性系统参考）
    "phase_field_fracture": ["demo_cahn-hilliard.py"],
    # 双调和方程 → DG/Interior Penalty
    "biharmonic": ["demo_biharmonic.py"],
}

# 默认 fallback Demo
FALLBACK_DEMOS = ["demo_poisson.py"]

# 辅助 Demo
GMSH_DEMO = "demo_gmsh.py"

# 触发 gmsh Demo 追加的关键词（在 domain 描述或 notes 中匹配）
GMSH_KEYWORDS = [
    "hole", "孔", "circle", "cylinder", "sphere", "复杂", "complex",
    "boolean", "cut", "subtract", "union", "fillet", "chamfer",
    "irregular", "不规则", "custom", "自定义几何",
    "gmsh", "step", ".step", ".iges",
]


def _needs_gmsh(parsed_data: dict) -> bool:
    """判断是否需要追加 gmsh Demo 作为辅助参考。"""
    # 条件1：存在自定义几何文件
    if parsed_data.get("domain_geometry_file"):
        return True

    # 条件2：3D 问题更可能需要 gmsh
    if parsed_data.get("dimension") == 3:
        return True

    # 条件3：domain 描述中包含复杂几何关键词
    domain_desc = str(parsed_data.get("domain", "")).lower()
    notes = str(parsed_data.get("notes", "")).lower()
    combined = domain_desc + " " + notes

    return any(kw in combined for kw in GMSH_KEYWORDS)


def _strip_comments(source: str) -> str:
    """剥离 Python 源码中的纯注释行，保留代码和行内注释。

    规则：
    - 移除以 # 开头的整行注释（允许前导空格）
    - 移除 jupytext 头部元数据块（# --- 到 # ---）
    - 保留代码行中的行内注释
    - 保留空行以维持可读性（但合并连续空行）
    """
    lines = source.split("\n")
    result = []
    in_jupytext_header = False
    prev_blank = False

    for line in lines:
        stripped = line.strip()

        # 跳过 jupytext 元数据头
        if stripped == "# ---":
            in_jupytext_header = not in_jupytext_header
            continue
        if in_jupytext_header:
            continue

        # 跳过纯注释行（包括 # +, # - 等 jupytext 标记）
        if stripped.startswith("#"):
            continue

        # 合并连续空行
        if not stripped:
            if not prev_blank:
                result.append("")
                prev_blank = True
            continue

        prev_blank = False
        result.append(line)

    return "\n".join(result).strip()


def _read_demo(filename: str, max_chars: int = 3000) -> str | None:
    """读取并预处理单个 Demo 文件。

    Args:
        filename: Demo 文件名（如 demo_poisson.py）
        max_chars: 单文件最大字符数

    Returns:
        处理后的代码字符串，或 None（文件不存在时）
    """
    filepath = os.path.join(DEMO_DIR, filename)
    if not os.path.isfile(filepath):
        logger.warning(f"Demo file not found: {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    code = _strip_comments(raw)

    # 截断
    if len(code) > max_chars:
        # 在最近的换行符处截断，避免截断一行代码到一半
        cut_pos = code.rfind("\n", 0, max_chars)
        if cut_pos == -1:
            cut_pos = max_chars
        code = code[:cut_pos] + "\n# ... (truncated)"

    return code


def _format_demo_block(demos: dict[str, str]) -> str:
    """将多个 Demo 格式化为可注入 Prompt 的文本块。"""
    if not demos:
        return ""

    sections = ["【DOLFINx 官方参考代码 — 请参考以下 API 用法和代码模式】"]
    for filename, code in demos.items():
        # 从文件名提取可读标题
        title = filename.replace("demo_", "").replace(".py", "").replace("-", " ").title()
        sections.append(f"=== {title} ({filename}) ===")
        sections.append(code)
        sections.append("")  # 空行分隔

    return "\n".join(sections)


def retrieve_demos(parsed_data: dict, max_chars: int = 4000) -> str:
    """根据解析后的问题参数检索相关 Demo，返回格式化的参考代码文本。

    Args:
        parsed_data: ParsingAgent 输出的结构化参数（含 problem_type 等字段）
        max_chars: 注入 Prompt 的总字符上限

    Returns:
        格式化的 Demo 参考文本，供注入 Prompt。无匹配时返回空字符串。
    """
    problem_type = str(parsed_data.get("problem_type", "")).lower().strip()

    # 主映射查找
    demo_files = list(DEMO_MAP.get(problem_type, FALLBACK_DEMOS))

    # 辅助 Demo 追加
    if _needs_gmsh(parsed_data) and GMSH_DEMO not in demo_files:
        demo_files.append(GMSH_DEMO)

    # 计算每个 Demo 的字符预算
    per_file_budget = max_chars // len(demo_files) if demo_files else max_chars

    # 读取并组装
    demos: dict[str, str] = {}
    total_chars = 0

    for filename in demo_files:
        remaining = max_chars - total_chars
        if remaining <= 200:  # 剩余空间太小，停止
            break

        budget = min(per_file_budget, remaining)
        code = _read_demo(filename, max_chars=budget)
        if code:
            demos[filename] = code
            total_chars += len(code)

    result = _format_demo_block(demos)
    if result:
        logger.info(
            f"Retrieved {len(demos)} demos for problem_type='{problem_type}': "
            f"{list(demos.keys())} ({total_chars} chars)"
        )

    return result


def _infer_problem_type_from_context(error_message: str, code: str) -> str:
    """从错误信息和代码中推断问题类型。

    复用 ErrorMemory 中的关键词匹配思路，但简化为直接返回 problem_type。
    """
    combined = (error_message + " " + code).lower()

    type_keywords = {
        "fluid": ["navier", "stokes", "velocity", "pressure", "流体", "ns_"],
        "heat": ["heat", "thermal", "temperature", "热传导", "温度", "poisson"],
        "elasticity": ["elastic", "stress", "strain", "displacement", "弹性", "应力", "位移"],
        "hyperelasticity": ["hyperelastic", "neo-hookean", "mooney", "超弹性", "neo_hookean"],
        "phase_field_fracture": ["phase_field", "damage", "fracture", "断裂", "相场"],
    }

    for ptype, keywords in type_keywords.items():
        if any(kw in combined for kw in keywords):
            return ptype

    return ""


def retrieve_demos_for_error(
    error_message: str, code: str, max_chars: int = 4000
) -> str:
    """根据错误信息和代码上下文检索相关 Demo，用于错误修复阶段。

    与 retrieve_demos 的区别：没有 parsed_data，需要从错误信息和代码中推断问题类型。

    Args:
        error_message: 错误信息或评估反馈
        code: 当前的仿真代码
        max_chars: 注入 Prompt 的总字符上限

    Returns:
        格式化的 Demo 参考文本，供注入 Prompt。无匹配时返回空字符串。
    """
    # 推断问题类型
    problem_type = _infer_problem_type_from_context(error_message, code)

    # 构造一个伪 parsed_data 用于复用 retrieve_demos 的逻辑
    pseudo_parsed = {"problem_type": problem_type}

    # 检测代码中是否使用了 gmsh（辅助 Demo 追加）
    if "gmsh" in code.lower():
        pseudo_parsed["domain_geometry_file"] = "detected"

    return retrieve_demos(pseudo_parsed, max_chars=max_chars)
