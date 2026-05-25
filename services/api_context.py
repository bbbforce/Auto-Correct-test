"""
API 上下文注入服务 —— 从 dolfinx_api_db 中检索相关 API 签名，
动态注入到 CodeBuilder 和 ErrorDiagnosis 的 prompt 中。

解决 LLM 幻觉 API 签名的核心痛点。
"""

import re
import logging
from typing import Optional

from dolfinx_api_db.loader import load_db, search

logger = logging.getLogger(__name__)

# 模块级缓存，避免每次调用都读磁盘
_db_cache: Optional[dict] = None


def _get_db() -> dict:
    global _db_cache
    if _db_cache is None:
        _db_cache = load_db()
    return _db_cache


# ═══════════════════════════════════════════════════════
# 问题类型 → 需要关注的 API 名称映射
# ═══════════════════════════════════════════════════════
PROBLEM_TYPE_APIS: dict[str, list[str]] = {
    "heat": [
        "functionspace", "dirichletbc", "LinearProblem",
        "Constant", "VTXWriter", "XDMFFile",
    ],
    "elasticity": [
        "functionspace", "dirichletbc", "LinearProblem",
        "Constant", "Expression", "VTXWriter",
    ],
    "hyperelasticity": [
        "functionspace", "dirichletbc", "NonlinearProblem",
        "NewtonSolver", "Function", "Constant",
    ],
    "fluid": [
        "functionspace", "dirichletbc", "Function",
        "Constant", "NonlinearProblem", "NewtonSolver",
    ],
    "phase_field_fracture": [
        "functionspace", "dirichletbc", "NonlinearProblem",
        "NewtonSolver", "Function", "Constant",
    ],
}

# 核心 API —— 无论什么问题类型都应包含
CORE_APIS = ["functionspace", "dirichletbc", "Function", "Constant"]


def get_api_context_for_builder(parsed_data: dict) -> str:
    """根据问题类型检索相关 API 签名，供 CodeBuilder prompt 注入。

    Args:
        parsed_data: ParsingAgent 输出的结构化参数

    Returns:
        格式化的 API 签名参考文本，无匹配返回空字符串。
    """
    problem_type = str(parsed_data.get("problem_type", "")).lower().strip()
    api_names = PROBLEM_TYPE_APIS.get(problem_type, CORE_APIS)

    # 合并核心 API
    all_names = list(dict.fromkeys(CORE_APIS + api_names))

    results = _search_apis(all_names)
    if not results:
        return ""

    text = _format_api_reference(results)
    logger.info(f"API context for builder: {len(results)} APIs for problem_type='{problem_type}'")
    return text


def get_api_context_for_error(error_message: str, code: str) -> str:
    """从错误信息和代码中提取涉及的 API 名称，检索正确签名供修复参考。

    Args:
        error_message: 执行错误信息
        code: 当前代码

    Returns:
        格式化的 API 签名参考文本。
    """
    api_names = _extract_api_names(error_message + "\n" + code)
    if not api_names:
        return ""

    results = _search_apis(api_names[:8])
    if not results:
        return ""

    text = _format_api_reference(results)
    logger.info(f"API context for error: {len(results)} APIs from {api_names[:5]}")
    return text


# ═══════════════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════════════

def _search_apis(names: list[str]) -> list[dict]:
    """根据名称列表批量检索 API，去重。"""
    db = _get_db()
    results = []
    seen = set()

    for name in names:
        hits = search(db, name)
        for hit in hits:
            qname = hit.get("qualified_name", hit.get("name", ""))
            if qname and qname not in seen:
                seen.add(qname)
                results.append(hit)
    return results


def _extract_api_names(text: str) -> list[str]:
    """从错误信息和代码中提取可能的 DOLFINx API 名称。"""
    names = set()

    # 匹配 dolfinx.xxx.yyy 形式
    for m in re.finditer(r"dolfinx\.(\w+(?:\.\w+)*)", text):
        parts = m.group(1).split(".")
        names.add(parts[-1])  # 取最后一级名称

    # 匹配已知的 API 类/函数名
    known = [
        "LinearProblem", "NonlinearProblem", "NewtonSolver",
        "XDMFFile", "VTXWriter", "DirichletBC",
        "functionspace", "dirichletbc", "assemble_scalar",
        "assemble_vector", "assemble_matrix",
        "locate_dofs_topological", "locate_dofs_geometrical",
        "Function", "FunctionSpace", "Constant", "Expression",
        "create_rectangle", "create_box", "create_unit_square",
    ]
    text_lower = text.lower()
    for api in known:
        if api.lower() in text_lower:
            names.add(api)

    return list(names)


def _format_api_reference(results: list[dict]) -> str:
    """格式化 API 检索结果为 prompt 可注入文本。"""
    lines = ["【DOLFINx API 签名参考 — 以下为当前版本的正确调用方式，请严格遵循】"]

    for r in results:
        qname = r.get("qualified_name", r.get("name", ""))
        rtype = r.get("type", "")
        desc = r.get("description", "")
        sig = r.get("signature", "")

        if rtype == "function":
            lines.append(f"  ▸ {qname}({sig})")
            if desc:
                lines.append(f"    {desc[:120]}")
        elif rtype == "class":
            lines.append(f"  ▸ [class] {qname}")
            if desc:
                lines.append(f"    {desc[:120]}")
            # 如果有构造函数签名
            constructor = None
            # search result 不含完整 class info，这里简略处理
        elif rtype == "enum":
            values = r.get("values", [])
            lines.append(f"  ▸ [enum] {qname} — values: {values}")

    return "\n".join(lines)
