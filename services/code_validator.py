"""
代码验证服务 —— AST 语法预检、Import 合规性验证、API 调用模式检查、规则自动修复。

在代码生成后 / 错误修复后立即运行，拦截明显错误，避免浪费执行 retry 额度。
"""

import ast
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# 禁止使用的 import 模块
# ═══════════════════════════════════════════════════════
FORBIDDEN_IMPORTS: dict[str, str] = {
    "dolfin": "严禁使用 dolfin（旧版 FEniCS），请使用 dolfinx",
    "fenics": "严禁使用 fenics（旧版 FEniCS），请使用 dolfinx",
    "mshr": "mshr 已不存在，请使用 gmsh 进行网格生成",
}

# ═══════════════════════════════════════════════════════
# 已知的废弃 / 错误 API 模式
# ═══════════════════════════════════════════════════════
DEPRECATED_PATTERNS: list[tuple[str, str]] = [
    (r"\bio\.gmshio\b", "io.gmshio 已废弃，应改为 from dolfinx.io import gmsh as gmshio"),
    (r"\.vector\(\)\s*\.\s*(?:max|min|array)\(\)", "严禁使用 .vector().max()，应使用 .x.array.max()"),
    (r"\bsolve\s*\(\s*a\s*==\s*L", "严禁使用 solve(a==L) 老版本语法"),
    (r"\bplot\s*\(\s*u\s*\)", "严禁使用 plot(u) 老版本语法，应使用 pyvista 离屏渲染"),
    (r"\bfem\.petsc\.", "不要使用 fem.petsc.X 的点号调用，应显式导入：from dolfinx.fem.petsc import X"),
]


# ═══════════════════════════════════════════════════════
# 核心验证函数
# ═══════════════════════════════════════════════════════

def validate_syntax(code: str) -> tuple[bool, str]:
    """AST 语法检查。

    Returns:
        (True, "") 或 (False, 错误描述)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def validate_imports(code: str) -> list[str]:
    """检查 import 语句的合规性，返回警告列表。"""
    warnings = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return warnings  # 语法错误由 validate_syntax 处理

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden, msg in FORBIDDEN_IMPORTS.items():
                    if alias.name == forbidden or alias.name.startswith(forbidden + "."):
                        warnings.append(msg)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for forbidden, msg in FORBIDDEN_IMPORTS.items():
                    if node.module == forbidden or node.module.startswith(forbidden + "."):
                        warnings.append(msg)
    return warnings


def validate_api_patterns(code: str) -> list[str]:
    """检查代码中的已知错误 API 调用模式。"""
    warnings = []

    # 废弃模式检查
    for pattern, msg in DEPRECATED_PATTERNS:
        if re.search(pattern, code):
            warnings.append(msg)

    # LinearProblem 参数检查
    if "LinearProblem(" in code:
        if "petsc_options_prefix" not in code:
            warnings.append("LinearProblem 缺少必需的 petsc_options_prefix 参数")
        if re.search(r'petsc_options_prefix\s*=\s*["\'][\s]*["\']', code):
            warnings.append("petsc_options_prefix 不能为空字符串")

    # XDMF + 高阶函数检查
    if "write_function" in code and re.search(r'["\']Lagrange["\'],\s*2', code):
        if "VTXWriter" not in code and "interpolate" not in code:
            warnings.append("P2 函数不能直接写 XDMF，需要先插值到 P1 或使用 VTXWriter")

    # result_dir 检查
    if "result_dir" not in code:
        warnings.append("缺少 result_dir 定义，应在脚本顶部定义 result_dir = os.environ.get('RESULT_DIR', 'result')")

    return warnings


def validate_code(code: str) -> tuple[bool, list[str]]:
    """综合验证入口。

    Returns:
        (is_valid, warnings_list)
    """
    all_warnings: list[str] = []

    # 1. 语法检查（致命）
    syntax_ok, syntax_err = validate_syntax(code)
    if not syntax_ok:
        all_warnings.append(f"[语法错误] {syntax_err}")
        return False, all_warnings

    # 2. Import 合规性
    import_warnings = validate_imports(code)
    all_warnings.extend(f"[Import] {w}" for w in import_warnings)

    # 3. API 模式检查
    pattern_warnings = validate_api_patterns(code)
    all_warnings.extend(f"[API] {w}" for w in pattern_warnings)

    is_valid = len(all_warnings) == 0
    return is_valid, all_warnings


# ═══════════════════════════════════════════════════════
# 规则自动修复
# ═══════════════════════════════════════════════════════

# 自动修复规则：(regex_pattern, replacement, description)
AUTO_FIX_RULES: list[tuple[str, str, str]] = [
    # io.gmshio → from dolfinx.io import gmsh as gmshio
    (
        r"from\s+dolfinx\s+import\s+io\b",
        "from dolfinx.io import gmsh as gmshio",
        "修复 dolfinx.io 导入为显式 gmshio 导入",
    ),
    (
        r"\bio\.gmshio\b",
        "gmshio",
        "将 io.gmshio 替换为 gmshio",
    ),
    # fem.petsc.LinearProblem → from dolfinx.fem.petsc import LinearProblem
    (
        r"\bfem\.petsc\.LinearProblem\b",
        "LinearProblem",
        "将 fem.petsc.LinearProblem 替换为直接引用（需确保已导入）",
    ),
    # .vector().max() → .x.array.max()
    (
        r"\.vector\(\)\s*\.\s*max\(\)",
        ".x.array.max()",
        "修复 .vector().max() 为 .x.array.max()",
    ),
    (
        r"\.vector\(\)\s*\.\s*min\(\)",
        ".x.array.min()",
        "修复 .vector().min() 为 .x.array.min()",
    ),
    (
        r"\.vector\(\)\s*\.\s*array\(\)",
        ".x.array",
        "修复 .vector().array() 为 .x.array",
    ),
]


def auto_fix_common_errors(code: str, error_message: str = "") -> tuple[str, list[str]]:
    """尝试自动修复已知的常见错误，不需要 LLM 调用。

    Args:
        code: 当前代码
        error_message: 执行错误信息（用于定向修复）

    Returns:
        (fixed_code, applied_fixes_descriptions)
    """
    fixed = code
    applied: list[str] = []

    # 规则修复
    for pattern, replacement, desc in AUTO_FIX_RULES:
        if re.search(pattern, fixed):
            fixed = re.sub(pattern, replacement, fixed)
            applied.append(desc)

    # 确保 LinearProblem 的 import 存在（如果代码中使用了 LinearProblem）
    if "LinearProblem(" in fixed and "import LinearProblem" not in fixed:
        if "from dolfinx.fem.petsc import" not in fixed:
            # 在所有 import 行之后插入
            fixed = _insert_import(fixed, "from dolfinx.fem.petsc import LinearProblem")
            applied.append("添加 LinearProblem 的显式 import")

    # 修复空的 petsc_options_prefix
    empty_prefix = re.search(r"petsc_options_prefix\s*=\s*['\"]['\"]", fixed)
    if empty_prefix:
        fixed = re.sub(
            r"petsc_options_prefix\s*=\s*['\"]['\"]",
            "petsc_options_prefix='solver_'",
            fixed,
        )
        applied.append("修复空的 petsc_options_prefix 为 'solver_'")

    # 如果 error_message 明确指示缺少 petsc_options_prefix
    if "petsc_options_prefix" in error_message and "petsc_options_prefix" not in fixed:
        # 在 LinearProblem( 调用中注入 prefix
        fixed = re.sub(
            r"(LinearProblem\s*\([^)]*)(bcs\s*=\s*\[[^\]]*\])",
            r"\1\2, petsc_options_prefix='solver_'",
            fixed,
        )
        if "petsc_options_prefix" in fixed:
            applied.append("在 LinearProblem 调用中注入 petsc_options_prefix")

    return fixed, applied


def _insert_import(code: str, import_line: str) -> str:
    """在代码的 import 块末尾插入一行 import。"""
    lines = code.split("\n")
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i
    lines.insert(last_import_idx + 1, import_line)
    return "\n".join(lines)
