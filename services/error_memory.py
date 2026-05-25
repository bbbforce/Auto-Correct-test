"""
错误记忆系统 —— 从历史修复中学习，避免重复犯错。

持久化存储"错误模式 → 修复方法"对，并在代码生成和错误诊断阶段
将相关历史知识注入 prompt。

CLI 用法:
    python -m services.error_memory --list              列出所有知识条目
    python -m services.error_memory --prune <id>        删除指定条目
    python -m services.error_memory --stats             按 Tag 统计错误频率
"""

import argparse
import datetime
import difflib
import hashlib
import json
import os
import re
import uuid
from collections import Counter
from typing import Optional

from core import PROJECT_ROOT

# 知识库文件路径
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")
KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "error_knowledge.json")


def extract_signature(error_message: str) -> str:
    """从错误消息中提取不变的核心签名。

    针对不同错误来源分层提取：
    1. FEniCS/DOLFIN 结构化错误 → *** Error + *** Reason
    2. Python 标准异常 → 异常类名 + 消息（去掉行号/路径）
    3. Physical/Logical Error → 提取首行关键描述
    4. Fallback → 前 200 字符归一化
    """
    # 1. DOLFIN 结构化错误
    error_match = re.search(r'\*\*\* Error:\s*(.+)', error_message)
    reason_match = re.search(r'\*\*\* Reason:\s*(.+)', error_message)
    if error_match:
        parts = ["DOLFIN", error_match.group(1).strip()]
        if reason_match:
            parts.append(reason_match.group(1).strip())
        return "|".join(parts)

    # 2. Python 标准异常（取最后一个匹配）
    exc_pattern = (
        r'^(RuntimeError|TypeError|ValueError|KeyError|AttributeError'
        r'|ImportError|ModuleNotFoundError|NameError|IndexError'
        r'|FileNotFoundError|OSError|ZeroDivisionError):\s*(.+)$'
    )
    exc_matches = re.findall(exc_pattern, error_message, re.MULTILINE)
    if exc_matches:
        cls, msg = exc_matches[-1]
        # 去掉可变部分：行号、文件路径
        msg = re.sub(r'line \d+', '', msg)
        msg = re.sub(r'"[^"]*\.py"', '""', msg)
        msg = re.sub(r"'[^']*\.py'", "''", msg)
        return f"{cls}|{msg.strip()}"

    # 3. Physical/Logical Error（来自 ResultEvaluationAgent 的反馈）
    if error_message.startswith("Physical/Logical Error:"):
        # 提取首行核心描述，去掉多余细节
        first_line = error_message.split("\n")[0]
        # 截断过长内容
        desc = first_line[:200].strip()
        return f"LOGIC|{desc}"

    # 4. Fallback
    normalized = re.sub(r'\s+', ' ', error_message[:200]).strip()
    return f"FALLBACK|{normalized}"


def signature_hash(signature: str) -> str:
    """生成签名的短哈希，用于快速去重。"""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def _auto_tags(error_message: str, code: str = "") -> list[str]:
    """从错误消息和代码中自动提取标签。"""
    tags = set()
    combined = (error_message + " " + code).lower()

    # 问题类型标签
    type_keywords = {
        "heat": ["heat", "thermal", "temperature", "热传导", "温度"],
        "elasticity": ["elastic", "stress", "strain", "displacement", "弹性", "应力", "位移"],
        "hyperelasticity": ["hyperelastic", "neo-hookean", "mooney", "超弹性"],
        "fluid": ["navier-stokes", "fluid", "velocity", "pressure", "流体"],
        "phase_field": ["phase_field", "damage", "fracture", "断裂", "相场"],
    }
    for tag, keywords in type_keywords.items():
        if any(kw in combined for kw in keywords):
            tags.add(tag)

    # 技术标签
    tech_keywords = {
        "gmsh": ["gmsh"],
        "meshio": ["meshio"],
        "boundary_condition": ["dirichletbc", "neumannbc", "boundary", "边界条件"],
        "solver": ["solve(", "linear_solver", "mumps", "newton_solver"],
        "variational_form": ["trialfunction", "testfunction", "bilinear", "weak form", "弱形式"],
        "post_processing": ["project(", "von_mises", "compute_vertex_values"],
        "visualization": ["matplotlib", "plt.", "colorbar", "savefig"],
        "xdmf": ["xdmf", "h5py"],
    }
    for tag, keywords in tech_keywords.items():
        if any(kw in combined for kw in keywords):
            tags.add(tag)

    return sorted(tags)


class ErrorMemory:
    """错误知识库管理器。"""

    def __init__(self, knowledge_file: str = None):
        self.knowledge_file = knowledge_file or KNOWLEDGE_FILE
        self._ensure_file()
        self._data = self._load()

    def _ensure_file(self):
        """确保知识库文件存在。"""
        os.makedirs(os.path.dirname(self.knowledge_file), exist_ok=True)
        if not os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "entries": []}, f)

    def _load(self) -> dict:
        with open(self.knowledge_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        with open(self.knowledge_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @property
    def entries(self) -> list[dict]:
        return self._data.get("entries", [])

    def add_entry(
        self,
        error_message: str,
        root_cause: str,
        fix_description: str,
        code_before: str = "",
        code_after: str = "",
        confidence: float = 0.5,
        tags: list[str] = None,
        success: bool = True,
    ) -> str:
        """添加一条错误记录。自动去重：若签名相同则合并。

        Args:
            success: 此修复是否最终成功。False 表示"此路不通"的记录。

        Returns:
            条目 ID（新建或已合并的）
        """
        sig = extract_signature(error_message)
        sig_hash = signature_hash(sig)

        # 提取有意义的代码差异片段（优先 diff，回退到截断）
        snippet_before, snippet_after = _make_snippets(code_before, code_after)

        # 自动提取标签，与手动标签合并
        auto = _auto_tags(error_message, code_before)
        all_tags = sorted(set((tags or []) + auto))

        # 查找是否已存在相同签名
        for entry in self.entries:
            if entry.get("signature_hash") == sig_hash:
                entry["occurrences"] = entry.get("occurrences", 1) + 1
                entry["updated_at"] = datetime.datetime.now().isoformat()
                if confidence > entry.get("confidence", 0):
                    entry["confidence"] = confidence
                    entry["root_cause"] = root_cause
                    entry["fix_description"] = fix_description
                    entry["code_snippet_before"] = snippet_before
                    entry["code_snippet_after"] = snippet_after
                # 合并标签
                existing_tags = set(entry.get("tags", []))
                entry["tags"] = sorted(existing_tags | set(all_tags))
                self._save()
                return entry["id"]

        # 新建条目
        entry_id = uuid.uuid4().hex[:8]
        new_entry = {
            "id": entry_id,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "error_pattern": sig,
            "signature_hash": sig_hash,
            "error_type": _classify_error_type(error_message),
            "root_cause": root_cause,
            "fix_description": fix_description,
            "tags": all_tags,
            "occurrences": 1,
            "confidence": confidence,
            "success": success,
            "code_snippet_before": snippet_before,
            "code_snippet_after": snippet_after,
        }
        self._data["entries"].append(new_entry)
        self._save()
        return entry_id

    def search(self, error_message: str, top_k: int = 3) -> list[dict]:
        """根据错误消息检索相关知识条目。

        优先通过签名精确匹配，其次通过关键词模糊匹配。
        """
        if not self.entries:
            return []

        # 1. 签名精确匹配
        sig = extract_signature(error_message)
        sig_hash = signature_hash(sig)
        exact = [e for e in self.entries if e.get("signature_hash") == sig_hash]
        if exact:
            return exact[:top_k]

        # 2. 关键词模糊匹配
        keywords = _extract_keywords(error_message)
        if not keywords:
            return []

        scored = []
        for entry in self.entries:
            searchable = (
                entry.get("error_pattern", "") + " " +
                entry.get("root_cause", "") + " " +
                entry.get("fix_description", "") + " " +
                " ".join(entry.get("tags", []))
            ).lower()
            score = sum(1 for kw in keywords if kw in searchable)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: (-x[0], -x[1].get("occurrences", 0)))
        return [e for _, e in scored[:top_k]]

    def search_by_tags(self, tags: list[str], top_k: int = 5) -> list[dict]:
        """根据标签检索相关知识条目。"""
        if not tags or not self.entries:
            return []

        tags_lower = {t.lower() for t in tags if t}
        scored = []
        for entry in self.entries:
            entry_tags = {t.lower() for t in entry.get("tags", [])}
            overlap = len(tags_lower & entry_tags)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: (-x[0], -x[1].get("occurrences", 0)))
        return [e for _, e in scored[:top_k]]

    def prune(self, entry_id: str) -> bool:
        """删除指定 ID 的条目。"""
        before = len(self.entries)
        self._data["entries"] = [
            e for e in self.entries if e.get("id") != entry_id
        ]
        if len(self._data["entries"]) < before:
            self._save()
            return True
        return False

    def format_for_prompt(self, entries: list[dict]) -> str:
        """将检索到的条目格式化为 prompt 可注入的文本。"""
        if not entries:
            return ""

        lines = ["【已知错误经验 — 来自历史修复记录，请务必避免重蹈覆辙】"]
        for i, e in enumerate(entries, 1):
            status = "✅ 成功修复" if e.get("success", True) else "❌ 修复失败（此路不通）"
            lines.append(f"  经验 {i} [{status}] (出现 {e.get('occurrences', 1)} 次, "
                         f"置信度 {e.get('confidence', 0):.1%}):")
            lines.append(f"    错误模式: {e.get('error_pattern', 'N/A')}")
            lines.append(f"    根因: {e.get('root_cause', 'N/A')}")
            lines.append(f"    修复方法: {e.get('fix_description', 'N/A')}")
            if e.get("code_snippet_before"):
                lines.append(f"    错误代码片段: {e['code_snippet_before'][:200]}")
            if e.get("code_snippet_after"):
                lines.append(f"    修复代码片段: {e['code_snippet_after'][:200]}")
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, int]:
        """按 Tag 统计错误频率。"""
        counter = Counter()
        for entry in self.entries:
            occ = entry.get("occurrences", 1)
            for tag in entry.get("tags", []):
                counter[tag] += occ
        return dict(counter.most_common())


def _extract_keywords(text: str) -> list[str]:
    """从文本中提取用于检索的关键词。"""
    # 去掉常见噪声
    noise = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "not", "no", "in", "on", "at", "to", "for", "of", "with",
        "error", "file", "line", "info", "warning", "traceback",
        "most", "recent", "call", "last", "module", "import",
    }
    # 提取有意义的词
    words = re.findall(r'[a-zA-Z_]\w{2,}', text.lower())
    return list(dict.fromkeys(w for w in words if w not in noise))[:20]


def _classify_error_type(error_message: str) -> str:
    """自动分类错误类型。"""
    if "Physical/Logical Error" in error_message:
        return "logic"
    if "parsing" in error_message.lower():
        return "parsing"
    return "code"


def _truncate(text: str, max_len: int = 500) -> str:
    """截断代码片段，只保留关键部分。"""
    if not text or len(text) <= max_len:
        return text
    lines = text.strip().split("\n")
    if len(lines) <= 15:
        return text[:max_len] + "\n... (truncated)"
    head = "\n".join(lines[:7])
    tail = "\n".join(lines[-7:])
    return f"{head}\n... (truncated {len(lines) - 14} lines) ...\n{tail}"


def _extract_diff_snippets(
    before: str, after: str, context_lines: int = 3
) -> tuple[str, str]:
    """从两段完整代码中提取差异区域及上下文，返回 (diff_before, diff_after)。

    只保留实际发生变更的行及其前后各 context_lines 行，
    多个不连续的变更块之间用 '...' 分隔。
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()

    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    # 收集所有变更行号（在各自序列中的位置）
    changed_before: set[int] = set()
    changed_after: set[int] = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        # replace / delete / insert
        for i in range(i1, i2):
            changed_before.add(i)
        for j in range(j1, j2):
            changed_after.add(j)

    if not changed_before and not changed_after:
        return "", ""  # 完全相同，无需记录

    def _collect(lines: list[str], changed: set[int]) -> str:
        """收集变更行 ± context_lines 的区域，合并重叠区间。"""
        if not changed:
            return ""  # 纯插入 / 纯删除，对应侧为空
        total = len(lines)
        # 计算需要保留的行号集合
        keep: set[int] = set()
        for idx in changed:
            for c in range(max(0, idx - context_lines),
                           min(total, idx + context_lines + 1)):
                keep.add(c)
        # 按行号排序输出，不连续处插入省略号
        sorted_keep = sorted(keep)
        result_lines: list[str] = []
        prev = -2  # 哨兵
        for idx in sorted_keep:
            if idx > prev + 1:
                if result_lines:  # 非开头
                    result_lines.append("...")
            result_lines.append(lines[idx])
            prev = idx
        return "\n".join(result_lines)

    return _collect(before_lines, changed_before), _collect(after_lines, changed_after)


def _make_snippets(code_before: str, code_after: str) -> tuple[str, str]:
    """生成适合存储的代码片段对。

    策略：
    1. 若两段代码都存在且都较长（>15行），用 diff 提取差异区域
    2. 否则回退到简单截断
    """
    if code_before and code_after:
        before_lines = code_before.strip().splitlines()
        after_lines = code_after.strip().splitlines()
        # 只有当两段代码都足够长时才提取 diff（短代码直接存全量）
        if len(before_lines) > 15 or len(after_lines) > 15:
            diff_b, diff_a = _extract_diff_snippets(code_before, code_after)
            if diff_b or diff_a:  # diff 有内容才使用
                return _truncate(diff_b, 800), _truncate(diff_a, 800)
    # 回退：简单截断
    return _truncate(code_before), _truncate(code_after)


# ═══════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════

def cli_list(args):
    """--list: 列出所有知识条目。"""
    memory = ErrorMemory()
    entries = memory.entries
    if not entries:
        print("知识库为空。")
        return

    print(f"共 {len(entries)} 条记录:\n")
    print(f"{'ID':<10} {'频率':>4}  {'置信度':>6}  {'标签':<30} {'错误摘要'}")
    print("-" * 100)
    for e in sorted(entries, key=lambda x: -x.get("occurrences", 1)):
        tags_str = ", ".join(e.get("tags", []))[:28]
        pattern = e.get("error_pattern", "")[:40]
        print(
            f"{e['id']:<10} {e.get('occurrences', 1):>4}  "
            f"{e.get('confidence', 0):>5.0%}  "
            f"{tags_str:<30} {pattern}"
        )


def cli_prune(args):
    """--prune <id>: 删除指定条目。"""
    memory = ErrorMemory()
    if memory.prune(args.prune):
        print(f"✅ 已删除条目: {args.prune}")
    else:
        print(f"❌ 未找到条目: {args.prune}")


def cli_stats(args):
    """--stats: 按 Tag 统计错误频率。"""
    memory = ErrorMemory()
    stats = memory.get_stats()
    if not stats:
        print("知识库为空，无统计数据。")
        return

    total = sum(stats.values())
    print(f"错误频率统计 (共 {total} 次错误):\n")
    print(f"{'标签':<25} {'次数':>6}  {'占比':>6}  {'分布'}")
    print("-" * 65)
    max_count = max(stats.values())
    for tag, count in stats.items():
        bar = "█" * int(count / max_count * 20)
        print(f"{tag:<25} {count:>6}  {count/total:>5.1%}  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="错误记忆系统 — 管理 FEniCS 仿真错误知识库"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="列出所有知识条目")
    group.add_argument("--prune", type=str, metavar="ID", help="删除指定 ID 的条目")
    group.add_argument("--stats", action="store_true", help="按 Tag 统计错误频率")

    args = parser.parse_args()

    if args.list:
        cli_list(args)
    elif args.prune:
        cli_prune(args)
    elif args.stats:
        cli_stats(args)


if __name__ == "__main__":
    main()
