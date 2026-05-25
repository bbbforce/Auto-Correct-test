"""
DOLFINx API Database - 加载与查询工具

用法:
    from dolfinx_api_db.loader import load_db, search, get_module
    
    db = load_db()                      # 加载全部数据库
    results = search(db, "LinearProblem")  # 搜索 API
    mod = get_module(db, "dolfinx.fem")    # 获取指定模块
"""

import json
import os
from pathlib import Path
from typing import Any

DB_DIR = Path(__file__).parent


def load_db() -> dict[str, Any]:
    """加载所有模块 JSON 文件，返回以模块名为键的字典。"""
    db = {}
    meta_path = DB_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            db["_metadata"] = json.load(f)

    for fpath in sorted(DB_DIR.glob("dolfinx_*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        module_name = data.get("module", fpath.stem)
        db[module_name] = data
    return db


def get_module(db: dict, module_name: str) -> dict | None:
    """按模块名获取 API 数据。"""
    return db.get(module_name)


def search(db: dict, query: str, case_insensitive: bool = True) -> list[dict]:
    """在所有模块中搜索类或函数名称，返回匹配列表。"""
    results = []
    q = query.lower() if case_insensitive else query

    for mod_name, mod_data in db.items():
        if mod_name.startswith("_"):
            continue

        # 搜索类
        for cls in mod_data.get("classes", []):
            name = cls.get("name", "")
            cmp = name.lower() if case_insensitive else name
            if q in cmp:
                results.append({
                    "type": "class",
                    "module": mod_name,
                    "name": name,
                    "qualified_name": cls.get("qualified_name", ""),
                    "description": cls.get("description", ""),
                })

        # 搜索函数
        for func in mod_data.get("functions", []):
            name = func.get("name", "")
            cmp = name.lower() if case_insensitive else name
            if q in cmp:
                results.append({
                    "type": "function",
                    "module": mod_name,
                    "name": name,
                    "qualified_name": func.get("qualified_name", ""),
                    "signature": func.get("signature", ""),
                    "description": func.get("description", ""),
                })

        # 搜索枚举
        for enum in mod_data.get("enums", []):
            name = enum.get("name", "")
            cmp = name.lower() if case_insensitive else name
            if q in cmp:
                results.append({
                    "type": "enum",
                    "module": mod_name,
                    "name": name,
                    "qualified_name": enum.get("qualified_name", ""),
                    "values": enum.get("values", []),
                })

    return results


def list_all_apis(db: dict) -> list[str]:
    """列出数据库中所有 API 的 qualified_name。"""
    apis = []
    for mod_name, mod_data in db.items():
        if mod_name.startswith("_"):
            continue
        for cls in mod_data.get("classes", []):
            apis.append(cls.get("qualified_name", cls.get("name")))
        for func in mod_data.get("functions", []):
            apis.append(func.get("qualified_name", func.get("name")))
        for enum in mod_data.get("enums", []):
            apis.append(enum.get("qualified_name", enum.get("name")))
    return sorted(apis)


def summary(db: dict) -> dict:
    """返回数据库统计概要。"""
    stats = {"modules": 0, "classes": 0, "functions": 0, "enums": 0}
    for mod_name, mod_data in db.items():
        if mod_name.startswith("_"):
            continue
        stats["modules"] += 1
        stats["classes"] += len(mod_data.get("classes", []))
        stats["functions"] += len(mod_data.get("functions", []))
        stats["enums"] += len(mod_data.get("enums", []))
    return stats


if __name__ == "__main__":
    db = load_db()
    s = summary(db)
    print(f"DOLFINx API Database v{db.get('_metadata', {}).get('version', 'unknown')}")
    print(f"  模块: {s['modules']}, 类: {s['classes']}, 函数: {s['functions']}, 枚举: {s['enums']}")
    print(f"  总 API 条目: {len(list_all_apis(db))}")
    print()
    
    # 演示搜索
    for q in ["Function", "mesh", "assemble"]:
        hits = search(db, q)
        print(f'搜索 "{q}" → {len(hits)} 条结果')
        for h in hits[:3]:
            print(f"  [{h['type']}] {h['qualified_name']}")
        if len(hits) > 3:
            print(f"  ... 还有 {len(hits)-3} 条")
        print()
