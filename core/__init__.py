"""
核心工具包 —— 配置、数据模型、LLM 工具、通用工具。
"""

import os

# 项目根目录（供需要定位 prompts/、memory/ 等目录的模块使用）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
