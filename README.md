# Multi-Agent Simulation Workbench (多Agent协同仿真工作台)

基于 [AutoGen](https://github.com/microsoft/autogen) 框架的多 Agent 协同 FEniCS 仿真系统。用户提供自然语言描述，系统自动完成「输入解析 → 代码生成 → 执行校正 → 结果评估 → 分析报告」的完整工作流。

## 核心特性

- 🤖 **7 个专业 Agent** 协同分工（输入清晰化、结构化解析、代码构建、仿真执行、错误诊断、结果评估、力学洞察）
- 🔁 **自动纠错循环** — 执行失败后自动诊断并修复代码，支持多轮重试
- 📚 **错误记忆系统** — 从历史修复中学习，避免重复犯错
- 🌐 **双入口** — 支持 CLI 和 Web 界面（WebSocket 实时推送）
- 📎 **多格式文件解析** — 支持 PDF、Word、Excel、PPT、图片等输入
- 🎨 **ParaView 离屏渲染** — 自动将仿真结果渲染为高质量图片

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd Auto-Correct-test

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 2. CLI 模式

```bash
# 使用文本描述
python main.py --prompt "模拟一个 1×1 正方形区域的稳态热传导，上边界 100°C，下边界 0°C"

# 使用文件输入
python main.py --files input.pdf

# 使用默认 prompt 文件
python main.py --prompt_file examples/prompt.txt
```

### 3. Web 模式

```bash
python web_app.py
# 打开浏览器访问 http://localhost:8000
```

## 项目结构

```
Auto-Correct-test/
├── main.py                     # CLI 入口
├── web_app.py                  # Web 入口 (FastAPI + WebSocket)
├── pipeline.py                 # 核心工作流编排
│
├── agents/                     # Agent 子包
│   ├── base.py                 # Agent 公共基类
│   ├── input_clarifier.py      # 输入清晰化
│   ├── parsing.py              # 结构化解析
│   ├── code_builder.py         # 代码构建
│   ├── simulation_executor.py  # 仿真执行
│   ├── error_diagnosis.py      # 错误诊断
│   ├── mechanical_insight.py   # 力学洞察报告
│   └── result_evaluation.py    # 结果评估
│
├── core/                       # 核心工具
│   ├── config.py               # LLM 客户端工厂 + Prompt 加载
│   ├── models.py               # Pydantic 数据模型
│   ├── llm_utils.py            # LLM 输出解析与流式处理
│   └── utils.py                # 日志与目录工具
│
├── services/                   # 服务模块
│   ├── file_parser.py          # 多格式文件解析
│   ├── error_memory.py         # 错误记忆系统
│   └── paraview_render.py      # ParaView 离屏渲染
│
├── prompts/                    # Prompt 模板
├── static/                     # Web 前端 (HTML/CSS/JS)
├── memory/                     # 错误知识库持久化
├── examples/                   # 示例输入文件
├── tests/                      # 单元测试
├── .env.example                # 环境变量模板
└── requirements.txt            # Python 依赖
```

## 错误知识库管理

```bash
# 列出所有已知错误模式
python -m services.error_memory --list

# 查看按标签统计
python -m services.error_memory --stats

# 删除指定条目
python -m services.error_memory --prune <entry_id>
```

## 各 Agent 配置独立模型

在 `.env` 中为每个 Agent 指定不同的 LLM：

```env
# 全局默认
LLM_MODEL=qwen3.5-plus
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 让 CodeBuilder 使用 DeepSeek
CODE_BUILDER_MODEL=deepseek-chat
CODE_BUILDER_BASE_URL=https://api.deepseek.com
CODE_BUILDER_API_KEY=sk-xxx
```

## License

MIT License
