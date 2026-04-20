"""
CLI 入口 —— 通过命令行启动仿真管道。
"""

import asyncio
import argparse
from pipeline import run_pipeline


async def main():
    parser = argparse.ArgumentParser(description="多agent协同仿真工作台")
    parser.add_argument("--prompt", type=str, help="输入对模拟过程的自然语言描述", default=None)
    parser.add_argument("--prompt_file", type=str, help="输入包含模拟提示词的文本文件路径", default="examples/prompt.txt")
    parser.add_argument("--files", nargs="+", type=str, help="上传文件路径（支持图片、PDF、Word、Excel、PPT等）", default=[])
    parser.add_argument("--max_retries", type=int, default=3, help="设置代码执行的最大重试次数")
    args = parser.parse_args()

    async def terminal_handler(event):
        """将管道事件格式化输出到终端。"""
        t = event.get("type")
        if t == "step_start":
            print("\n" + "=" * 80)
            print(f"{'--- Step ' + str(event['step']) + ': ' + event['name'] + ' ---':^80}")
            print("=" * 80 + "\n")
        elif t == "attempt_start":
            print(f"\n  🔁 第 {event['attempt']}/{event['max_retries']} 次尝试")
        elif t == "execution_result":
            status_icon = {"success": "✅", "error": "❌", "timeout": "⏰"}.get(event["status"], "❓")
            print(f"\n  {status_icon} 执行结果: {event['status']}")
        elif t == "info":
            print(f"  ℹ️  {event['message']}")
        elif t == "pipeline_error":
            print(f"\n  ❌ 错误: {event['message']}")
        elif t == "pipeline_complete":
            if event.get("report"):
                print("\n" + "=" * 80)
                print("--- 分析报告 ---")
                print("=" * 80)
                print(event["report"])
            if event.get("output"):
                print("\n--- 最终输出 ---")
                print(event["output"])

    result = await run_pipeline(
        prompt=args.prompt or "",
        prompt_file=args.prompt_file if not args.prompt and not args.files else None,
        files=args.files or [],
        max_retries=args.max_retries,
        on_event=terminal_handler,
    )

    if not result.success:
        print(f"\n❌ 仿真失败: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
