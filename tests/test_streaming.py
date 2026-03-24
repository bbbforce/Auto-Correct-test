import asyncio
import os
from dotenv import load_dotenv

from input_clarifier_agent import InputClarifierAgent

async def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "gpt-4o")
    base_url = os.environ.get("LLM_BASE_URL", None)

    if not api_key:
        print("❌ Please set OPENAI_API_KEY")
        return

    print(f"✅ Initializing InputClarifierAgent with model: {model_name}")
    agent = InputClarifierAgent(api_key=api_key, model=model_name, base_url=base_url)

    mock_input = "计算 1×1 正方形区域内的二维稳态传热。热传导系数 k=1。上边界温度恒定为 100 摄氏度，下边界温度恒定为 0 摄氏度，其余边界绝热。输出最高温度与最低温度，并保存温度分布的图像"
    
    print("🔍 Sending request to agent...")
    clarified = await agent.clarify(mock_input)
    
    print("\n" + "=" * 60)
    print("📋 Final Received Output:")
    print(clarified)
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
