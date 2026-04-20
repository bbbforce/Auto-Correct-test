import asyncio
import os
from dotenv import load_dotenv

from agents.mechanical_insight import MechanicalInsightAgent

async def main():
    if not load_dotenv(".env"): 
        print("Warning: no .env file found.")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "qwen3.5-plus")
    base_url = os.environ.get("LLM_BASE_URL", None)

    if not api_key:
        print("❌ Please set OPENAI_API_KEY")
        return

    print(f"✅ Initializing MechanicalInsightAgent with model: {model_name}")
    agent = MechanicalInsightAgent(api_key=api_key, model=model_name, base_url=base_url)

    mock_code = "print('Hello, FEniCSx!')"
    
    print("🔍 Sending request to agent...")
    report = await agent.generate_report(mock_code, language="中文")
    
    print("\n" + "=" * 60)
    print("📋 Final Received Output:")
    print(report)
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
