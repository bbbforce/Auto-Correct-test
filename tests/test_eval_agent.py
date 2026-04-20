"""
独立测试脚本：仅测试 ResultEvaluationAgent 是否能正常工作。
不运行整个仿真流程，直接使用已有的结果文件和模拟数据来调用评估智能体。

用法：
    conda run -n autogen-env python test_eval_agent.py
"""

import asyncio
import os
from dotenv import load_dotenv
from agents.result_evaluation import ResultEvaluationAgent

async def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "gpt-4o")
    base_url = os.environ.get("LLM_BASE_URL", None)

    if not api_key:
        print("❌ 未找到 OPENAI_API_KEY，请检查 .env 文件")
        return

    # 1. 初始化评估智能体
    print(f"✅ 初始化 ResultEvaluationAgent，模型: {model_name}")
    evaluator = ResultEvaluationAgent(api_key=api_key, model=model_name, base_url=base_url)

    # 2. 准备模拟输入数据（不需要真正跑 FEniCS）
    mock_prompt = "Calculate the 2D steady state heat transfer on a 1x1 square domain. Top=100°C, Bottom=0°C, others insulated."
    
    mock_code = """
from fenics import *
import matplotlib.pyplot as plt
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)
u_D_top = Constant(100.0)
u_D_bottom = Constant(0.0)
bc_top = DirichletBC(V, u_D_top, 'near(x[1], 1.0)')
bc_bottom = DirichletBC(V, u_D_bottom, 'near(x[1], 0.0)')
bcs = [bc_top, bc_bottom]
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
L = Constant(0.0) * v * dx
u = Function(V)
solve(a == L, u, bcs)
print(f"MAX_TEMPERATURE: {u.vector().max()}")
print(f"MIN_TEMPERATURE: {u.vector().min()}")
"""

    mock_output = "MAX_TEMPERATURE: 100.0\nMIN_TEMPERATURE: 0.0"

    # 3. 收集已有的图片文件
    image_paths = []
    result_dir = "result"
    if os.path.exists(result_dir):
        for f in os.listdir(result_dir):
            if f.endswith(".png"):
                image_paths.append(os.path.join(result_dir, f))

    print(f"📷 找到 {len(image_paths)} 张结果图片: {image_paths}")

    # 4. 调用评估
    print("🔍 正在调用 ResultEvaluationAgent.evaluate_results()...")
    result = await evaluator.evaluate_results(
        prompt=mock_prompt,
        code=mock_code,
        simulation_output=mock_output,
        image_paths=image_paths
    )

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("📋 评估结果:")
    print(f"  is_correct : {result.get('is_correct')}")
    print(f"  confidence : {result.get('confidence')}")
    print(f"  feedback   : {result.get('feedback')}")
    if result.get('after_code'):
        print(f"  after_code : (包含 {len(result['after_code'])} 字符的修正代码)")
    else:
        print(f"  after_code : (无)")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
