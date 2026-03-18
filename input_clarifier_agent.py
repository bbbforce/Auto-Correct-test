from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from utils import setup_logger


class InputClarifierAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('input_clarifier_agent', 'input_clarifier_agent.log', log_dir=log_dir)
        self.model_client = OpenAIChatCompletionClient(
            model=model, 
            api_key=api_key, 
            base_url=base_url,
            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},
            temperature=0.2
        )
        system_message = """
You are given a user's simulation request in natural language. Based on the following guidelines, generate a **single-paragraph, fully specified simulation description** suitable for direct use in FEniCS.

📌 Carefully consider the following aspects when clarifying:
──────────────────────────────────────────────────────

🔷 Problem Type and Structure
- Identify the PDE type: heat, fluid (Navier-Stokes), elasticity, fracture, reaction-diffusion, etc.
- Specify whether the problem is steady or time-dependent.
- Determine if it is multiphysics (e.g., thermo-elasticity, fluid-structure interaction, electro-mechanics)..
- Identify spatial dimension: 2D or 3D.
- Describe the domain shape: rectangle, circle, cylinder, presence of notch, etc.

🔷 Field Variables and Conditions
- Define field variables such as u, p, T, d, k, etc.
- Infer and describe boundary and initial conditions clearly.
- Estimate required material properties: E, nu, k, rho, cp, mu, Gc, etc.

🔷 Numerical Settings
- Suggest appropriate time step dt (if transient).
- Recommend solver structure (e.g., nonlinear Newton solver, staggered scheme).
- Mention output format (e.g., whether to store results in .xdmf).

📦 Format your output as one complete paragraph in clear technical English.
📦 Do NOT include section headings or bullet points.
"""
        self.agent = AssistantAgent(
            name="input_clarifier_agent",
            model_client=self.model_client,
            system_message=system_message
        )

    async def clarify(self, raw_input: str) -> str:
        try:
            prompt = f"[User Request]\n{raw_input}\n\n[Refined Simulation Specification]\n"
            self.logger.info("Clarifying input: %s", raw_input)
            
            result = await self.agent.run(task=prompt)
            refined = result.messages[-1].content.strip()
            
            self.logger.info("Clarified result: %s", refined)
            return refined
        except Exception as e:
            self.logger.error("Clarification failed: %s", str(e))
            return raw_input  # fallback
