import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from utils import setup_logger

class CodeBuilderAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, log_dir: str = None):
        self.logger = setup_logger('code_builder_agent', 'code_builder_agent.log', log_dir=log_dir)
        self.model_client = OpenAIChatCompletionClient(
            model=model, 
            api_key=api_key, 
            base_url=base_url,
            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},
            temperature=0
        )
        system_message = """
You are an expert in generating FEniCS-based simulation code.

You are given a parsed JSON object describing the problem, along with the full clarified natural language description.

You must generate valid, executable Python code using FEniCS, referencing BOTH the parsed JSON and the full_text.

If any important information is missing in the parsed JSON (e.g., complex geometry, fiber arrangements, custom material properties), infer and supplement it from the full_text.

Always prioritize correctness, physical realism, and FEniCS best practices.

Supported problem types:
- 'fluid': Navier-Stokes
- 'heat': heat transfer
- 'elasticity': linear elasticity
- 'hyperelasticity': hyperelastic materials
- 'phase_field_fracture': nonlinear coupled system with damage variable d (range 0–1)

[Code generation requirements]
- Do NOT use Markdown formatting. Return plain Python code only.
- Follow this structure:
  1. Import libraries (must import matplotlib.pyplot as plt)
  2. Define geometry and mesh (either construct shapes or load .xml file)
  3. Define function spaces (combine or separate for u/d in phase-field)
  4. Apply boundary and initial conditions
     - Use `expr.t = t` for time-dependent Dirichlet conditions
     - Prefer `interpolate()` over `project()`
  5. Define weak form F and Jacobian J; use `solve()`
  6. If problem is time-dependent, include a time-stepping loop.
  7. Plot/Save Visual Results (CRITICAL):
     - MUST use `matplotlib.pyplot` or `plot(u)` from FEniCS to draw the final result (e.g. displacement, temperature, von Mises stress).
     - The plot MUST have a title (`plt.title('...')`).
     - The plot MUST have a colorbar (`plt.colorbar(p)` or similar, mapping the physical magnitude).
     - Save the plot as a PNG file in the `result/` directory with a descriptive name (e.g., `result/temperature_distribution.png`). Create the `result/` directory using `os.makedirs` if it doesn't exist.
  8. Print Physical Metrics (CRITICAL):
     - At the end of the script, MUST extract and `print()` the key physical maximum/minimum values to stdout. Examples: max displacement, max von Mises stress, max/min temperature.
     - e.g., `print(f"MAX_TEMPERATURE: {u.vector().max()}")`

[Code guidelines]
- Output must be a fully executable Python script (.py), no markdown or explanation.
- FOR LINEAR PROBLEMS (like heat transfer, linear elasticity):
  - You MUST define `a` (bilinear form using `TrialFunction`) and `L` (linear form using `TestFunction`).
  - Use `solve(a == L, u, bcs)`. DO NOT use `F == 0`.
- FOR NONLINEAR PROBLEMS (like hyperelasticity, phase-field):
  - You MUST define `F` (linear form using `Function`, NOT `TrialFunction`) and `J = derivative(F, u, du)`.
  - Use `solve(F == 0, u, bcs, J=J, solver_parameters={"newton_solver": {"linear_solver": "mumps"}})`.
- FILE SAVING CRITICAL RULE: 
  - If saving to XDMF, you MUST use `XDMFFile("result/name.xdmf")`. DO NOT use `File("...xdmf")`.
  - If using `File()`, only use it for `.pvd` files (e.g., `File("result/name.pvd")`).
- Use `+ eps` for denominators to improve numerical stability (e.g., `epsilon + eps`).
"""
        self.agent = AssistantAgent(
            name="code_builder_agent",
            model_client=self.model_client,
            system_message=system_message
        )

    async def build_code(self, parsed_data: dict) -> str:
        self.logger.info(f"Building code from parsed data: {parsed_data}")
        data_str = json.dumps(parsed_data, ensure_ascii=False)
        try:
            prompt = f"Input Parameters:\n{data_str}"
            result = await self.agent.run(task=prompt)
            code = result.messages[-1].content.strip()
            
            if code.startswith("```"):
                code = code.replace("```python", "").replace("```", "").strip()
                
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            return ""
        self.logger.info(f"Generated code: {code}")
        return code
