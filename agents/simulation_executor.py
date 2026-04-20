"""Agent: 仿真执行 —— 运行生成的 FEniCS 代码并收集结果。"""

import subprocess
import os
import re
import json
from core.utils import setup_logger
from core.models import ExecutionResult
from core import PROJECT_ROOT

EXECUTION_TIMEOUT = 300  # 5分钟超时

# ── ParaView OsMesa 离屏渲染配置 ──
PVPYTHON = os.environ.get("PVPYTHON_PATH", "/opt/paraview-osmesa/bin/pvpython")
PARAVIEW_RENDER_SCRIPT = os.path.join(PROJECT_ROOT, "services", "paraview_render.py")
PARAVIEW_RENDER_TIMEOUT = 60  # 秒
PARAVIEW_ENV = {
    "GALLIUM_DRIVER": "llvmpipe",
    "MESA_GL_VERSION_OVERRIDE": "3.3",
}


class SimulationExecutorAgent:
    """仿真执行器（非 LLM Agent，不继承 BaseAgent）。"""

    def __init__(self, result_dir: str = "result", log_dir: str = None):
        self.logger = setup_logger('simulation_executor_agent', 'simulation_executor_agent.log', log_dir=log_dir)
        self.simulation_filename = "generated_simulation.py"
        self.result_dir = result_dir

    def save_code_to_file(self, code: str):
        """将生成的 Python 仿真代码保存到文件中。"""
        with open(self.simulation_filename, "w", encoding="utf-8") as f:
            f.write(code)
        self.logger.info(f"Simulation code saved to {self.simulation_filename}")

    def execute_simulation(self) -> ExecutionResult:
        """执行仿真脚本，带超时保护。

        Returns:
            ExecutionResult: 结构化的执行结果，status 为 success / error / timeout。
        """
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # 快照执行前的文件
        before_files = set(os.listdir(self.result_dir))

        try:
            res = subprocess.run(
                ["conda", "run", "-n", "fenics-env", "python", self.simulation_filename],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT,
                env={**os.environ, "RESULT_DIR": self.result_dir},
            )
            stdout = res.stdout.strip()
            stderr = res.stderr.strip()

            new_file_paths = self._collect_new_files(before_files)

            if res.returncode != 0:
                combined_error = "\n".join(filter(None, [stdout, stderr]))
                self.logger.error(f"Simulation execution failed with return code {res.returncode}.")
                self.logger.error(f"Simulation error output: {combined_error}")
                return ExecutionResult(status="error", output=combined_error, files=new_file_paths)

            self.logger.info("Simulation executed successfully.")
            self.logger.info(f"Simulation output: {stdout}")

            # ── ParaView 离屏渲染 ──
            pv_files = self._run_paraview_render()
            new_file_paths.extend(pv_files)

            return ExecutionResult(status="success", output=stdout, files=new_file_paths)

        except subprocess.TimeoutExpired:
            self.logger.error(f"Simulation execution timed out after {EXECUTION_TIMEOUT}s.")
            new_file_paths = self._collect_new_files(before_files)
            return ExecutionResult(
                status="timeout",
                output=f"执行超时（{EXECUTION_TIMEOUT}秒）。可能原因：网格过密、时间步过多或代码死循环。",
                files=new_file_paths,
            )

        except Exception as e:
            self.logger.error(f"Simulation execution failed with exception: {e}")
            new_file_paths = self._collect_new_files(before_files)
            return ExecutionResult(status="error", output=str(e), files=new_file_paths)

    def _run_paraview_render(self) -> list[str]:
        """调用 pvpython 对 result_dir 中的 XDMF 文件进行离屏渲染。

        Returns:
            渲染生成的 PNG 文件路径列表。失败时返回空列表。
        """
        if not os.path.exists(PVPYTHON):
            self.logger.warning(f"pvpython not found at {PVPYTHON}, skipping ParaView render.")
            return []

        if not os.path.exists(PARAVIEW_RENDER_SCRIPT):
            self.logger.warning(f"Render script not found: {PARAVIEW_RENDER_SCRIPT}")
            return []

        # 构建环境变量：继承当前环境 + 追加 OsMesa 专用变量
        env = {**os.environ, **PARAVIEW_ENV}
        lib_path = "/opt/paraview-osmesa/lib/mesa:/opt/paraview-osmesa/lib"
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")

        try:
            self.logger.info(f"Starting ParaView render for: {self.result_dir}")
            res = subprocess.run(
                [PVPYTHON, PARAVIEW_RENDER_SCRIPT, self.result_dir],
                capture_output=True,
                text=True,
                timeout=PARAVIEW_RENDER_TIMEOUT,
                env=env,
            )

            # stderr 是日志，stdout 是 JSON 结果
            if res.stderr:
                self.logger.info(f"ParaView render log:\n{res.stderr.strip()}")

            if res.returncode != 0:
                self.logger.error(f"ParaView render failed (exit {res.returncode})")
                return []

            # 解析 stdout 中的 JSON 路径列表
            rendered = json.loads(res.stdout.strip())
            self.logger.info(f"ParaView rendered {len(rendered)} image(s): {rendered}")
            return rendered

        except subprocess.TimeoutExpired:
            self.logger.warning(f"ParaView render timed out ({PARAVIEW_RENDER_TIMEOUT}s), skipping.")
            return []
        except Exception as e:
            self.logger.error(f"ParaView render error: {e}")
            return []

    def _collect_new_files(self, before_files: set) -> list[str]:
        """收集执行后新生成的图片文件。"""
        if not os.path.exists(self.result_dir):
            return []
        after_files = set(os.listdir(self.result_dir))
        new_files = list(after_files - before_files)
        return [
            os.path.join(self.result_dir, f)
            for f in new_files
            if f.endswith('.png') or f.endswith('.jpg')
        ]
