# simulation_executor_agent.py
import subprocess
import os
import re
from utils import setup_logger
from models import ExecutionResult

EXECUTION_TIMEOUT = 300  # 5分钟超时


class SimulationExecutorAgent:
    def __init__(self, result_dir: str = "result", log_dir: str = None):
        self.logger = setup_logger('simulation_executor_agent', 'simulation_executor_agent.log', log_dir=log_dir)
        self.simulation_filename = "generated_simulation.py"
        self.result_dir = result_dir

    def save_code_to_file(self, code: str):
        """Save the generated Python simulation code to a file.(将生成的 Python 仿真代码保存到文件中。)"""
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