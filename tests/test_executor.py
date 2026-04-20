"""tests/test_executor.py — SimulationExecutorAgent 的单元测试。"""
import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.simulation_executor import SimulationExecutorAgent
from core.models import ExecutionResult


class TestSimulationExecutorAgent:
    """测试 SimulationExecutorAgent 的执行与超时保护。"""

    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.result_dir = os.path.join(self.test_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)
        self.agent = SimulationExecutorAgent(result_dir=self.result_dir)

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_code_to_file(self):
        code = "print('hello world')"
        self.agent.save_code_to_file(code)
        with open(self.agent.simulation_filename, "r") as f:
            assert f.read() == code

    @patch("agents.simulation_executor.subprocess.run")
    def test_successful_execution(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="MAX_TEMPERATURE: 100.0\nMIN_TEMPERATURE: 0.0",
            stderr="",
        )
        result = self.agent.execute_simulation()
        assert isinstance(result, ExecutionResult)
        assert result.status == "success"
        assert "MAX_TEMPERATURE" in result.output

    @patch("agents.simulation_executor.subprocess.run")
    def test_error_detection_in_stdout(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="Traceback (most recent call last):\n  ...\nNameError: name 'x' is not defined",
            stderr="",
        )
        result = self.agent.execute_simulation()
        assert result.status == "error"
        assert "Traceback" in result.output

    @patch("agents.simulation_executor.subprocess.run")
    def test_timeout_handling(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python test.py", timeout=300)
        result = self.agent.execute_simulation()
        assert result.status == "timeout"
        assert "超时" in result.output

    @patch("agents.simulation_executor.subprocess.run")
    def test_called_process_error(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="python test.py", stderr="Segfault!"
        )
        result = self.agent.execute_simulation()
        assert result.status == "error"

    def test_collect_new_files(self):
        # 在 result 目录放一个 png
        test_png = os.path.join(self.result_dir, "test.png")
        with open(test_png, "w") as f:
            f.write("fake png")
        before = set()  # 之前为空
        new_files = self.agent._collect_new_files(before)
        assert len(new_files) == 1
        assert "test.png" in new_files[0]

    def test_collect_new_files_ignores_non_images(self):
        # 非图片文件应被忽略
        test_txt = os.path.join(self.result_dir, "data.txt")
        with open(test_txt, "w") as f:
            f.write("not an image")
        before = set()
        new_files = self.agent._collect_new_files(before)
        assert len(new_files) == 0


class TestExecutionResultModel:
    def test_valid_statuses(self):
        for status in ("success", "error", "timeout"):
            r = ExecutionResult(status=status)
            assert r.status == status

    def test_invalid_status(self):
        with pytest.raises(Exception):
            ExecutionResult(status="unknown")
