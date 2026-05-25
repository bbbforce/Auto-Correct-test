"""
多agent协同仿真工作流管道 —— 从 main.py 提取的核心编排逻辑。
支持 CLI 和 Web 两种入口，通过 on_event 回调推送实时状态。
"""

import asyncio
import os
import subprocess
import logging
from typing import Callable, Optional
from dotenv import load_dotenv

from agents.input_clarifier import InputClarifierAgent
from agents.parsing import ParsingAgent
from agents.code_builder import CodeBuilderAgent
from agents.simulation_executor import SimulationExecutorAgent
from agents.error_diagnosis import ErrorDiagnosisAgent
from agents.mechanical_insight import MechanicalInsightAgent
from agents.result_evaluation import ResultEvaluationAgent
from services.file_parser import parse_files as do_parse_files
from core.models import SimulationContext, PipelineResult
from core.utils import setup_logger, create_run_dir
from core.llm_utils import stream_callback_var

STEP_NAMES = {
    1: "输入清晰化",
    2: "结构化解析",
    3: "代码构建",
    4: "执行与校正",
    5: "分析报告",
}


def _probe_fenicsx_env() -> str:
    """P1-7a: 探测 fenicsx-env 的实际版本和可用库。"""
    probe_script = (
        "import dolfinx; print('dolfinx:', dolfinx.__version__); "
        "import ufl; print('ufl:', ufl.__version__); "
        "import basix; print('basix:', basix.__version__); "
        "try:\n import gmsh; print('gmsh: available')\n"
        "except ImportError: print('gmsh: NOT available')\n"
        "try:\n import pyvista; print('pyvista: available')\n"
        "except ImportError: print('pyvista: NOT available')"
    )
    try:
        res = subprocess.run(
            ["conda", "run", "-n", "fenicsx-env", "python", "-c", probe_script],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""


async def _emit(on_event, event: dict):
    """安全地发送事件。"""
    if on_event:
        try:
            await on_event(event)
        except Exception:
            pass  # WebSocket 断开等情况，静默忽略


def _make_stream_cb(on_event, step: int):
    """为指定步骤创建流式回调。"""
    async def cb(chunk: str):
        await _emit(on_event, {"type": "stream", "step": step, "chunk": chunk})
    return cb


async def _persist_repairs(context, logger, on_event):
    """将成功修复记录持久化到错误知识库。"""
    if not context.repair_history:
        return
    try:
        from services.error_memory import ErrorMemory
        memory = ErrorMemory()
        count = 0
        for record in context.repair_history:
            error_msg = record.get("error_message", "")
            if not error_msg:
                continue
            memory.add_entry(
                error_message=error_msg,
                root_cause=record.get("hint", ""),
                fix_description=record.get("hint", ""),
                code_before=record.get("code_before", ""),
                code_after=record.get("code_after", ""),
                confidence=record.get("confidence", 0.5),
            )
            count += 1
        if count:
            logger.info(f"Persisted {count} repair records to error memory")
            await _emit(on_event, {
                "type": "info", "step": 4,
                "message": f"已将 {count} 条修复经验写入知识库 📚"
            })
    except Exception as e:
        logger.warning(f"Failed to persist repairs to error memory: {e}")


async def _persist_failed_repairs(context, logger, on_event):
    """将失败的修复记录持久化到错误知识库（标记为 success=False）。"""
    if not context.repair_history:
        return
    try:
        from services.error_memory import ErrorMemory
        memory = ErrorMemory()
        count = 0
        for record in context.repair_history:
            error_msg = record.get("error_message", "")
            if not error_msg:
                continue
            memory.add_entry(
                error_message=error_msg,
                root_cause=record.get("hint", ""),
                fix_description=record.get("hint", ""),
                code_before=record.get("code_before", ""),
                code_after=record.get("code_after", ""),
                confidence=record.get("confidence", 0.5),
                success=False,
            )
            count += 1
        if count:
            logger.info(f"Persisted {count} failed repair records to error memory")
            await _emit(on_event, {
                "type": "info", "step": 4,
                "message": f"已将 {count} 条失败修复记录写入知识库 📚"
            })
    except Exception as e:
        logger.warning(f"Failed to persist failed repairs to error memory: {e}")


async def run_pipeline(
    prompt: str = "",
    prompt_file: str = None,
    files: list[str] = None,
    max_retries: int = 3,
    on_event: Callable = None,
) -> PipelineResult:
    """执行完整的多agent协同仿真管道。"""

    result = PipelineResult()

    # ── 环境与日志 ──
    load_dotenv()
    run_dir = create_run_dir()
    result.run_dir = run_dir
    result_dir = os.path.join(run_dir, "result")
    logger = setup_logger('pipeline', 'pipeline.log', log_dir=run_dir)

    await _emit(on_event, {"type": "info", "message": f"运行目录: {run_dir}"})

    default_api_key = os.environ.get("OPENAI_API_KEY")
    default_base_url = os.environ.get("LLM_BASE_URL", None)
    default_model = os.environ.get("LLM_MODEL", "gpt-4o")

    if not default_api_key:
        msg = "未找到 OPENAI_API_KEY 环境变量"
        await _emit(on_event, {"type": "pipeline_error", "message": msg})
        result.error_message = msg
        return result

    def agent_cfg(prefix: str):
        model = os.environ.get(f"{prefix}_MODEL") or default_model
        base_url = os.environ.get(f"{prefix}_BASE_URL") or default_base_url
        api_key = os.environ.get(f"{prefix}_API_KEY") or default_api_key
        return model, base_url, api_key

    # ── 创建 Agents ──
    cfg = {name: agent_cfg(name) for name in [
        "INPUT_CLARIFIER", "PARSING", "CODE_BUILDER",
        "ERROR_DIAGNOSIS", "MECHANICAL_INSIGHT", "RESULT_EVALUATION"
    ]}

    input_clarifier = InputClarifierAgent(api_key=cfg["INPUT_CLARIFIER"][2], model=cfg["INPUT_CLARIFIER"][0], base_url=cfg["INPUT_CLARIFIER"][1], log_dir=run_dir)
    parsing_agent = ParsingAgent(api_key=cfg["PARSING"][2], model=cfg["PARSING"][0], base_url=cfg["PARSING"][1], log_dir=run_dir)
    code_builder = CodeBuilderAgent(api_key=cfg["CODE_BUILDER"][2], model=cfg["CODE_BUILDER"][0], base_url=cfg["CODE_BUILDER"][1], log_dir=run_dir)
    executor = SimulationExecutorAgent(result_dir=result_dir, log_dir=run_dir)
    error_diagnosis = ErrorDiagnosisAgent(api_key=cfg["ERROR_DIAGNOSIS"][2], model=cfg["ERROR_DIAGNOSIS"][0], base_url=cfg["ERROR_DIAGNOSIS"][1], log_dir=run_dir)
    insight_agent = MechanicalInsightAgent(api_key=cfg["MECHANICAL_INSIGHT"][2], model=cfg["MECHANICAL_INSIGHT"][0], base_url=cfg["MECHANICAL_INSIGHT"][1], log_dir=run_dir)
    result_evaluator = ResultEvaluationAgent(api_key=cfg["RESULT_EVALUATION"][2], model=cfg["RESULT_EVALUATION"][0], base_url=cfg["RESULT_EVALUATION"][1], log_dir=run_dir)

    # ── P1-7a: 探测运行环境 ──
    env_info = ""
    try:
        env_info = _probe_fenicsx_env()
        if env_info:
            logger.info(f"FEniCSx environment:\n{env_info}")
            await _emit(on_event, {"type": "info", "message": f"环境探测完成: {env_info.splitlines()[0]}"})
    except Exception as e:
        logger.warning(f"Environment probe failed (non-fatal): {e}")

    # ── 解析附件 ──
    attached_images = []
    file_text = ""
    if files:
        logger.info(f"解析附件: {files}")
        parse_result = do_parse_files(files)
        file_text = parse_result.text
        attached_images = parse_result.images

    # ── 构建 prompt ──
    user_prompt = prompt or ""
    if not user_prompt and not files:
        if prompt_file:
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    user_prompt = f.read().strip()
            except FileNotFoundError:
                msg = f"提示文件未找到: {prompt_file}"
                await _emit(on_event, {"type": "pipeline_error", "message": msg})
                result.error_message = msg
                return result

    if file_text:
        if user_prompt:
            user_prompt = f"[从附件文件中提取的内容]\n{file_text}\n\n[用户补充说明]\n{user_prompt}"
        else:
            user_prompt = file_text

    if not user_prompt and not attached_images:
        msg = "输入不能为空：请提供文本描述或上传文件。"
        await _emit(on_event, {"type": "pipeline_error", "message": msg})
        result.error_message = msg
        return result

    logger.info(f"User prompt: {user_prompt[:300]}")

    # ══════════════════════════════════════════════════════
    # Step 1: 输入清晰化
    # ══════════════════════════════════════════════════════
    await _emit(on_event, {"type": "step_start", "step": 1, "name": STEP_NAMES[1]})
    token = stream_callback_var.set(_make_stream_cb(on_event, 1))
    try:
        clarified_input = await input_clarifier.clarify(user_prompt, images=attached_images or None)
    finally:
        stream_callback_var.reset(token)

    if not clarified_input:
        await _emit(on_event, {"type": "step_error", "step": 1, "message": "清晰化失败"})
        result.error_message = "清晰化失败"
        return result
    await _emit(on_event, {"type": "step_complete", "step": 1})

    # ══════════════════════════════════════════════════════
    # Step 2: 结构化解析
    # ══════════════════════════════════════════════════════
    await _emit(on_event, {"type": "step_start", "step": 2, "name": STEP_NAMES[2]})
    token = stream_callback_var.set(_make_stream_cb(on_event, 2))
    try:
        parsed_data = await parsing_agent.parse(clarified_input)
    finally:
        stream_callback_var.reset(token)

    if not parsed_data:
        await _emit(on_event, {"type": "step_error", "step": 2, "message": "解析失败"})
        result.error_message = "解析失败"
        return result
    await _emit(on_event, {"type": "step_complete", "step": 2})

    # ══════════════════════════════════════════════════════
    # Step 3: 代码构建
    # ══════════════════════════════════════════════════════
    await _emit(on_event, {"type": "step_start", "step": 3, "name": STEP_NAMES[3]})
    token = stream_callback_var.set(_make_stream_cb(on_event, 3))
    try:
        generated_code = await code_builder.build_code(parsed_data, env_info=env_info)
    finally:
        stream_callback_var.reset(token)

    if not generated_code:
        await _emit(on_event, {"type": "step_error", "step": 3, "message": "代码生成失败"})
        result.error_message = "代码生成失败"
        return result
    result.generated_code = generated_code
    await _emit(on_event, {"type": "step_complete", "step": 3})

    # ══════════════════════════════════════════════════════
    # Step 4: 执行与校正循环
    # ══════════════════════════════════════════════════════
    await _emit(on_event, {"type": "step_start", "step": 4, "name": STEP_NAMES[4]})
    context = SimulationContext(current_code=generated_code)

    for attempt in range(max_retries):
        await _emit(on_event, {
            "type": "attempt_start", "step": 4,
            "attempt": attempt + 1, "max_retries": max_retries
        })

        executor.save_code_to_file(context.current_code)
        exec_result = await asyncio.to_thread(executor.execute_simulation)

        await _emit(on_event, {
            "type": "execution_result", "step": 4,
            "status": exec_result.status,
            "output": exec_result.output[:500]
        })

        # 收集新生成的图片
        for f in exec_result.files:
            url = f"/files/{f}" if not f.startswith("/") else f
            result.result_images.append(url)
            await _emit(on_event, {"type": "image", "step": 4, "path": url})

        if exec_result.status in ("error", "timeout"):
            if attempt < max_retries - 1:
                # ── P1-5c: 先尝试规则自动修复，省去 LLM 调用 ──
                from services.code_validator import auto_fix_common_errors
                auto_fixed, fix_descriptions = auto_fix_common_errors(
                    context.current_code, exec_result.output
                )
                if fix_descriptions and auto_fixed != context.current_code:
                    logger.info(f"规则自动修复应用了 {len(fix_descriptions)} 项: {fix_descriptions}")
                    await _emit(on_event, {
                        "type": "info", "step": 4,
                        "message": f"规则自动修复: {'; '.join(fix_descriptions[:3])} 🔧"
                    })
                    code_before_fix = context.current_code
                    context.current_code = auto_fixed
                    context.add_repair_record(
                        f"[规则自动修复] {'; '.join(fix_descriptions)}", 0.8,
                        error_message=exec_result.output,
                        code_before=code_before_fix,
                        code_after=auto_fixed,
                    )
                    await _emit(on_event, {"type": "info", "step": 4, "message": "规则修复已应用，重新执行... 🔄"})
                    continue  # 跳过 LLM 诊断，直接重试执行

                # 规则修复无效，回退到 LLM 诊断
                await _emit(on_event, {"type": "info", "step": 4, "message": "正在诊断错误..."})
                token = stream_callback_var.set(_make_stream_cb(on_event, 4))
                try:
                    diagnosis = await error_diagnosis.diagnose_and_fix(
                        error_message=exec_result.output, code=context.current_code,
                        simulation_output=exec_result.output, iteration=attempt,
                        repair_history=context.repair_history, log_dir=run_dir,
                    )
                finally:
                    stream_callback_var.reset(token)
                code_before_fix = context.current_code
                context.add_repair_record(
                    diagnosis.hint, diagnosis.confidence,
                    error_message=exec_result.output,
                    code_before=code_before_fix,
                    code_after=diagnosis.after_code or code_before_fix,
                )
                if diagnosis.after_code:
                    context.current_code = diagnosis.after_code
                    await _emit(on_event, {"type": "info", "step": 4, "message": "代码已修复 ✅"})
        else:
            # 执行成功 → 评估
            await _emit(on_event, {"type": "info", "step": 4, "message": "执行成功，开始评估..."})
            token = stream_callback_var.set(_make_stream_cb(on_event, 4))
            try:
                eval_result = await result_evaluator.evaluate_results(
                    prompt=user_prompt, code=context.current_code,
                    simulation_output=exec_result.output, image_paths=exec_result.files,
                )
            finally:
                stream_callback_var.reset(token)

            if not context.best_code or eval_result.confidence > context.best_confidence:
                context.update_best(context.current_code, eval_result.confidence, exec_result.output)

            if eval_result.is_correct:
                context.simulation_success = True
                context.final_output = exec_result.output
                await _emit(on_event, {"type": "info", "step": 4, "message": "仿真结果正确 ✅"})
                # 持久化成功的修复记录到知识库
                await _persist_repairs(context, logger, on_event)
                break
            else:
                await _emit(on_event, {"type": "info", "step": 4, "message": f"评估未通过: {eval_result.feedback[:200]}"})
                if attempt < max_retries - 1:
                    token = stream_callback_var.set(_make_stream_cb(on_event, 4))
                    try:
                        diagnosis = await error_diagnosis.diagnose_and_fix(
                            error_message="Physical/Logical Error: " + eval_result.feedback,
                            code=context.current_code, simulation_output=exec_result.output,
                            iteration=attempt, repair_history=context.repair_history, log_dir=run_dir,
                        )
                    finally:
                        stream_callback_var.reset(token)
                    code_before_fix = context.current_code
                    context.add_repair_record(
                        diagnosis.hint, diagnosis.confidence,
                        error_message="Physical/Logical Error: " + eval_result.feedback,
                        code_before=code_before_fix,
                        code_after=diagnosis.after_code or code_before_fix,
                    )
                    if diagnosis.after_code:
                        context.current_code = diagnosis.after_code

    # Fallback
    if not context.simulation_success and context.best_code:
        executor.save_code_to_file(context.best_code)
        context.current_code = context.best_code

    # 失败时也持久化修复记录（标记为失败）
    if not context.simulation_success and context.repair_history:
        await _persist_failed_repairs(context, logger, on_event)

    await _emit(on_event, {"type": "step_complete", "step": 4})
    result.generated_code = context.current_code

    # ══════════════════════════════════════════════════════
    # Step 5: 分析报告
    # ══════════════════════════════════════════════════════
    if context.simulation_success or context.best_code:
        await _emit(on_event, {"type": "step_start", "step": 5, "name": STEP_NAMES[5]})
        token = stream_callback_var.set(_make_stream_cb(on_event, 5))
        try:
            report = await insight_agent.generate_report(context.current_code)
        finally:
            stream_callback_var.reset(token)

        report_path = os.path.join(run_dir, "simulation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        result.report = report
        result.final_output = context.final_output
        result.success = True
        await _emit(on_event, {"type": "step_complete", "step": 5})
    else:
        result.error_message = "仿真工作流完全失败，无法生成报告。"

    await _emit(on_event, {
        "type": "pipeline_complete",
        "success": result.success,
        "report": result.report,
        "output": result.final_output,
        "run_dir": result.run_dir,
    })

    return result
