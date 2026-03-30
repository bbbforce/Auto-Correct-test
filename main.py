import asyncio
import os
import logging
import argparse
from dotenv import load_dotenv

from input_clarifier_agent import InputClarifierAgent
from parsing_agent import ParsingAgent
from code_builder_agent import CodeBuilderAgent
from simulation_executor_agent import SimulationExecutorAgent
from error_diagnosis_agent import ErrorDiagnosisAgent
from mechanical_insight_agent import MechanicalInsightAgent
from result_evaluation_agent import ResultEvaluationAgent
from file_parser import parse_files
from utils import setup_logger, create_run_dir


async def main():
    parser = argparse.ArgumentParser(description="MCP-SIM Asynchronous Orchestrator")
    parser.add_argument("--prompt", type=str, help="输入对模拟过程的自然语言描述", default=None)
    parser.add_argument("--prompt_file", type=str, help="输入包含模拟提示词的文本文件路径", default="prompt.txt")
    parser.add_argument("--files", nargs="+", type=str, help="上传文件路径（支持图片、PDF、Word、Excel、PPT等）", default=[])
    parser.add_argument("--max_retries", type=int, default=3, help="设置代码执行的最大重试次数")
    args = parser.parse_args()

    # 创建本次运行的独立输出目录
    run_dir = create_run_dir()
    result_dir = os.path.join(run_dir, "result")

    logger = setup_logger('main_execution', 'main_execution.log', log_dir=run_dir)
    logger.info("="*50)
    logger.info(f"Run directory: {run_dir}")
    logger.info("="*50 + "\n")

    # Load environment variables (加载环境变量)
    load_dotenv()
    default_api_key  = os.environ.get("OPENAI_API_KEY")
    default_base_url = os.environ.get("LLM_BASE_URL", None)
    default_model    = os.environ.get("LLM_MODEL", "gpt-4o")

    if not default_api_key:
        logger.error("未找到 OPENAI_API_KEY 环境变量。请在.env 文件或环境中进行设置")
        return

    def agent_cfg(prefix: str):
        """读取 Agent 专属配置，未设置则回退到全局默认。"""
        model    = os.environ.get(f"{prefix}_MODEL")    or default_model
        base_url = os.environ.get(f"{prefix}_BASE_URL") or default_base_url
        api_key  = os.environ.get(f"{prefix}_API_KEY")  or default_api_key
        return model, base_url, api_key

    cfg_input_clarifier    = agent_cfg("INPUT_CLARIFIER")
    cfg_parsing            = agent_cfg("PARSING")
    cfg_code_builder       = agent_cfg("CODE_BUILDER")
    cfg_error_diagnosis    = agent_cfg("ERROR_DIAGNOSIS")
    cfg_mechanical_insight = agent_cfg("MECHANICAL_INSIGHT")
    cfg_result_evaluation  = agent_cfg("RESULT_EVALUATION")

    logger.info("="*50)
    logger.info(f"全局默认: model={default_model}  base_url={default_base_url}")
    logger.info(f"  InputClarifierAgent    -> model={cfg_input_clarifier[0]}  base_url={cfg_input_clarifier[1]}")
    logger.info(f"  ParsingAgent           -> model={cfg_parsing[0]}  base_url={cfg_parsing[1]}")
    logger.info(f"  CodeBuilderAgent       -> model={cfg_code_builder[0]}  base_url={cfg_code_builder[1]}")
    logger.info(f"  ErrorDiagnosisAgent    -> model={cfg_error_diagnosis[0]}  base_url={cfg_error_diagnosis[1]}")
    logger.info(f"  MechanicalInsightAgent -> model={cfg_mechanical_insight[0]}  base_url={cfg_mechanical_insight[1]}")
    logger.info(f"  ResultEvaluationAgent  -> model={cfg_result_evaluation[0]}  base_url={cfg_result_evaluation[1]}")
    logger.info("="*50 + "\n")

    input_clarifier = InputClarifierAgent(api_key=cfg_input_clarifier[2], model=cfg_input_clarifier[0], base_url=cfg_input_clarifier[1], log_dir=run_dir)
    parsing_agent = ParsingAgent(api_key=cfg_parsing[2], model=cfg_parsing[0], base_url=cfg_parsing[1], log_dir=run_dir)
    code_builder = CodeBuilderAgent(api_key=cfg_code_builder[2], model=cfg_code_builder[0], base_url=cfg_code_builder[1], log_dir=run_dir)
    executor = SimulationExecutorAgent(result_dir=result_dir, log_dir=run_dir)
    error_diagnosis_agent = ErrorDiagnosisAgent(api_key=cfg_error_diagnosis[2], model=cfg_error_diagnosis[0], base_url=cfg_error_diagnosis[1], log_dir=run_dir)
    insight_agent = MechanicalInsightAgent(api_key=cfg_mechanical_insight[2], model=cfg_mechanical_insight[0], base_url=cfg_mechanical_insight[1], log_dir=run_dir)
    result_evaluator = ResultEvaluationAgent(api_key=cfg_result_evaluation[2], model=cfg_result_evaluation[0], base_url=cfg_result_evaluation[1], log_dir=run_dir)

    # ── 解析附件文件 ──────────────────────────────────────────────────────────
    attached_images = []
    file_text = ""
    if args.files:
        logger.info(f"解析附件文件: {args.files}")
        parse_result = parse_files(args.files)
        file_text = parse_result.text
        attached_images = parse_result.images
        logger.info(f"从文件中提取文本: {len(file_text)} 字符, 图片: {len(attached_images)} 张")

    # ── 获取用户 prompt ───────────────────────────────────────────────────────
    if args.prompt:
        user_prompt = args.prompt
    elif not args.files:
        # 没有 --files 时才尝试从文件读取 prompt
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                user_prompt = f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {args.prompt_file}. Please create it or use --prompt.")
            return
    else:
        user_prompt = ""

    # 合并文件提取的文本与用户 prompt
    if file_text:
        if user_prompt:
            user_prompt = f"[从附件文件中提取的内容]\n{file_text}\n\n[用户补充说明]\n{user_prompt}"
        else:
            user_prompt = file_text

    if not user_prompt and not attached_images:
        logger.error("输入不能为空：请提供 --prompt、--prompt_file 或 --files 中的至少一项。")
        return

    logger.info("="*50)
    logger.info(f"User Request: {user_prompt[:500]}{'...' if len(user_prompt) > 500 else ''}")
    if attached_images:
        logger.info(f"Attached images: {attached_images}")
    logger.info("="*50 + "\n")

    # Step 1: Clarify 输入清晰化
    print("\n" + "="*80)
    print(f"{'--- Step 1: Clarification ---':^80}") 
    print("="*80 + "\n")

    logger.info("="*50)
    logger.info("--- Step 1: Clarification ---")
    logger.info("="*50 + "\n")
    clarified_input = await input_clarifier.clarify(user_prompt, images=attached_images if attached_images else None)
    if not clarified_input:
        logger.error("Clarification failed.")
        return

    # Step 2: Parse 结构化解析为JSON
    print("\n" + "="*80)
    print(f"{'--- Step 2: Parsing ---':^80}") 
    print("="*80 + "\n")

    logger.info("="*50)
    logger.info("--- Step 2: Parsing ---")
    logger.info("="*50 + "\n")
    parsed_data = await parsing_agent.parse(clarified_input)
    if not parsed_data:
        logger.error("Parsing failed.")
        return

    # Step 3: Code Building 代码构建
    print("\n" + "="*80)
    print(f"{'--- Step 3: Code Building ---':^80}") 
    print("="*80 + "\n")

    logger.info("="*50)
    logger.info("--- Step 3: Code Building ---")
    logger.info("="*50 + "\n")
    generated_code = await code_builder.build_code(parsed_data)
    if not generated_code:
        logger.error("Code generation failed.")
        return

    # Step 4: Execution Loop（自校正执行循环）
    # 责任链：ResultEvaluationAgent 只做诊断 → ErrorDiagnosisAgent 统一负责修复
    print("\n" + "="*80)
    print(f"{'--- Step 4: Execution Loop (Max Retries: {args.max_retries}) ---':^80}") 
    print("="*80 + "\n")

    logger.info("="*50)
    logger.info(f"--- Step 4: Execution Loop (Max Retries: {args.max_retries}) ---")
    logger.info("="*50 + "\n")
    
    from models import SimulationContext
    context = SimulationContext(current_code=generated_code)

    for attempt in range(args.max_retries):
        logger.info("="*50)
        logger.info(f"Execution Attempt {attempt + 1}/{args.max_retries}")
        logger.info("="*50 + "\n")
        executor.save_code_to_file(context.current_code)
        
        exec_result = executor.execute_simulation()

        # 处理执行失败（runtime error 或 timeout）
        if exec_result.status in ("error", "timeout"):
            error_label = "Timeout" if exec_result.status == "timeout" else "Runtime error"
            logger.warning(f"{error_label} detected during attempt {attempt + 1}.")

            if attempt < args.max_retries - 1:
                logger.info("="*50)
                logger.info("Diagnosing error...")
                logger.info("="*50 + "\n")
                diagnosis = await error_diagnosis_agent.diagnose_and_fix(
                    error_message=exec_result.output,
                    code=context.current_code,
                    simulation_output=exec_result.output,
                    iteration=attempt,
                    repair_history=context.repair_history,
                    log_dir=run_dir,
                )
                context.add_repair_record(diagnosis.hint, diagnosis.confidence)

                if diagnosis.after_code:
                    logger.info("="*50)
                    logger.info("Applying fixed code from ErrorDiagnosisAgent.")
                    logger.info("="*50 + "\n")
                    context.current_code = diagnosis.after_code
                else:
                    logger.error("Error diagnosis failed to produce a valid fix.")
            else:
                logger.error(f"Max retries reached on {error_label.lower()}s.")
        else:
            # 执行成功 → 交给 ResultEvaluationAgent 做物理/逻辑评估
            logger.info("="*50)
            logger.info("Simulation executed successfully! Starting physical/logical evaluation...")
            logger.info("="*50 + "\n")
            eval_result = await result_evaluator.evaluate_results(
                prompt=user_prompt,
                code=context.current_code,
                simulation_output=exec_result.output,
                image_paths=exec_result.files,
            )

            logger.info("="*50)
            logger.info(f"Evaluation feedback: {eval_result.feedback}")
            logger.info("="*50 + "\n")

            # 记录历史最优
            if not context.best_code or (eval_result.is_correct and eval_result.confidence > context.best_confidence) \
                    or (not eval_result.is_correct and context.best_code and eval_result.confidence > context.best_confidence):
                context.update_best(context.current_code, eval_result.confidence, exec_result.output)

            if eval_result.is_correct:
                logger.info("="*50)
                logger.info("Simulation evaluated as logically AND physically correct!")
                logger.info("="*50 + "\n")
                context.simulation_success = True
                context.final_output = exec_result.output
                break
            else:
                # 评估不通过 → 统一由 ErrorDiagnosisAgent 修复
                logger.warning(f"Simulation result is physically incorrect. Attempt {attempt + 1}/{args.max_retries}.")
                if attempt < args.max_retries - 1:
                    logger.info("="*50)
                    logger.info("Forwarding physics feedback to ErrorDiagnosisAgent to rewrite code.")
                    logger.info("="*50 + "\n")
                    diagnosis = await error_diagnosis_agent.diagnose_and_fix(
                        error_message="Physical/Logical Error: " + eval_result.feedback,
                        code=context.current_code,
                        simulation_output=exec_result.output,
                        iteration=attempt,
                        repair_history=context.repair_history,
                        log_dir=run_dir,
                    )
                    context.add_repair_record(diagnosis.hint, diagnosis.confidence)

                    if diagnosis.after_code:
                        logger.info("="*50)
                        logger.info("Applying fixed code from ErrorDiagnosisAgent.")
                        logger.info("="*50 + "\n")
                        context.current_code = diagnosis.after_code
                    else:
                        logger.error("Revising code based on physical feedback failed.")
                else:
                    logger.error("Max retries reached without passing physical evaluation.")

    # Fallback Mechanism (防错后备方案)
    if not context.simulation_success and context.best_code:
        logger.warning("Falling back to the best historical code version.")
        executor.save_code_to_file(context.best_code)
        context.current_code = context.best_code

    # Step 5: Generate Report 生成报告
    if context.simulation_success or context.best_code:
        print("\n" + "="*80)
        print(f"{'--- Step 5: Mechanical Insight Report ---':^80}") 
        print("="*80 + "\n")

        logger.info("="*50)
        logger.info(" --- Step 5: Mechanical Insight Report --- ")
        logger.info("="*50 + "\n")
        report = await insight_agent.generate_report(context.current_code)

        # 保存报告到运行目录
        report_path = os.path.join(run_dir, "simulation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("="*50)
        logger.info("Simulation workflow completed.")
        logger.info("="*50 + "\n")
        print("\n--- Insight Report ---")
        print(report)
        print("\n--- Final Output Metrics ---")
        print(context.final_output)
    else:
        logger.error("Simulation workflow failed completely. No insight report generated.")

if __name__ == "__main__":
    asyncio.run(main())
