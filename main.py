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
from utils import setup_logger, create_run_dir


async def main():
    parser = argparse.ArgumentParser(description="MCP-SIM Asynchronous Orchestrator")
    parser.add_argument("--prompt", type=str, help="Natural language description of the simulation.", default=None)
    parser.add_argument("--prompt_file", type=str, help="Path to a text file containing the simulation prompt.", default="prompt.txt")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for code execution.")
    args = parser.parse_args()

    # 创建本次运行的独立输出目录
    run_dir = create_run_dir()
    result_dir = os.path.join(run_dir, "result")

    logger = setup_logger('main_execution', 'main_execution.log', log_dir=run_dir)
    logger.info(f"Run directory: {run_dir}")

    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "gpt-4o")
    base_url = os.environ.get("LLM_BASE_URL", None)
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not found. Please set it in a .env file or environment.")
        return

    logger.info(f"Initializing MCP-SIM Agents with model: {model_name}...")
    input_clarifier = InputClarifierAgent(api_key=api_key, model=model_name, base_url=base_url, log_dir=run_dir)
    parsing_agent = ParsingAgent(api_key=api_key, model=model_name, base_url=base_url, log_dir=run_dir)
    code_builder = CodeBuilderAgent(api_key=api_key, model=model_name, base_url=base_url, log_dir=run_dir)
    executor = SimulationExecutorAgent(result_dir=result_dir, log_dir=run_dir)
    error_diagnosis_agent = ErrorDiagnosisAgent(api_key=api_key, model=model_name, base_url=base_url, log_dir=run_dir)
    insight_agent = MechanicalInsightAgent(api_key=api_key, model=model_name, base_url=base_url, log_dir=run_dir)
    result_evaluator = ResultEvaluationAgent(api_key=api_key, model=model_name, base_url=base_url, log_dir=run_dir)

    if args.prompt:
        user_prompt = args.prompt
    else:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                user_prompt = f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {args.prompt_file}. Please create it or use --prompt.")
            return
            
    if not user_prompt:
        logger.error("Simulation prompt cannot be empty.")
        return

    logger.info(f"User Request: {user_prompt}")

    # Step 1: Clarify 输入澄清 
    logger.info("--- Step 1: Clarification ---")
    clarified_input = await input_clarifier.clarify(user_prompt)
    if not clarified_input:
        logger.error("Clarification failed.")
        return

    # Step 2: Parse 结构化解析为JSON
    logger.info("--- Step 2: Parsing ---")
    parsed_data = await parsing_agent.parse(clarified_input)
    if not parsed_data:
        logger.error("Parsing failed.")
        return

    # Step 3: Code Building 代码构建
    logger.info("--- Step 3: Code Building ---")
    generated_code = await code_builder.build_code(parsed_data)
    if not generated_code:
        logger.error("Code generation failed.")
        return

    # Step 4: Execution Loop（自校正执行循环）
    # 责任链：ResultEvaluationAgent 只做诊断 → ErrorDiagnosisAgent 统一负责修复
    logger.info(f"--- Step 4: Execution Loop (Max Retries: {args.max_retries}) ---")
    simulation_success = False
    current_code = generated_code
    best_code = None
    best_confidence = -1.0
    final_output = ""
    repair_history: list[dict] = []  # 修复历史，避免重蹈覆辙

    for attempt in range(args.max_retries):
        logger.info(f"Execution Attempt {attempt + 1}/{args.max_retries}")
        executor.save_code_to_file(current_code)
        
        exec_result = executor.execute_simulation()

        # 处理执行失败（runtime error 或 timeout）
        if exec_result.status in ("error", "timeout"):
            error_label = "Timeout" if exec_result.status == "timeout" else "Runtime error"
            logger.warning(f"{error_label} detected during attempt {attempt + 1}.")

            if attempt < args.max_retries - 1:
                logger.info("Diagnosing error...")
                diagnosis = await error_diagnosis_agent.diagnose_and_fix(
                    error_message=exec_result.output,
                    code=current_code,
                    simulation_output=exec_result.output,
                    iteration=attempt,
                    repair_history=repair_history,
                    log_dir=run_dir,
                )
                repair_history.append({"hint": diagnosis.hint, "confidence": diagnosis.confidence})

                if diagnosis.after_code:
                    logger.info("Applying fixed code from ErrorDiagnosisAgent.")
                    current_code = diagnosis.after_code
                else:
                    logger.error("Error diagnosis failed to produce a valid fix.")
            else:
                logger.error(f"Max retries reached on {error_label.lower()}s.")
        else:
            # 执行成功 → 交给 ResultEvaluationAgent 做物理/逻辑评估
            logger.info("Simulation executed successfully! Starting physical/logical evaluation...")
            eval_result = await result_evaluator.evaluate_results(
                prompt=user_prompt,
                code=current_code,
                simulation_output=exec_result.output,
                image_paths=exec_result.files,
            )

            logger.info(f"Evaluation feedback: {eval_result.feedback}")

            # 记录历史最优
            if not best_code or (eval_result.is_correct and eval_result.confidence > best_confidence) \
                    or (not eval_result.is_correct and best_code and eval_result.confidence > best_confidence):
                best_code = current_code
                best_confidence = eval_result.confidence
                final_output = exec_result.output

            if eval_result.is_correct:
                logger.info("Simulation evaluated as logically AND physically correct!")
                simulation_success = True
                final_output = exec_result.output
                break
            else:
                # 评估不通过 → 统一由 ErrorDiagnosisAgent 修复（责任链清晰化）
                logger.warning(f"Simulation result is physically incorrect. Attempt {attempt + 1}/{args.max_retries}.")
                if attempt < args.max_retries - 1:
                    logger.info("Forwarding physics feedback to ErrorDiagnosisAgent to rewrite code.")
                    diagnosis = await error_diagnosis_agent.diagnose_and_fix(
                        error_message="Physical/Logical Error: " + eval_result.feedback,
                        code=current_code,
                        simulation_output=exec_result.output,
                        iteration=attempt,
                        repair_history=repair_history,
                        log_dir=run_dir,
                    )
                    repair_history.append({"hint": diagnosis.hint, "confidence": diagnosis.confidence})

                    if diagnosis.after_code:
                        logger.info("Applying fixed code from ErrorDiagnosisAgent.")
                        current_code = diagnosis.after_code
                    else:
                        logger.error("Revising code based on physical feedback failed.")
                else:
                    logger.error("Max retries reached without passing physical evaluation.")

    # Fallback Mechanism (防错后备方案)
    if not simulation_success and best_code:
        logger.warning("Falling back to the best historical code version.")
        executor.save_code_to_file(best_code)
        current_code = best_code

    # Step 5: Generate Report 生成报告
    if simulation_success or best_code:
        logger.info("--- Step 5: Mechanical Insight Report ---")
        report = await insight_agent.generate_report(current_code)

        # 保存报告到运行目录
        report_path = os.path.join(run_dir, "simulation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("Simulation workflow completed.")
        print("\n--- Insight Report ---")
        print(report)
        print("\n--- Final Output Metrics ---")
        print(final_output)
    else:
        logger.error("Simulation workflow failed completely. No insight report generated.")

if __name__ == "__main__":
    asyncio.run(main())
