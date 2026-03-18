import logging
import os
import datetime


def setup_logger(name, log_file, level=logging.INFO, log_dir=None):
    """创建并配置 logger。

    Args:
        name: logger 名称
        log_file: 日志文件名
        level: 日志级别
        log_dir: 日志目录（可选）。若指定则日志写入该目录下。
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_file)

    # File handler
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)

    return logger


def create_run_dir(base_dir="runs") -> str:
    """创建带时间戳的运行输出目录。

    Returns:
        新建目录的绝对路径，例如 runs/run_20260317_160000/
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    # 在 run_dir 下创建 result 子目录
    os.makedirs(os.path.join(run_dir, "result"), exist_ok=True)
    return run_dir
