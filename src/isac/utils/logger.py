"""
统一日志管理模块

提供统一的日志配置和管理功能，确保整个项目的日志格式和级别一致。
支持环境变量配置、日志轮转、线程安全等特性。
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


from .. import PROJECT_ROOT

from logging.handlers import RotatingFileHandler

# ============================================================================
# 常量定义
# ============================================================================

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 简洁日志格式（用于控制台输出）
CONSOLE_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# 默认日志级别
DEFAULT_LOG_LEVEL = logging.INFO

# 日志文件配置
LOG_DIR = PROJECT_ROOT / "log"
LOG_FILE = LOG_DIR / "isac.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# 初始化标志（防止重复初始化）
_logging_initialized = False


def _ensure_logger_debug2_compat() -> None:
    """兼容 Kaleido/Choreographer 可能调用的 Logger.debug2。"""
    if not hasattr(logging.Logger, "debug2"):
        def _debug2(self, msg, *args, **kwargs):
            return self.debug(msg, *args, **kwargs)

        logging.Logger.debug2 = _debug2


def _suppress_noisy_third_party_loggers() -> None:
    """抑制常见第三方库的冗余 INFO 日志。"""
    noisy_loggers = [
        "kaleido",
        "kaleido.kaleido",
        "kaleido._kaleido_tab",
        "choreographer",
        "choreographer.browser_async",
        "choreographer.browsers.chromium",
        "choreographer.utils._tmpfile",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def _resolve_log_level(level: Optional[int]) -> int:
    """解析最终日志级别。"""
    if level is not None:
        return level

    env_level = os.environ.get("ISAC_LOG_LEVEL", "").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(env_level, DEFAULT_LOG_LEVEL)


def _create_console_handler(level: int, console_format_string: Optional[str]) -> logging.Handler:
    """创建控制台 handler（stdout）。"""
    console_formatter = logging.Formatter(
        console_format_string or CONSOLE_LOG_FORMAT, datefmt=DATE_FORMAT
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    return console_handler


# 文件 handler 创建失败时返回 None（由 setup_logging 负责输出 warning）
def _try_create_file_handler(
    log_file_path: Path,
    level: int,
    format_string: Optional[str],
) -> tuple[Optional[logging.Handler], Optional[BaseException]]:
    try:
        # 确保日志目录存在
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_formatter = logging.Formatter(
            format_string or LOG_FORMAT, datefmt=DATE_FORMAT
        )
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        return file_handler, None
    except (OSError, PermissionError) as e:
        return None, e


# ============================================================================
# 日志配置函数
# ============================================================================
def setup_logging(
    level: Optional[int] = None,
    log_to_file: bool = True,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    console_format_string: Optional[str] = None,
    force: bool = False,
) -> None:
    """配置日志系统

    参数:
    -------
    - level : int, optional
        日志级别，默认为 INFO。如果为 None，则从环境变量 ISAC_LOG_LEVEL 读取
    - log_to_file : bool, optional
        是否将日志写入文件，默认为 True
    - log_file : Path, optional
        日志文件路径，默认为 logs/isac.log
    - format_string : str, optional
        文件日志格式字符串，默认使用标准格式
    - console_format_string : str, optional
        控制台日志格式字符串，默认使用简洁格式
    - force : bool, optional
        是否强制重新配置，即使已经初始化过，默认为 False
    """
    global _logging_initialized

    # 如果已经初始化且不强制重新配置，则直接返回
    if _logging_initialized and not force:
        return

    level = _resolve_log_level(level)
    _ensure_logger_debug2_compat()

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有的处理器
    root_logger.handlers.clear()

    # 控制台处理器（使用简洁格式）
    console_handler = _create_console_handler(level, console_format_string)
    root_logger.addHandler(console_handler)

    # 文件处理器（如果启用）
    if log_to_file:
        log_file_path = log_file or LOG_FILE
        file_handler, err = _try_create_file_handler(
            log_file_path=log_file_path,
            level=level,
            format_string=format_string,
        )
        if file_handler is not None:
            root_logger.addHandler(file_handler)
        else:
            # 如果无法创建日志文件，只输出警告，不影响程序运行
            # 保持原行为：改回标准 LOG_FORMAT（原代码就在异常里这样做）
            console_handler.setFormatter(
                logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            )
            root_logger.warning(
                f"无法创建日志文件 {log_file_path}: {err}，将只输出到控制台"
            )

    # 标记为已初始化
    _logging_initialized = True

    # 抑制第三方冗余日志，避免污染终端与日志文件
    _suppress_noisy_third_party_loggers()


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """获取日志记录器

    参数:
    -------
    - name : str
        日志记录器名称，通常使用 __name__
    - level : int, optional
        日志级别，如果不指定则使用根日志记录器的级别

    返回值:
    -------
    - logging.Logger
        配置好的日志记录器
    """
    logger = logging.getLogger(name)

    # 如果指定了级别，则设置该记录器的级别
    if level is not None:
        logger.setLevel(level)

    # 如果根日志记录器还没有配置，则进行默认配置
    if not _logging_initialized and not logging.getLogger().handlers:
        setup_logging()

    return logger


def reset_logging() -> None:
    """重置日志系统

    清除所有日志配置，恢复到未初始化状态。
    主要用于测试或需要重新配置日志系统的场景。
    """
    global _logging_initialized

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)  # 重置为默认级别
    _logging_initialized = False


# 只在模块首次导入时执行一次
if not _logging_initialized and not logging.getLogger().handlers:
    setup_logging()
