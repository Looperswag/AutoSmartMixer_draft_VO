# src/utils/logger_setup.py

import logging
import logging.config
import os
import sys

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO", # Default level for console
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG", # Default level for file
            "formatter": "detailed",
            "filename": "app.log", # Default filename, will be overridden by config
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 3,
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "DEBUG", # Capture all messages from DEBUG upwards by default
        "handlers": ["console", "file"],
    },
    # Example of configuring specific loggers:
    # "loggers": {
    #     "src.phase_1_analyzer": {
    #         "level": "DEBUG",
    #         "handlers": ["console", "file"],
    #         "propagate": False # Do not pass messages to the root logger
    #     },
    #     "src.main": {
    #         "level": "INFO",
    #         "handlers": ["console"],
    #         "propagate": True
    #     }
    # }
}

def setup_logging(logging_config_override=None):
    """
    Sets up logging for the application.

    It uses a default configuration which can be partially or fully
    overridden by a dictionary provided in `logging_config_override`.
    This override typically comes from the main `config.yaml`.

    Args:
        logging_config_override (dict, optional): A dictionary with logging
            configuration values to override the defaults.
            Expected structure is similar to Python's logging.config.dictConfig.
    """
    config = DEFAULT_LOGGING_CONFIG.copy() # Start with defaults

    if logging_config_override and isinstance(logging_config_override, dict):
        # --- Smartly merge override with defaults ---
        # Override top-level keys like 'version', 'disable_existing_loggers'
        for key in ["version", "disable_existing_loggers"]:
            if key in logging_config_override:
                config[key] = logging_config_override[key]

        # Override/extend formatters
        if "formatters" in logging_config_override:
            if "formatters" not in config: config["formatters"] = {}
            for fmt_name, fmt_conf in logging_config_override["formatters"].items():
                config["formatters"][fmt_name] = fmt_conf

        # Override/extend handlers
        if "handlers" in logging_config_override:
            if "handlers" not in config: config["handlers"] = {}
            for h_name, h_conf in logging_config_override["handlers"].items():
                # If handler exists in default, update it, else add new
                if h_name in config["handlers"]:
                    config["handlers"][h_name].update(h_conf)
                else:
                    config["handlers"][h_name] = h_conf

        # Override root logger settings
        if "root" in logging_config_override:
            if "root" not in config: config["root"] = {}
            config["root"].update(logging_config_override["root"])

        # Override/extend specific loggers
        if "loggers" in logging_config_override:
            if "loggers" not in config: config["loggers"] = {}
            for logger_name, logger_conf in logging_config_override["loggers"].items():
                config["loggers"][logger_name] = logger_conf

        # --- Handle specific config values from a simpler config.yaml structure ---
        # Example: if config.yaml has:
        # logging:
        #   level: "INFO"  # Applies to root console handler
        #   log_file: "logs/my_app.log" # Applies to file handler filename
        #   log_to_console: True
        #   log_to_file: True

        log_level_general = logging_config_override.get("level", "INFO").upper()
        log_to_console = logging_config_override.get("log_to_console", True)
        log_to_file = logging_config_override.get("log_to_file", True)
        log_file_path = logging_config_override.get("log_file", "app.log") # Default if not in config

        # Update console handler based on simpler config
        if "console" in config["handlers"]:
            config["handlers"]["console"]["level"] = log_level_general
            if not log_to_console and "console" in config["root"]["handlers"]:
                config["root"]["handlers"].remove("console")
                # Also remove from specific loggers if they use it and no other handler is left
                if "loggers" in config:
                    for logger_name in config["loggers"]:
                        if "console" in config["loggers"][logger_name].get("handlers", []):
                            config["loggers"][logger_name]["handlers"].remove("console")


        # Update file handler based on simpler config
        if "file" in config["handlers"]:
            config["handlers"]["file"]["level"] = logging_config_override.get("file_level", log_level_general) # Allow specific file level
            config["handlers"]["file"]["filename"] = log_file_path
            if not log_to_file and "file" in config["root"]["handlers"]:
                config["root"]["handlers"].remove("file")
                if "loggers" in config:
                    for logger_name in config["loggers"]:
                        if "file" in config["loggers"][logger_name].get("handlers", []):
                            config["loggers"][logger_name]["handlers"].remove("file")


        # Ensure log directory exists for the file handler
        if log_to_file and "file" in config["handlers"]:
            config["handlers"]["file"]["filename"] = log_file_path # 确保文件名被设置
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except OSError as e:
                    # Fallback: log to console about the error and disable file logging
                    print(f"警告: 无法创建日志目录 {log_dir}: {e}。禁用文件日志记录。", file=sys.stderr)
                    if "file" in config["root"]["handlers"]:
                        config["root"]["handlers"].remove("file")
                    if "file" in config["handlers"]:
                        del config["handlers"]["file"] # Remove the handler definition

    # If root handlers list becomes empty, add console back as a basic fallback
    if "root" in config and not config["root"].get("handlers"):
        config["root"]["handlers"] = ["console"] # Ensure console is default if all removed
        if "console" not in config["handlers"]: # If console was also removed, re-add basic
             config["handlers"]["console"] = DEFAULT_LOGGING_CONFIG["handlers"]["console"]


    try:
        logging.config.dictConfig(config)
        logger = logging.getLogger(__name__)
        #logger.info("Logging configured successfully.")
        if "file" in config["handlers"] and config["handlers"]["file"].get("filename"):
            print(f"日志将记录到文件: {os.path.abspath(config['handlers']['file']['filename'])}")

    except Exception as e:
        # Fallback to basicConfig if dictConfig fails
        print(f"使用 dictConfig 配置日志时出错: {e}。回退到 basicConfig。", file=sys.stderr)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
        logger = logging.getLogger(__name__)
        logger.warning("由于字典配置错误，已使用 basicConfig 进行日志记录。")

if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Default setup
    print("--- Testing Default Logging ---")
    setup_logging()
    logging.debug("This is a default debug message.")
    logging.info("This is a default info message.")
    logging.warning("This is a default warning message.")
    logging.getLogger("another_module").info("Info from another_module.")
    print(f"Default log file should be at: {os.path.abspath(DEFAULT_LOGGING_CONFIG['handlers']['file']['filename'])}")


    # 2. Setup with overrides from a simple config structure (like in our config.yaml)
    print("\n--- Testing Logging with Simple Config Override ---")
    simple_override_config = {
        "level": "DEBUG",  # For console
        "log_to_file": True,
        "log_file": "logs/app_simple_override.log",
        "log_to_console": True
    }
    setup_logging(simple_override_config)
    logging.debug("This is a simple override debug message (should appear on console and file).")
    logging.info("This is a simple override info message.")
    logging.getLogger("moduleA").debug("Debug from moduleA")


    # 3. Setup with more detailed dictConfig style override
    print("\n--- Testing Logging with Detailed dictConfig Override ---")
    detailed_override_config = {
        "formatters": {
            "brief": {"format": "%(levelname)s: %(message)s"}
        },
        "handlers": {
            "console": { # Override existing console handler
                "level": "WARNING",
                "formatter": "brief"
            },
            "file": { # Override existing file handler
                "filename": "logs/app_detailed_override.log",
                "level": "INFO",
                "formatter": "detailed" # Uses default detailed formatter
            }
        },
        "root": {
            "level": "INFO", # Root logger level
            "handlers": ["console", "file"]
        },
        "loggers": {
            "src.main": { # Specific logger for src.main
                "level": "DEBUG", # This won't show on console due to console handler level
                "handlers": ["file"], # Only log to file for src.main
                "propagate": False
            }
        }
    }
    setup_logging(detailed_override_config)
    logging.debug("Root debug: This debug message should NOT appear (root level INFO, console WARNING).")
    logging.info("Root info: This info message should appear in file, NOT on console (console WARNING).")
    logging.warning("Root warning: This warning message SHOULD appear on console (brief) and file (detailed).")

    main_logger = logging.getLogger("src.main")
    main_logger.debug("src.main debug: This should go to app_detailed_override.log only.")
    main_logger.info("src.main info: This should also go to app_detailed_override.log only.")

    print("\nCheck 'app.log', 'logs/app_simple_override.log', and 'logs/app_detailed_override.log' for output.")