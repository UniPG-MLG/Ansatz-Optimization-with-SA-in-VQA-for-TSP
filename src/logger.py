import logging
from src.args import args
from src.uuid import RUN_UUID
import os

_LOG_DIR = "logs"
_LOGGING_FILES = [
    f"{_LOG_DIR}/run-INFO_{RUN_UUID}.log",
    f"{_LOG_DIR}/run-DEBUG_{RUN_UUID}.log",
]


def setup_logger():
    # Create logger with the module's name
    logger = logging.getLogger("src")  # Common name for the entire module
    # NOTE: Must be in DEBUG mode in order to let other logger working
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # NOTE: this code will be only executed once !!!
    # Check if handler already exists to prevent adding multiple handlers
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)

        # WARNING: logging.FileHandler(mode="w") not working and leaves the file empty.
        #          This is a sort of "fix" to empty the file every run start in order to
        #          simulate a sort of mode=W.
        for l in _LOGGING_FILES:
            if os.path.exists(l):
                os.remove(l)
            formatter = logging.Formatter(
                "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
            )

        # FileHandlers
        for l in _LOGGING_FILES:
            file_handler = logging.FileHandler(
                filename=l,
                # mode="a",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG if "DEBUG" in l else logging.INFO)

            logger.addHandler(file_handler)

        # StreamHandler - STDOUT
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
        logger.addHandler(stdout_handler)

    return logger


def get_logger_files():
    return _LOGGING_FILES
