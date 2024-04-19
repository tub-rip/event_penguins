import os
import logging

logger = logging.getLogger(__name__)


def check_key_and_bool(config: dict, key: str) -> bool:
    """Check the existance of the key and if it's True

    Args:
        config (dict): dict.
        key (str): Key name to be checked.

    Returns:
        bool: Return True only if the key exists in the dict and its value is True.
            Otherwise returns False.
    """
    return key in config.keys() and config[key]


def check_file_exists(filename: str) -> bool:
    """Return True if the file exists.

    Args:
        filename (str): _description_

    Returns:
        bool: _description_
    """
    logger.debug(f"Check {filename}")
    res = os.path.exists(filename)
    if not res:
        logger.warning(f"{filename} does not exist!")
    return res


def uniquify_dir(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f"{filename}-{counter}{extension}"
        counter += 1
    return path
