import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    sleep_time = delay * (2 ** (attempts - 1))
                    logger.warning(f"Retry {attempts}/{max_attempts} in {sleep_time}s")
                    time.sleep(sleep_time)

        return wrapper

    return decorator