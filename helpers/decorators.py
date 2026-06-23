import functools
import logging
import time

logger = logging.getLogger(__name__)

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator để tự động thử lại (retry) một hàm khi xảy ra ngoại lệ (Exception).
    
    Args:
        max_attempts (int): Số lần thử tối đa trước khi ném ra ngoại lệ.
        delay (float): Thời gian chờ ban đầu giữa các lần thử (giây).
        backoff (float): Hệ số nhân thời gian chờ sau mỗi lần thất bại.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    # Thử thực thi hàm gốc
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.warning(
                        f"[Retry] Attempt {attempts}/{max_attempts} for '{func.__module__}.{func.__name__}' "
                        f"failed with error: {e}. Retrying in {current_delay}s..."
                    )
                    # Nếu đạt đến số lần thử tối đa, ghi log lỗi và ném lại ngoại lệ
                    if attempts >= max_attempts:
                        logger.error(
                            f"[Retry] Function '{func.__module__}.{func.__name__}' failed after {max_attempts} attempts."
                        )
                        raise e
                    # Chờ trước khi thử lại
                    time.sleep(current_delay)
                    # Tăng thời gian chờ theo hệ số backoff
                    current_delay *= backoff
        return wrapper
    return decorator

