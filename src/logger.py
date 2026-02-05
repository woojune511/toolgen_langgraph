import logging
import os
from datetime import datetime, timedelta, timezone
from src.config import LOG_DIR

def get_logger(name: str):
    """
    콘솔 출력과 파일 저장을 동시에 수행하는 로거 생성
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
        )

        # 1. 콘솔 핸들러
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 2. 파일 핸들러 (날짜별 로그 파일)
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(
            os.path.join(LOG_DIR, f"agent_run_{today}.log"), encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger