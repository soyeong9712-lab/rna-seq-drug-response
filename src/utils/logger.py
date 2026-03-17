import logging

def get_logger(name="ngs-moa"):
    logger = logging.getLogger(name)

    # ✅ 이미 handler가 있으면 그대로 반환 (중복 방지)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
