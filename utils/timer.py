import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("⏱️ Превышен лимит времени ожидания ответа")
