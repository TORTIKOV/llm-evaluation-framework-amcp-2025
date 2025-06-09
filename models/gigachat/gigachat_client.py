import signal
import time
from models.gigachat.gigachat_api import get_token, get_chat_completion
from utils.timer import timeout_handler
from config.config import SBER_AUTH
from models.base_model import BaseModel


class GigaChatClient(BaseModel):
    def __init__(self):
        self.auth_token = get_token(SBER_AUTH)

    def generate(self, document: str, prompt: str):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        start_time = time.time()
        try:
            result = get_chat_completion(self.auth_token, document, prompt)
            elapsed = time.time() - start_time
            signal.alarm(0)
            return result, elapsed  # ← возвращаем результат и время
        except Exception as e:
            signal.alarm(0)
            raise e
