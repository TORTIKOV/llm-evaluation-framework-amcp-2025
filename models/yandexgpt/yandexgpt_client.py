import time
from models.yandexgpt.yandexgpt_api import YandexGPTAPI
from models.base_model import BaseModel


class YandexGPTClient(BaseModel):
    def __init__(self):
        self.api = YandexGPTAPI()

    def generate(self, document: str, prompt: str):
        start_time = time.time()
        result = self.api.send_chat_completion(document, system_message=prompt)
        elapsed = time.time() - start_time
        return result, elapsed
