import requests
from config import config


class YandexGPTAPI:
    def __init__(self):
        self.api_key = config.API_KEY
        self.folder_id = config.FOLDER_ID
        self.url = config.URL

    def send_chat_completion(self, document, system_message, temperature=0.7, max_tokens=100):
        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens
            },
            "messages": [
                {"role": "system", "text": system_message},
                {"role": "user", "text": document}
            ]
        }
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        response = requests.post(self.url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["result"]["alternatives"][0]["message"]["text"].strip()
