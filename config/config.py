import os
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = str(os.getenv("FOLDER_ID"))
API_KEY = str(os.getenv("API_KEY"))
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
SBER_AUTH = str(os.getenv("SBER_AUTH"))
SBER_ID = str(os.getenv("SBER_ID"))
SBER_SECRET = str(os.getenv("SBER_SECRET"))
