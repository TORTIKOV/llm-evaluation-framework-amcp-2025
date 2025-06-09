import os
import base64
import requests
import uuid
import time
from config import config

# Конфигурация и авторизация
client_id = config.SBER_ID
secret = config.SBER_SECRET
auth = config.SBER_AUTH


def get_token(auth_token, scope='GIGACHAT_API_PERS'):
    rq_uid = str(uuid.uuid4())
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': rq_uid,
        'Authorization': f'Basic {auth_token}'
    }
    payload = {'scope': scope}
    try:
        response = requests.post(url, headers=headers, data=payload, verify=False)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.RequestException as e:
        print(f"Ошибка при получении токена: {str(e)}")
        return None


def get_chat_completion(auth_token, user_message, system_message=None, temperature=0.7, max_tokens=100, top_p=0.95):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    # Формируем список сообщений
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "gigachat",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "stream": False,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.2,
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"Ошибка при запросе к чату: {str(e)}")
        return None


def get_long_chat_completion(auth_token, conversation, temperature=0.7, max_tokens=150, top_p=0.95):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    messages = [{"role": msg["role"], "content": msg["text"]} for msg in conversation]

    payload = {
        "model": "gigachat",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "stream": False,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.2,
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }

    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"Ошибка при обработке длинного разговора: {str(e)}")
        return None
