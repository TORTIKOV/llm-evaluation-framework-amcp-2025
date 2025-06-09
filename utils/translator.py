from deep_translator import GoogleTranslator
import time


def translate(text, source_lang, target_lang, retries=3):
    for attempt in range(retries):
        try:
            return GoogleTranslator(source=source_lang, target=target_lang, timeout=15).translate(text)
        except Exception as e:
            print(f"⚠️ Перевод (попытка {attempt + 1}) {source_lang}→{target_lang} не удался: {e}")
            time.sleep(1.5)
    print(f"❌ Перевод не выполнен после {retries} попыток — возвращаю пустую строку")
    return ""
