CENSOR_PATTERNS = [
    "Посмотрите",
    "в интернете",
    "не могу",
    "найдите сами",
    "По вашему запросу ничего не найдено",
    "Моя задача — обеспечить безопасное общение"
]


def is_censored(text: str) -> bool:
    return any(phrase.lower() in text.lower() for phrase in CENSOR_PATTERNS)