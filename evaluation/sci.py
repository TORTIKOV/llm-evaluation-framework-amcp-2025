import math
import re
from sentence_transformers import SentenceTransformer, util
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
# Загружаем SBERT
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('ё', 'е').replace('Ё', 'Е').strip())


def lemmatize_word(word):
    return morph.parse(word)[0].normal_form


def extract_key_facts(text):
    text = normalize_text(text)
    raw_facts = re.findall(r'\b[А-ЯA-ZЁ][а-яa-zё]+\b|\b\d+\b', text)
    lemmatized_facts = {lemmatize_word(word.lower()) for word in raw_facts}
    return lemmatized_facts


def calculate_overlap(reference_facts, model_facts):
    if not reference_facts or not model_facts:
        return 0.0
    intersection = reference_facts.intersection(model_facts)
    precision = len(intersection) / len(model_facts)
    recall = len(intersection) / len(reference_facts)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_intent_score(model_text, reference_text):
    model_text = normalize_text(model_text)
    reference_text = normalize_text(reference_text)
    emb1 = sbert_model.encode(model_text, convert_to_tensor=True)
    emb2 = sbert_model.encode(reference_text, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def calculate_lengths(model_text, reference_text):
    model_len = len(normalize_text(model_text).split())
    reference_len = len(normalize_text(reference_text).split())
    return model_len, reference_len


def calculate_compression_penalty(model_len, reference_len, k=1.0, delta=5.0):
    if reference_len == 0 and model_len == 0:
        return 1.0  # оба пусты — идеальное совпадение
    if model_len == 0:
        return 0.0  # пустой ответ при ненулевом референсе — критическая ошибка
    if reference_len == 0:
        return 0.0  # если референс пустой, а модель сгенерировала — тоже ошибка

    compression = model_len / reference_len
    relative_diff = abs(compression - 1)
    abs_diff = abs(model_len - reference_len)
    gamma = abs_diff / (abs_diff + delta) if abs_diff > 0 else 0
    penalty = math.exp(-k * relative_diff * gamma)
    return penalty


def semantic_compression_index(
        model_text: str,
        reference_text: str,
        verbose: bool = False,
        return_details: bool = False,
        w: float = 0.7,
        k: float = 1.0,
        delta: float = None
):
    assert 0 <= w <= 1, "w must be in [0, 1]"
    overlap_weight = w
    intent_weight = 1 - w

    reference_facts = extract_key_facts(reference_text)
    model_facts = extract_key_facts(model_text)

    overlap = calculate_overlap(reference_facts, model_facts)
    intent_score = calculate_intent_score(model_text, reference_text)
    model_len, reference_len = calculate_lengths(model_text, reference_text)

    # Вычисляем delta внутри, если не передана явно
    if delta is None:
        delta = 0.5 * reference_len  # половина длины эталона

    compression_penalty = calculate_compression_penalty(model_len, reference_len, k=k, delta=delta)
    sci = (overlap_weight * overlap + intent_weight * intent_score) * compression_penalty
    sci = round(min(sci, 1.0), 3)

    if verbose:
        print("🧠 Ключевые факты (эталон):", reference_facts)
        print("🧠 Ключевые факты (ответ):", model_facts)
        print(f"✅ Overlap: {round(overlap, 3)}")
        print(f"🎯 IntentScore: {round(intent_score, 3)}")
        print(f"📏 Lengths: ref={reference_len}, gen={model_len}")
        print(f"⚖️ CompressionPenalty: {round(compression_penalty, 3)}")
        print(f"⚖️ Веса: overlap×{overlap_weight}, intent×{intent_weight}")
        print(f"💡 SCI = {sci}")

    if return_details:
        return {
            "SCI": sci,
            "Overlap": round(overlap, 3),
            "IntentScore": round(intent_score, 3),
            "CompressionPenalty": round(compression_penalty, 3),
            "ReferenceFacts": list(reference_facts),
            "ModelFacts": list(model_facts),
            "RefLength": reference_len,
            "GenLength": model_len
        }

    return sci