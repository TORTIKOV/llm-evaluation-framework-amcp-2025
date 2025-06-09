import time
from utils.translator import translate
from utils.censorship_filter import is_censored
from evaluation.evaluate import evaluate_all_metrics
from utils.save_results import save_to_excel

SYSTEM_PROMPT_RU = '''Ты — собеседник в диалоге. Прочитай историю и продолжи её одной репликой:
1. Ответ должен быть логичным продолжением предыдущих реплик.
2. Сохрани ключевые факты и контекст.
3. Не пиши более 20 слов.
Ответ должен быть на русском языке.'''

SYSTEM_PROMPT_EN = '''You are a dialogue agent. Read the previous turns and respond with one short message:
1. Your reply should logically follow the conversation.
2. Preserve key facts and context.
3. Use no more than 20 words.
The answer must be in English.'''


def run_dialogue_experiment(model, dataset, task, output_dir):
    results_to_save = []

    for idx, example in enumerate(dataset, start=1):
        dialog = example["dialog"]
        if len(dialog) < 3:
            continue

        print(f"🟡 Обработка диалога {idx}")
        doc_en = " ".join(dialog[:-1])
        ref_en = dialog[-1]

        doc_ru = translate(doc_en, "en", "ru")
        ref_ru = translate(ref_en, "en", "ru")

        if not doc_ru or not ref_ru:
            print("⚠️ Пропущен из-за неудачного перевода.")
            continue

        try:
            ans_ru_doc_ru, t1 = model.generate(doc_ru + "\n\nОтвет:", SYSTEM_PROMPT_RU)
            ans_ru_doc_en, t2 = model.generate(doc_en + "\n\nОтвет:", SYSTEM_PROMPT_RU)
            ans_en_doc_ru, t3 = model.generate(doc_ru + "\n\nAnswer:", SYSTEM_PROMPT_EN)
            ans_en_doc_en, t4 = model.generate(doc_en + "\n\nAnswer:", SYSTEM_PROMPT_EN)
        except Exception as e:
            print(f"❗ Ошибка генерации: {e}")
            continue

        if any(is_censored(ans) for ans in [ans_ru_doc_ru, ans_ru_doc_en, ans_en_doc_ru, ans_en_doc_en]):
            print("⚠️ Пропущен из-за цензуры.")
            continue

        ans_ru_doc_ru_en = translate(ans_ru_doc_ru, "ru", "en")
        ans_ru_doc_en_en = translate(ans_ru_doc_en, "ru", "en")
        ans_en_doc_ru_ru = translate(ans_en_doc_ru, "en", "ru")
        ans_en_doc_en_ru = translate(ans_en_doc_en, "en", "ru")

        result = {
            "Context (EN)": doc_en,
            "Reference Reply (EN)": ref_en,
            "Reference Reply (RU)": ref_ru,

            "Answer RU→RU": ans_ru_doc_ru,
            "Answer EN→RU": ans_ru_doc_en,
            "Answer RU→EN": ans_en_doc_ru,
            "Answer EN→EN": ans_en_doc_en,

            "RU→RU→EN": ans_ru_doc_ru_en,
            "EN→RU→EN": ans_ru_doc_en_en,
            "RU→EN→RU": ans_en_doc_ru_ru,
            "EN→EN→RU": ans_en_doc_en_ru,

            "Time RU→RU": round(t1, 2),
            "Time EN→RU": round(t2, 2),
            "Time RU→EN": round(t3, 2),
            "Time EN→EN": round(t4, 2),
        }

        for key, answer, ref, lang in [
            ("RU→RU", ans_ru_doc_ru, ref_ru, "ru"),
            ("EN→RU", ans_ru_doc_en, ref_ru, "ru"),
            ("RU→EN", ans_en_doc_ru, ref_en, "en"),
            ("EN→EN", ans_en_doc_en, ref_en, "en"),
        ]:
            metrics = evaluate_all_metrics(prediction=answer, reference=ref, lang=lang, w=0.7)

            result.update({
                f"{key} ROUGE-1": metrics.get("ROUGE-1"),
                f"{key} ROUGE-L": metrics.get("ROUGE-L"),
                f"{key} METEOR": metrics.get("METEOR"),
                f"{key} BERTScore": metrics.get("BERTScore"),
                f"{key} SCI": metrics.get("SCI"),
                f"{key} SCI Overlap": metrics.get("SCI_Overlap"),
                f"{key} SCI IntentScore": metrics.get("SCI_IntentScore"),
                f"{key} SCI Penalty": metrics.get("SCI_Penalty"),
                f"{key} SCI Compression": metrics.get("SCI_Compression"),
                f"{key} SCI Δ": metrics.get("SCI_Delta"),
            })

            for w_val in [0.25, 0.5, 0.75, 0.9]:
                try:
                    sci_w = evaluate_all_metrics(prediction=answer, reference=ref, lang=lang, w=w_val)["SCI"]
                    result[f"{key} SCI (w={w_val})"] = sci_w
                except Exception as e:
                    print(f"⚠️ SCI @ w={w_val}: {e}")
                    result[f"{key} SCI (w={w_val})"] = None

        results_to_save.append(result)
        time.sleep(1.5)

    save_to_excel(results_to_save, f"{output_dir}/results_{task}.xlsx")
