from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from evaluation.sci import semantic_compression_index


def evaluate_all_metrics(prediction: str, reference: str, lang: str = "en", w: float = 0.7):
    results = {}

    # ROUGE
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=(lang == "en"))
        rouge = scorer.score(reference, prediction)
        results['ROUGE-1'] = round(rouge["rouge1"].fmeasure, 4)
        results['ROUGE-L'] = round(rouge["rougeL"].fmeasure, 4)
    except Exception as e:
        print("❌ ROUGE error:", e)
        results['ROUGE-1'] = results['ROUGE-L'] = None

    # METEOR
    try:
        results['METEOR'] = round(meteor_score([reference.split()], prediction.split()), 4)
    except Exception as e:
        print("❌ METEOR error:", e)
        results['METEOR'] = None

    # BERTScore
    try:
        P, R, F1 = bert_score([prediction], [reference], lang=lang, verbose=False)
        results['BERTScore'] = round(F1.item(), 4)
    except Exception as e:
        print("❌ BERTScore error:", e)
        results['BERTScore'] = None

    # SCI
    try:
        sci_details = semantic_compression_index(
            model_text=prediction,
            reference_text=reference,
            verbose=False,
            return_details=True,
            w=w
        )
        results['SCI'] = sci_details["SCI"]
        results['SCI_Overlap'] = sci_details["Overlap"]
        results['SCI_IntentScore'] = sci_details["IntentScore"]
        results['SCI_Penalty'] = sci_details["CompressionPenalty"]
        results['SCI_Compression'] = round(
            sci_details["GenLength"] / sci_details["RefLength"], 3
        ) if sci_details["RefLength"] > 0 else None
        results['SCI_Delta'] = abs(sci_details["GenLength"] - sci_details["RefLength"])  # ← НОВОЕ
        results['SCI_Facts_Reference'] = "; ".join(sci_details["ReferenceFacts"])
        results['SCI_Facts_Model'] = "; ".join(sci_details["ModelFacts"])
    except Exception as e:
        print("❌ SCI error:", e)
        results['SCI'] = results['SCI_Overlap'] = results['SCI_IntentScore'] = results['SCI_Compression'] = None
        results['SCI_Penalty'] = results['SCI_Facts_Reference'] = results['SCI_Facts_Model'] = None

    return results
