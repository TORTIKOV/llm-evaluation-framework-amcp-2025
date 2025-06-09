from datasets import load_dataset


def load_narrativeqa_dataset(max_examples=50, max_words=400):
    dataset = load_dataset("narrativeqa", split="validation")
    filtered = []
    for example in dataset:
        if "summary" not in example["document"] or not example["document"]["summary"]:
            continue
        context = example["document"]["summary"]["text"]
        if len(context.split()) > max_words:
            continue
        if not example["question"]["text"] or not example["answers"]:
            continue
        filtered.append(example)
        if len(filtered) >= max_examples:
            break
    return filtered
