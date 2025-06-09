from datasets import load_dataset


def load_daily_dialog_dataset(max_examples=1):
    dataset = load_dataset("daily_dialog", split="test")
    return dataset.select(range(max_examples))
