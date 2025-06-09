from datasets import load_dataset


def load_xsum_dataset(max_examples=1):
    dataset = load_dataset("xsum", split="test")
    return dataset.select(range(max_examples))
