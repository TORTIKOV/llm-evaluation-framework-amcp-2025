from data_loaders.loader_narrativeqa import load_narrativeqa_dataset
from models.gigachat.gigachat_client import GigaChatClient
from experiments.qa.common import run_narrativeqa_experiment

if __name__ == "__main__":
    model = GigaChatClient()
    data = load_narrativeqa_dataset(max_examples=1)
    run_narrativeqa_experiment(model, data, task="gigachat_nqa", output_dir="outputs/qa")
