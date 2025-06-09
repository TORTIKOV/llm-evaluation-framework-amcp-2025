from data_loaders.loader_xsum import load_xsum_dataset
from models.gigachat.gigachat_client import GigaChatClient
from experiments.summarization.common import run_experiment

if __name__ == "__main__":
    model = GigaChatClient()
    data = load_xsum_dataset()
    run_experiment(model, data, task="gigachat_xsum", output_dir="outputs/summarization")
