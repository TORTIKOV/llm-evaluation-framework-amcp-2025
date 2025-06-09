from data_loaders.loader_xsum import load_xsum_dataset
from models.yandexgpt.yandexgpt_client import YandexGPTClient
from experiments.summarization.common import run_experiment

if __name__ == "__main__":
    model = YandexGPTClient()
    data = load_xsum_dataset()
    run_experiment(model, data, task="yandexgpt_xsum", output_dir="outputs/summarization")
