from data_loaders.loader_narrativeqa import load_narrativeqa_dataset
from models.yandexgpt.yandexgpt_client import YandexGPTClient
from experiments.qa.common import run_narrativeqa_experiment

if __name__ == "__main__":
    model = YandexGPTClient()
    data = load_narrativeqa_dataset(max_examples=3)
    run_narrativeqa_experiment(model, data, task="yandexgpt_nqa", output_dir="outputs/qa")
