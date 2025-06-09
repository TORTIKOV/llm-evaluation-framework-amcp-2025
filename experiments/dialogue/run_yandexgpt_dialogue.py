from data_loaders.loader_dialogue import load_daily_dialog_dataset
from models.yandexgpt.yandexgpt_client import YandexGPTClient
from experiments.dialogue.common import run_dialogue_experiment

if __name__ == "__main__":
    model = YandexGPTClient()
    data = load_daily_dialog_dataset(max_examples=3)
    run_dialogue_experiment(model, data, task="yandexgpt_dd", output_dir="outputs/dialogue")
