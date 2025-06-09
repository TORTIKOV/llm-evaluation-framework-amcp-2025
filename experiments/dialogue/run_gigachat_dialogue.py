from data_loaders.loader_dialogue import load_daily_dialog_dataset
from models.gigachat.gigachat_client import GigaChatClient
from experiments.dialogue.common import run_dialogue_experiment

if __name__ == "__main__":
    model = GigaChatClient()
    data = load_daily_dialog_dataset(max_examples=1)
    run_dialogue_experiment(model, data, task="gigachat_dd", output_dir="outputs/dialogue")
