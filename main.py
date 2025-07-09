import time
import typer
from typing_extensions import Annotated

app = typer.Typer()

@app.command(help="Prep the dataset")
def prep(
    dataset_output: Annotated[str, typer.Argument(
        help="Where to generate the dataset to",
    )] = 'punctuation_restoration_dataset.jsonl',
    lang: Annotated[str, typer.Option(
        "-l", "--lang",
        help="Language code for Wikipedia",
    )] = 'en',
):
    print(f"[INFO] Writing dataset {lang} to {dataset_output}")
    start = time.time()
    from prep import generate_punctuation_restoration_data as gen_pr_data
    gen_pr_data(lang, dataset_output)
    end = time.time()
    print(f"[INFO] Dataset writing completed after {(end - start) // 60 + 1} minutes")

@app.command(help="Train on a prepped dataset")
def train(
    dataset_path: Annotated[str, typer.Argument(
        help="Path to the prepped wikipedia jsonl dataset",
    )] = 'punctuation_restoration_dataset.jsonl',
    model: Annotated[str, typer.Option(
        "-m", "--model_name_or_path",
        help="Name of model or path to model",
    )] = 't5-base',
    output_path: Annotated[str, typer.Option(
        '-o', '--output_dir',
        help='Output directory of model'
    )] = 'outputs'
):
    print(f"[INFO] Training {model} on dataset at {dataset_path} to {output_path}")
    start = time.time()
    from train import run as train_on_data
    train_on_data(
        data_dir=dataset_path,
        model_name_or_path=model,
        output_dir=output_path
    )
    end = time.time()
    duration = end - start
    hours = duration // 3600
    minutes = (duration - hours * 3600) // 60 + 1
    print(f"[INFO] Model training completed after {hours} hours and {minutes} minutes")

if __name__ == "__main__":
    app()