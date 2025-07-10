from time import ctime as now
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
    print(f"[INFO] Writing dataset {lang} to {dataset_output} at {now()}")
    from prep import generate_punctuation_restoration_data as gen_pr_data
    gen_pr_data(lang, dataset_output)
    print(f"[INFO] Dataset writing completed at {now()}")

@app.command(help="Train on a prepped dataset")
def train(
    data_dir: Annotated[str, typer.Argument(
        help="Path to the prepped wikipedia jsonl dataset",
    )] = 'punctuation_restoration_dataset.jsonl',
    max_epochs: Annotated[int, typer.Option(
        "-e", '--max-epoch',
        help="Max number of epochs"
    )] = 3,
    model_name_or_path: Annotated[str, typer.Option(
        "-m", "--model_name_or_path",
        help="Name of model or path to model",
    )] = 't5-base',
    output_path: Annotated[str, typer.Option(
        '-o', '--output_dir',
        help='Output directory of model'
    )] = 'outputs',
    seed: Annotated[int, typer.Option(
        '-s', '--seed',
        help='Seed'
    )] = 42,
):
    print(f"[INFO] Training {model_name_or_path} on dataset at {data_dir} to {output_path} at {now()}")
    from train import run as train_on_data
    train_on_data(
        data_dir=data_dir,
        max_epochs=max_epochs,
        model_name_or_path=model_name_or_path,
        output_dir=output_path,
        seed=seed,
    )
    print(f"[INFO] Model training completed at {now()}")

if __name__ == "__main__":
    app()