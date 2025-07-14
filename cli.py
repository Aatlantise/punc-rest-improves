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
    print(f"[INFO] Prepping dataset started {now()}")
    print(f"[INFO] Language code is {lang}")
    print(f"[INFO] Dataset output is {dataset_output}")
    from prep_pr_data import generate_punctuation_restoration_data as gen_pr_data
    gen_pr_data()
    print(f"[INFO] Prepping dataset finished {now()}")

# @app.command(help="Train on a prepped dataset")
# def train(
#     data_dir: Annotated[str, typer.Argument(
#         help="Path to the prepped wikipedia jsonl dataset",
#     )] = 'punctuation_restoration_dataset.jsonl',
#     max_epochs: Annotated[int, typer.Option(
#         "-e", '--max-epoch',
#         help="Max number of epochs"
#     )] = 3,
#     model_name_or_path: Annotated[str, typer.Option(
#         "-m", "--model_name_or_path",
#         help="Name of model or path to model",
#     )] = 't5-base',
#     output_path: Annotated[str, typer.Option(
#         '-o', '--output_dir',
#         help='Output directory of model'
#     )] = 'outputs',
#     seed: Annotated[int, typer.Option(
#         '-s', '--seed',
#         help='Seed'
#     )] = 42,
# ):
#     print(f"[INFO] Model training started {now()}")
#     print(f"[INFO] Dataset input is {data_dir}")
#     print(f"[INFO] Seed is {seed}")
#     print(f"[INFO] Model is {model_name_or_path}")
#     print(f"[INFO] Output is {output_path}")
#     from main import run as train_on_data
#     train_on_data(
#         data_dir=data_dir,
#         max_epochs=max_epochs,
#         model_name_or_path=model_name_or_path,
#         output_dir=output_path,
#         seed=seed,
#     )
#     print(f"[INFO] Model training finished {now()}")

if __name__ == "__main__":
    app()