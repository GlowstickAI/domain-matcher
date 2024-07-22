import os

import typer

from domain_matcher.experiments import experiment_script

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    typer.run(experiment_script)
