"""Launch script for the probing experiments behind this repository.

The probes are trained with the AIRI Probing Framework
(https://github.com/AIRI-Institute/Probing_framework), adapted from its
example scripts. The framework converts Universal Dependencies CONLL-U
files into SentEval-format classification tasks (one CSV per
(language, morphosyntactic category) pair); this script runs a layer-wise
probing experiment for every task file of the chosen model.

Task CSVs are grouped into folders by language family (e.g. `data/romance/`,
`data/slavic/`), so a run can cover one family or all of them.

Every run produces a log.json with the full config and per-layer
f1/accuracy scores — these logs are the raw input of ../data_launch.py,
which aggregates them for the dashboard.

Usage:
    python run_probing.py --model_name bert-base-multilingual-cased --tasks_folder data/
    python run_probing.py --model_name xlm-roberta-base --tasks_folder data/romance/

Note: BLOOM (176B) does not fit on a single GPU; for it we used the
framework's multi-GPU launcher (scripts/big_bloom_multiGPU.py) with the
same task files and probe settings.
"""

import gc
import glob
import random
import traceback
import uuid
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from tqdm import tqdm

from probing.pipeline import ProbingPipeline


def init_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(
    model_name: str = "bert-base-multilingual-cased",
    tasks_folder: str = "data/",
    encoding_batch_size: int = 64,
    classifier_batch_size: int = 64,
    train_epochs: int = 20,
    device: Optional[str] = None,
):
    experiment = ProbingPipeline(
        hf_model_name=model_name,
        device=device,
        metric_names=["f1", "accuracy"],
        encoding_batch_size=encoding_batch_size,
        classifier_batch_size=classifier_batch_size,
    )
    init_seed()
    unique_id = uuid.uuid4().hex

    task_files = sorted(glob.glob(f"{tasks_folder}/**/*.csv", recursive=True))
    for f in tqdm(task_files, desc=f"Probing {model_name}"):
        # clear caches between tasks so long runs survive on a single GPU
        experiment.transformer_model.Caching.clear()
        torch.cuda.empty_cache()
        gc.collect()

        try:
            experiment.run(
                probe_task=Path(f).stem,
                path_to_task_file=f,
                verbose=True,
                train_epochs=train_epochs,
            )
        except Exception:
            with open(f"logErrors_{unique_id}.txt", "a") as log_errors:
                log_errors.write(f + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
