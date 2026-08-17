# Experiment design

This folder documents how the probing experiments behind the dashboard were run.

## Pipeline

```
Universal Dependencies (CONLL-U treebanks, 104 languages)
        │  AIRI Probing Framework: UD → SentEval-format tasks
        ▼
task CSVs, one per (language, category) pair, grouped by language family
        │  run_probing.py — one launch per model
        ▼
layer-wise probes: logistic regression on frozen activations of every layer
        │  one log.json per experiment (config + per-layer f1/accuracy)
        ▼
../data_launch.py — aggregation into data.csv (85K+ layer-level scores)
        │
        ▼
../dash_app.py — interactive analysis dashboard
```

## Setup

| | |
|---|---|
| Models | BLOOM (176B), BLOOM-1b7, mBERT (cased), XLM-R (base), BERT (base, uncased) |
| Data | Universal Dependencies treebanks, converted to SentEval-format classification tasks |
| Coverage | 104 languages, 18 language families, 80 morphosyntactic categories — 1,705 experiments |
| Probe | logistic regression over frozen hidden states, trained separately for every layer |
| Metrics | F1 and accuracy per layer, on train/val/test splits |
| Training | 20 epochs, batch size 64, fixed seed (42) |

Each (model, language, category) triple gets its own probe. The classifier reads
the frozen hidden states of one layer and predicts the value of the
morphosyntactic category (e.g. `Gender: Fem/Masc` for `fr_pud_Gender`); class
ratios and label encodings are recorded in each run's `log.json`.

## Hardware notes

- mBERT, XLM-R, BERT, and BLOOM-1b7 were probed on a single GPU (runs executed
  in Google Colab, task files grouped by language family).
- BLOOM (176B) does not fit on one GPU; it was run with the framework's
  multi-GPU launcher (`big_bloom_multiGPU.py`), which shards the model across
  GPUs, with the same task files and probe settings.

## Reproducing

1. Install the [AIRI Probing Framework](https://github.com/AIRI-Institute/Probing_framework) and its requirements.
2. Prepare task CSVs from UD treebanks with the framework's converter.
3. Run `run_probing.py` for each model (see the usage examples in the script).
4. Point `../data_launch.py` at the produced logs, then start `../dash_app.py`.

Sample raw logs (37 mBERT runs) are already included in
`../Probing_framework/results/`, so the dashboard can be launched end-to-end
without re-running anything.
