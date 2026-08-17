# Multilingual Probing at Scale: Experiments & Analysis Dashboard

How do multilingual language models encode grammar across a hundred languages — and where do they fail? This repository contains a large-scale probing study of multilingual transformers and the interactive dashboard I built to make its results interpretable.

**The scale of the study:**

- **5 models**: BLOOM (176B), BLOOM-1b7, mBERT, XLM-R, BERT;
- **104 languages** from **18 language families** — from French and Arabic to Sanskrit, Tamil, and Erzya;
- **80 morphosyntactic categories** (Case, Gender, Tense, Aspect, Mood, Voice, Definiteness, ...);
- **1,705 probing experiments** (model × language × category), layer-wise — **85K+ per-layer measurements** in total (`data.csv`).

## How the experiments were run — and why

The full launch pipeline and configuration live in [`experiment/`](experiment/); the design in short:

**Data.** Probing tasks are built from [Universal Dependencies](https://universaldependencies.org/) treebanks: the [AIRI Probing Framework](https://github.com/AIRI-Institute/Probing_framework) converts CONLL-U annotation into SentEval-format classification tasks, one CSV per (language, category) pair — e.g. `fr_pud_Gender` asks a probe to predict `Fem/Masc` from a French sentence representation. UD was chosen because it is the only resource with consistent morphosyntactic annotation across a hundred typologically diverse languages, which makes cross-language comparison meaningful.

**Models.** The four multilingual models form two architecture pairs — BLOOM (176B) / BLOOM-1b7 (decoder-only) and mBERT / XLM-R (encoder-only) — so that differences in results can be attributed to architecture family or model scale rather than to a single model's quirks. Their pretraining data also differ in language coverage (e.g. Slavic languages are present in mBERT/XLM-R but not in BLOOM), which lets us test whether pretraining exposure predicts probing accuracy. Monolingual BERT serves as a reference point.

**Probes.** Each (model, language, category) triple gets its own layer-wise probe: a logistic regression classifier trained on the frozen hidden states of every layer separately (F1 and accuracy, 20 epochs, batch size 64, fixed seed). Logistic regression is the standard choice for probing: a probe should be too simple to *learn* the property itself, so that high accuracy indicates the property is *linearly present* in the representation rather than extracted by a powerful classifier. Layer-wise training shows *where* in the model the information lives.

**Launches.** Runs were organized by language family (task files grouped into per-family folders) and executed per model — mBERT, XLM-R, BERT, and BLOOM-1b7 on a single GPU in Google Colab, BLOOM (176B) via the framework's multi-GPU launcher that shards the model across GPUs. Every run writes a `log.json` with the full config, class ratios, and per-layer scores; `data_launch.py` then aggregates ~1,700 such logs into `data.csv`.

## Why a dashboard?

At this scale, no single figure can answer questions like: *which categories are universally hard? do genealogically related languages behave alike? at which layers does grammatical information live, and does that depend on the model?* So I built an interactive Dash/Plotly application that slices the results by model, language, language family, category, and layer.

It includes 10+ chart types, each chosen for a specific analytical goal:

- **heatmaps** for the big picture — hundreds of (language, category) cells at once, with a gradient scale that stays readable at any size;
- **line graphs** for layer-wise dynamics — overlaying trends across languages or models to spot intersections;
- **boxplots** for the spread of accuracy values behind every average;
- plus horizontal bar charts, scatter plots, treemaps, and combined views.

Most charts have interactive dropdowns; a built-in manual (pop-up instructions) helps first-time users, and all Plotly figures support zoom, autoscale, and PNG export.

Try this: http://alinaavanesyan999.pythonanywhere.com/

## Some findings

Analysis of the results across the four multilingual models (described in detail in my course paper) shows, among other things:

- **Some categories are universally hard.** NameType, InflClass, Clusivity, and Style land in the worst-10 for every model — and not because of any particular language: the language sets behind them differ across models.
- **For Case and Gender, difficulty is a property of the language, not the model.** The richer the case system, the lower the scores (compare Romance vs. Turkic languages) — consistently across all models.
- **Pretraining coverage matters less than expected.** Slavic languages score almost identically in mBERT/XLM-R (which saw them in pretraining) and in BLOOM models (which did not).
- **BLOOM shows layer "dips"** — near-zero accuracy at individual layers where neighboring layers score ~0.5 — a pattern the BERT-family models do not exhibit.

## Repository structure & how to run

- `experiment/` — **the experiment design and launch pipeline**: the run script and a full description of the setup (models, data, probe configuration, hardware notes, reproduction steps);
- `data.csv` — the aggregated results of all 1,705 experiments (85K+ layer-level scores);
- `all_languages.csv`, `genealogy.csv` — language metadata and genealogy (built by `table_of_lang.py`, no need to run it);
- `Probing_framework/results/` — **sample raw logs** (37 mBERT runs) so that anyone can launch the dashboard end-to-end with minimal data; the full set of raw logs is much larger;
- `data_launch.py` — aggregates raw experiment logs into the files behind the dashboard graphs (run it only when you add new probing results);
- `dash_app.py` — the dashboard itself;
- `manual.txt` — the in-app user manual.

To run: put the files above in one folder, run `data_launch.py` once, then `dash_app.py`.
