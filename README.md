# Protein BioWord Embeddings based on Maximal Repeats

## Overview

This project builds protein vector representations from Maximal Repeat (MR) sequences. It takes a pre-computed MR dataset, assembles a BioWord corpus in BigQuery, trains a FastText model on it, and then produces per-protein embedding vectors that can be reduced, visualized, and evaluated.

The pipeline is split into independent stages: each one reads from the previous stage's output, so you can re-run any step in isolation without going back to the start.

> **Running long stages:** most pipeline commands (corpus prep, training, vector computation, dimensionality reduction) take minutes to hours. Run them with `nohup` so they survive terminal disconnection, and redirect stdout to a log file:
> ```bash
> nohup python -m <module> <args> > output.log &
> ```
> Progress and errors are also written to the stage-specific log file under the `logs/` folder in the project root.

```
MR dataset  →  [Stage 1]  Corpus prep (BigQuery)
            →  [Stage 1b] Control corpus (optional — sidestep of Stage 1)
            →  [Stage 2]  FastText training
            →  [Stage 3]  Protein vector computation
            →  [Stage 3b] Cross-dataset embedding (optional)
            →  [Stage 3c] Combine real + control vectors (optional)
            →  [Stage 4]  Dimensionality reduction (openTSNE / UMAP / densMAP)
            →  [Stage 5]  Metrics & evaluation
            →  [Stage 6]  Interactive visualization (Dash explorer)
```

**Tech stack:**
- Python · Pandas · BigQuery + Cloud Storage · Gensim FastText
- scikit-learn PCA · openTSNE · UMAP / densMAP
- Dash + Plotly (interactive explorer)

---

## Input dataset

This project works with the output of the Maximal Repeat calculator:
[mgiordano/prot-mr-generator](https://github.com/mgiordano/prot-mr-generator)

After running that project on a protein dataset, organize the output as:

```
processed_datasets/
├── testGroupDataset/
│   ├── testGroupDataset_1_999999_1_PATTERNS.csv     # all computed MRs
│   ├── testGroupDataset_1_999999_1_POSITIONS.csv    # MR ↔ protein relations
│   ├── testGroupDataset_sequence_dataset.csv        # proteins + attributes
│   └── control_testGroupDataset_sequence_dataset.csv  # control proteins (optional)
└── familyDataset/
    └── ...
```

`INPUT_DATA_ROOT_PATH` must point to the root `processed_datasets/` directory.

---

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Google Cloud setup**

BigQuery is used both to run the heavy corpus-prep joins (at scale, local DataFrames and even DuckDB with 128 GB RAM can't handle 3B MR positions) and as a convenient environment for ad-hoc data exploration on massive protein/MR tables.

- Create a GCP project with BigQuery and Cloud Storage APIs enabled. If a project already exists, you can reuse it — just make sure the service account below has the required roles and that the bucket is in the same project.
- Create a Service Account with roles `BigQuery Admin` and `Storage Admin`.
- Download its JSON key and place it anywhere accessible (e.g. project root).
- Create a Storage bucket that will be used for BigQuery ↔ local data transfers.

  Optionally set an Object Lifecycle rule on the bucket to auto-delete files older than 1 day under the dataset prefixes (e.g. `testGroupDataset/20`, `familyDataset/20`) to avoid accumulating transient exports.

**3. Create a `.env` file** in the project root (see `.env.test` for a template):

```bash
INPUT_DATA_ROOT_PATH="/full/path/to/processed_datasets/"
GCLOUD_SERVICE_ACCOUNT_KEY="/full/path/to/gcloud_saccount_key.json"
GCLOUD_INPUT_BUCKET="your-gcs-bucket-name"
GCLOUD_PROJECT_ID="your-gcp-project-id"
```

---

## Parameter reference

Most scripts share the same four positional parameters that together form a **Run ID**:

| Parameter | What it is | Valid values (from `utils/utils.py`) |
|---|---|---|
| `TIMESTAMP` | When the corpus prep run was started | `YYYYMMDD_HH_MM_SS` (e.g. `20250101_11_14_21`) |
| `DATASET` | Protein dataset name | `TEST_GROUP`, `NANO_GROUP`, `FAMILY`, `BSC`, `FAMILY_200`, `BSC_ANK` |
| `MR_FILTER` | MR filtering rule | `MR_FILTER_NONE`, `MR_FILTER_DROP_NE`, `MR_FILTER_KEEP_NE`, `MR_FILTER_KEEP_SMR`, `MR_FILTER_KEEP_4_10`, etc. |
| `PARTITION_RULE` | How BioWords are partitioned | `PARTITION_RULE_USE_ALL` |

The **Run ID** (`TIMESTAMP-DATASET-MR_FILTER-PARTITION_RULE`) is the key that links outputs across stages. All intermediate and final outputs are tagged with it.

---

## Pipeline stages

### Stage 1 — Corpus preparation

Loads the input MR dataset into BigQuery, applies the chosen MR filter, and computes the BioWord partition for each protein.

> **Why BigQuery?** At scale (700K proteins, 3B MR positions) local DataFrames and even DuckDB with 128 GB RAM can't handle the joins. BigQuery runs the same computation in a few minutes.

**What happens:**
1. Input CSVs are uploaded to your GCS bucket, then loaded into BigQuery (`protein_input` dataset).
2. Intermediate results land in the `stage_results` dataset under tables prefixed with the Run ID and stage name (`s1_filtered_mrs`, `s2_joined_mrs`, `s3_corpus`).

```bash
python -m corpus_prep.corpus_prep_pipeline DATASET MR_FILTER PARTITION_RULE [--dry-run] \
    [--metadata-table TABLE] [--metadata-join-key CORPUS_COL:METADATA_COL]
```

```bash
# Example
nohup python -m corpus_prep.corpus_prep_pipeline TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL > output.log &

# Dry run: stage tables get a `99_tmp-` prefix and expire after 1 day
python -m corpus_prep.corpus_prep_pipeline TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL --dry-run

# Custom metadata (see below)
nohup python -m corpus_prep.corpus_prep_pipeline BSC MR_FILTER_KEEP_NE PARTITION_RULE_USE_ALL \
    --metadata-table bscDataset_metadata --metadata-join-key sequence_name:seq_id > output.log &
```

**Custom metadata table (overrides default metadata structure)**

By default, the corpus query joins against the `family_types` table in `protein_input` to attach `sequence_family_name` and `sequence_family_type` to each protein. This works for datasets that follow the standard family-type schema.

For datasets with a different metadata structure (e.g. BSC, which has no canonical `family_types` table), use `--metadata-table` and `--metadata-join-key` together. When both flags are set, the default `family_types` join is replaced entirely by a left join against the specified metadata table. All columns from that table (except the join key column itself) are carried into the corpus output. The `sequence_family_name`-based BigQuery clustering is also skipped, since that column is no longer guaranteed to exist.

Both flags must be provided together:

| Flag | Description |
|---|---|
| `--metadata-table TABLE` | Name of the metadata table in the `protein_input` BQ dataset (e.g. `bscDataset_metadata`) |
| `--metadata-join-key CORPUS_COL:METADATA_COL` | Join key as `corpus_column:metadata_column` (e.g. `sequence_name:seq_id`) |

Explore results with `corpus_prep/corpus_prep_exploration.ipynb`.

---

### Stage 1b — Control corpus (optional)

Control data is a sidestep of the corpus prep stage, not part of model training. Its purpose is to evaluate how well the trained model generalizes: control proteins are embedded and projected alongside the real dataset so that their distribution can be compared against the expected clustering.

**Key constraint:** control proteins must not influence the model in any way — they should not be used to compute Maximal Repeats and must not appear in the training corpus. This means the control corpus cannot go through BigQuery like the main pipeline. Instead, this script downloads the MR vocabulary already computed for the real run and runs the Aho-Corasick algorithm locally to find MR pattern matches in the control protein sequences, producing a BioWord corpus for them using the same vocabulary.

Requires a `control_<dataset>_sequence_dataset.csv` file inside the corresponding dataset input folder (see Input dataset section).

```bash
python -m corpus_prep.corpus_prep_control TIMESTAMP DATASET MR_FILTER PARTITION_RULE
```

```bash
# Example
python -m corpus_prep.corpus_prep_control 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL
```

Output: a `20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-s3_corpus_control_for_eval.csv` file placed alongside the regular corpus shards in the dataset date folder.

---

### Stage 2 — Model training

Trains a FastText model on the corpus from Stage 1. Downloads the corpus from BigQuery, assembles shards into a single joined file, then trains.

```bash
python -m corpus_prep.corpus_train TIMESTAMP DATASET MR_FILTER PARTITION_RULE \
    [--vector-size N] [--max-cpu K]
```

```bash
# Example
nohup python -m corpus_prep.corpus_train 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL > output.log &

# With custom embedding size and CPU limit
nohup python -m corpus_prep.corpus_train 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    --vector-size 200 --max-cpu 8 > output.log &
```

| Flag | Default | Description |
|---|---|---|
| `--vector-size N` | `100` | Dimensionality of the trained embeddings |
| `--max-cpu K` | all cores | Cores to use during parallel training |

**Output** is written to `INPUT_DATA_ROOT_PATH/DATASET/YYYYMMDD/models/`:

```
processed_datasets/
└── testGroupDataset/
    └── 20250101/
        └── models/
            ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model
            ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model.syn1.npy
            ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model.wv.vectors_ngrams.npy
            └── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model.wv.vectors_vocab.npy
```

> **Resources:** training on 700K proteins / 103M MR vocabulary requires ~150 GB RAM and ~5 hours on a 48-vCPU machine.

---

### Stage 3 — Protein vector computation

Computes a mean embedding vector for each protein by summing its BioWord embeddings and dividing by the number of words. Outputs `.tsv` files (vectors + metadata) for downstream stages.

```bash
python -m model_eval.model_embeddings TIMESTAMP DATASET MR_FILTER PARTITION_RULE \
    [--control] [--mmap] [--metadata]
```

```bash
# Standard run
nohup python -m model_eval.model_embeddings 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL > output.log &

# Control dataset
nohup python -m model_eval.model_embeddings 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL --control > output.log &

# Memory-constrained machine: use mmap to reduce RAM (slower load)
nohup python -m model_eval.model_embeddings 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL --mmap > output.log &
```

| Flag | Description |
|---|---|
| `--control` | Build vectors for the control dataset using the control corpus |
| `--mmap` | Memory-map the model (lower RAM, 4× slower load) |
| `--metadata` | Only produce the `metadata.tsv`, skip vector computation |

**Output** under `INPUT_DATA_ROOT_PATH/DATASET/YYYYMMDD/RUNID/vector_output/`:
```
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-metadata.tsv     # protein labels
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_bio.tsv  # protein vectors
```

> **Resources:** iterating 700K proteins takes ~5 hours. Large model load: ~30 min and ~120 GB RAM (mmap saves ~30% RAM but increases load time 4×).

---

### Stage 3b — Cross-dataset embedding (optional)

Embeds proteins from one dataset using a model trained on a different dataset. Useful for comparing embedding spaces or evaluating generalization.

Takes two Run IDs: one for the **input sequences** (proteins to embed) and one for the **model** (trained FastText + MR patterns).

```bash
python -m model_eval.model_embeddings_cross \
    INPUT_TIMESTAMP INPUT_DATASET INPUT_FILTER INPUT_PARTITION_RULE \
    MODEL_TIMESTAMP MODEL_DATASET MODEL_FILTER MODEL_PARTITION_RULE \
    MODEL_TAG \
    [--mmap] [--metadata]
```

`MODEL_TAG` is a short label (e.g. `family`, `bscAnk8`) appended to output filenames so results from different models can coexist in the same folder.

```bash
# Example: embed BSC proteins using a model trained on familyDataset
nohup python -m model_eval.model_embeddings_cross \
    20250101_11_14_21 BSC MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    20250101_09_00_00 FAMILY MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    family > output.log &
```

Output is placed under the **input dataset** run folder, tagged with a compact model label (`cross_<MODEL_TAG>`):
```
20250101_11_14_21-bscDataset-filter_none-partition_use_all-cross_family-vectors_bio.tsv
20250101_11_14_21-bscDataset-filter_none-partition_use_all-cross_family-metadata.tsv
```

Pass `--cross <MODEL_TAG>` to Stage 4 to use these vectors in dimensionality reduction.

---

### Stage 3c — Combine real + control vectors (optional)

If you want to run dimensionality reduction on real and control proteins together (required for t-SNE, which needs all points to compute the joint projection), run this before Stage 4:

```bash
python -m model_eval.model_combine_datasets TIMESTAMP DATASET MR_FILTER PARTITION_RULE
```

Produces `20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-combined-metadata.tsv` and `20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-combined-vectors_bio.tsv` in the `vector_output` folder. Pass `--control` to Stage 4 and beyond to use the combined files.

---

### Stage 4 — Dimensionality reduction

Reduces protein vectors (typically 100D) to 2D or 3D for visualization. Supports **openTSNE** and **UMAP / densMAP** via a JSON experiment configuration.

**Experiment config** — create `experiment.json` under `vector_output/experiments/`. Arrays drive parameter sweeps: the longest array determines how many runs execute; single-element arrays (or scalars) are reused across all runs.

**openTSNE config example:**
```json
{
    "reduction_method": "tsne",
    "tsne_implementation": "openTSNE",
    "pca_n_components": 50,
    "n_components": [2],
    "random_state": [0, 1000, 537],
    "perplexity": [10, 30, 50],
    "learning_rate": ["auto"],
    "max_iter": [5000],
    "n_jobs": [8],
    "verbose": 1
}
```

**UMAP / densMAP config example:**
```json
{
    "reduction_method": "umap",
    "pca_n_components": 50,
    "n_components": [2],
    "n_neighbors": [15, 30],
    "min_dist": [0.1],
    "random_state": [42],
    "densmap": [false, true],
    "dens_lambda": [2.0]
}
```

> `densmap: true` enables densMAP, which additionally preserves local density information on top of UMAP's topology.

```bash
python -m model_eval.model_dim_reduction TIMESTAMP DATASET MR_FILTER PARTITION_RULE EXPERIMENT_FILE.json \
    [--control] [--cross MODEL_TAG] [--filter-col COLUMN VALUE]
```

The experiment file must be passed **with the `.json` extension** — the script strips it to derive the output folder name.

```bash
# openTSNE sweep (3 perplexity values → 3 output files)
nohup python -m model_eval.model_dim_reduction 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment.json > output.log &

# With combined control data
nohup python -m model_eval.model_dim_reduction 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment.json --control > output.log &

# Using cross-embedded vectors (from Stage 3b)
nohup python -m model_eval.model_dim_reduction 20250101_11_14_21 BSC MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment.json --cross family > output.log &

# Drop proteins where metadata column 'partition_type' == 'mr' before reducing
nohup python -m model_eval.model_dim_reduction 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment.json \
    --filter-col partition_type mr > output.log &
```

| Flag | Description |
|---|---|
| `--control` | Use combined real + control vectors |
| `--cross MODEL_TAG` | Use cross-embedded vectors tagged with `MODEL_TAG` |
| `--filter-col COLUMN VALUE` | Drop rows where `COLUMN == VALUE` before reducing |

**Output file naming** under `vector_output/experiments/my_experiment/`:
```
# PCA baseline (always produced)
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_pca-50.tsv

# t-SNE output pattern: vectors_tsne-[n_components]_[method]_[perplexity]_[learning_rate]_[max_iter]_[random_state]
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_tsne-2_auto_10_auto_5000_0.tsv
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_tsne-2_auto_30_auto_5000_1000.tsv
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_tsne-2_auto_50_auto_5000_537.tsv

# UMAP output pattern: vectors_umap-[n_components]_n[n_neighbors]_d[min_dist]_[random_state]
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_umap-2_n15_d0.1_42.tsv

# densMAP output (same pattern, prefixed with densmap)
20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_densmap-2_n30_d0.1_42.tsv
```

---

### Stage 5 — Metrics & evaluation

Computes quantitative metrics on dimensionality reduction results: nearest-neighbor preservation, silhouette scores, Spearman rank correlations, and more.

```bash
python -m model_eval.model_metrics TIMESTAMP DATASET MR_FILTER PARTITION_RULE EXPERIMENT_NAME CLASS_LABEL_COLUMN \
    [--control] [--max-workers N]
```

```bash
# Evaluate experiment using family_name as class label
python -m model_eval.model_metrics 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    my_experiment family_name

# With parallelization
python -m model_eval.model_metrics 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    my_experiment family_name --max-workers 8
```

| Argument | Description |
|---|---|
| `CLASS_LABEL_COLUMN` | Metadata column to use as ground-truth class (e.g. `family_name`, `family_type`) |
| `--control` | Use combined dataset |
| `--max-workers N` | Parallel workers for metric computation |

---

### Stage 6 — Interactive visualization (Dash explorer)

A browser-based explorer that handles 1M+ points via WebGL. Load any dimensionality reduction output and explore it interactively.

**Features:** distinct-color / highlight / density-heatmap rendering modes · searchable dropdowns for high-cardinality metadata columns · CATH hierarchical drill-down tab · hover tooltips · PNG export

```bash
python -m model_viz.explorer_app
# Open http://localhost:8050
```

The app reads reduction outputs from the `vector_output` folder; the Run ID and experiment are selected via UI dropdowns.

---

## Legacy visualization tools

The following tools predate the Dash explorer and are kept for reference. They are not actively maintained.

**Static charts (`model_viz/model_viz_enhanced.py`)** — renders Matplotlib/Seaborn scatter plots for each experiment run, colored by `family_name` or `family_type`. Useful for producing publication-quality static figures.

```bash
python -m model_viz.model_viz_enhanced TIMESTAMP DATASET MR_FILTER PARTITION_RULE EXPERIMENT_NAME \
    [--control] [--mode single|grid] [--chart-type name|type|overlap]
```

**TensorBoard Projector** — interactive 3D projector with zoom, filter, and nearest-neighbor search. Browser memory limits it to ~10K points.

```bash
python -m corpus_prep.tensorboard_setup tsne TIMESTAMP DATASET MR_FILTER PARTITION_RULE [--control]
# prints the tensorboard serve command to run
```

**Jupyter notebooks** — for ad-hoc exploration:
- `corpus_prep/corpus_prep_exploration.ipynb` — Stage 1 BigQuery outputs
- `model_eval/model_eval_exploration.ipynb` — embeddings and reduction results
- `model_viz/interactive_viz_explorer.py` — ipywidgets-based notebook explorer

---

## End-to-end example (test dataset)

The `TEST_GROUP` dataset is small (~483 proteins) and fast to process — good for verifying your setup works before running on large datasets.

```bash
# 0. Activate environment
source venv/bin/activate

# 1. Corpus prep — loads MR data into BigQuery and computes the BioWord corpus
nohup python -m corpus_prep.corpus_prep_pipeline TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL > output.log &
# Note the printed Run ID, e.g. 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all

# 2. Train FastText model on the corpus
nohup python -m corpus_prep.corpus_train 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL > output.log &

# 3. Compute protein embedding vectors
nohup python -m model_eval.model_embeddings 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL > output.log &

# 4. Dimensionality reduction — place my_experiment.json in vector_output/experiments/ first
nohup python -m model_eval.model_dim_reduction 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment.json > output.log &

# 5. Visualize
python -m model_viz.model_viz_enhanced 20250101_11_14_21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment
# Or launch the interactive explorer:
python -m model_viz.explorer_app
```

---

## Output folder structure

```
processed_datasets/
└── testGroupDataset/
    ├── testGroupDataset_1_999999_1_PATTERNS.csv      # Stage 1 input
    ├── testGroupDataset_1_999999_1_POSITIONS.csv     # Stage 1 input
    ├── testGroupDataset_sequence_dataset.csv         # Stage 1 input
    ├── control_testGroupDataset_sequence_dataset.csv # Stage 1b input (optional)
    └── 20250101/
        ├── models/
        │   ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model
        │   ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model.syn1.npy
        │   ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model.wv.vectors_ngrams.npy
        │   └── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model.wv.vectors_vocab.npy
        ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-s3_corpus_for_train_0.gz
        ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-s3_corpus_for_train_joined.csv
        ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-s3_corpus_control_for_eval.csv
        └── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all/
            └── vector_output/
                ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-metadata.tsv
                ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_bio.tsv
                └── experiments/
                    ├── my_experiment.json
                    └── my_experiment/
                        ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_pca-50.tsv
                        ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_tsne-2_auto_30_auto_5000_0.tsv
                        ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-vectors_umap-2_n15_d0.1_42.tsv
                        └── charts/
                            └── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-family_name.png
```

---

## Scale & resource guide

| Dataset | Proteins | Unique MRs | MR positions | RAM (train) | Train time |
|---|---|---|---|---|---|
| TEST_GROUP | ~483 | ~110K | ~1M | < 8 GB | < 5 min |
| FAMILY | ~700K | ~103M | ~3B | ~150 GB | ~5 h (48 vCPU) |

- **Stage 1 (BigQuery):** scales to any size; cost is determined by GCP query bytes processed.
- **Stage 3 (vector compute):** iterating 700K proteins takes ~5 h; no parallelization currently.
- **Stage 4 (openTSNE):** generally faster than sklearn's implementation; automatically selects the best approximation method based on dataset size.
- **Stage 4 (UMAP / densMAP):** faster than t-SNE for large datasets; densMAP additionally preserves local density information.
