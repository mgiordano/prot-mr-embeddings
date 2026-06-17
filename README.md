# Protein BioWord Embeddings based on Maximal Repeats

## Overview

This project builds protein vector representations from Maximal Repeat (MR) sequences. It takes a pre-computed MR dataset, assembles a BioWord corpus in BigQuery, trains a FastText model on it, and then produces per-protein embedding vectors that can be reduced, visualized, and evaluated.

The pipeline is split into independent stages: each one reads from the previous stage's output, so you can re-run any step in isolation without going back to the start.

```
MR dataset  →  [Stage 1] Corpus prep (BigQuery)
            →  [Stage 2] FastText training
            →  [Stage 2b] Control corpus (optional)
            →  [Stage 3] Protein vector computation
            →  [Stage 3b] Cross-dataset embedding (optional)
            →  [Stage 3c] Combine real + control vectors (optional)
            →  [Stage 4] Dimensionality reduction (t-SNE / UMAP / densMAP)
            →  [Stage 5] Metrics & evaluation
            →  [Stage 6] Visualization (charts / interactive explorer)
```

**Tech stack:**
- Python · Pandas · BigQuery + Cloud Storage · Gensim FastText
- scikit-learn PCA · t-SNE (sklearn / openTSNE) · UMAP / densMAP
- Dash + Plotly (interactive explorer) · Seaborn + Matplotlib (static charts)
- Tensorboard (projector)

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

- Create a GCP project with BigQuery and Cloud Storage APIs enabled.
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
| `TIMESTAMP` | When the corpus prep run was started | `YYYY_MM_DD_Hh_Mm_ss` |
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
python -m corpus_prep.corpus_prep_pipeline DATASET MR_FILTER PARTITION_RULE [--dry-run]
```

```bash
# Example
python -m corpus_prep.corpus_prep_pipeline TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL

# Dry run: stage tables get a `99_tmp-` prefix and expire after 1 day
python -m corpus_prep.corpus_prep_pipeline TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL --dry-run
```

Explore results with `corpus_prep/corpus_prep_exploration.ipynb`.

---

### Stage 2 — Model training

Trains a FastText model on the corpus from Stage 1. Downloads the corpus from BigQuery, assembles shards into a single joined file, then trains.

```bash
python -m corpus_prep.corpus_train TIMESTAMP DATASET MR_FILTER PARTITION_RULE \
    [--vector-size N] [--max-cpu K]
```

```bash
# Example
python -m corpus_prep.corpus_train 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL

# With custom embedding size and CPU limit
python -m corpus_prep.corpus_train 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    --vector-size 200 --max-cpu 8
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
            ├── *.model.syn1.npy
            ├── *.model.wv.vectors_ngrams.npy
            └── *.model.wv.vectors_vocab.npy
```

> **Resources:** training on 700K proteins / 103M MR vocabulary requires ~150 GB RAM and ~5 hours on a 48-vCPU machine.

---

### Stage 2b — Control corpus (optional)

If you have a control dataset, this stage builds its BioWord corpus locally using the MR vocabulary from the real run (without re-running BigQuery). It runs the Aho-Corasick algorithm to find pattern matches.

Requires a `control_DATASET_sequence_dataset.csv` file inside the corresponding dataset input folder.

```bash
python -m corpus_prep.corpus_prep_control TIMESTAMP DATASET MR_FILTER PARTITION_RULE
```

```bash
# Example
python -m corpus_prep.corpus_prep_control 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL
```

Output: `*-s3_corpus_control_for_eval` file placed alongside the regular corpus shards.

---

### Stage 3 — Protein vector computation

Computes a mean embedding vector for each protein by summing its BioWord embeddings and dividing by the number of words. Outputs `.tsv` files (vectors + metadata) for downstream stages.

```bash
python -m model_eval.model_embeddings TIMESTAMP DATASET MR_FILTER PARTITION_RULE \
    [--control] [--mmap] [--metadata]
```

```bash
# Standard run
python -m model_eval.model_embeddings 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL

# Control dataset
python -m model_eval.model_embeddings 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL --control

# Memory-constrained machine: use mmap to reduce RAM (slower load)
python -m model_eval.model_embeddings 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL --mmap
```

| Flag | Description |
|---|---|
| `--control` | Build vectors for the control dataset using the control corpus |
| `--mmap` | Memory-map the model (lower RAM, 4× slower load) |
| `--metadata` | Only produce the `metadata.tsv`, skip vector computation |

**Output** under `INPUT_DATA_ROOT_PATH/DATASET/YYYYMMDD/RUNID/vector_output/`:
```
*-metadata.tsv       # protein labels (family_name, family_type, etc.)
*-vectors_bio.tsv    # protein vectors
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
    [--mmap] [--metadata]
```

```bash
# Example: embed BSC proteins using a model trained on familyDataset
python -m model_eval.model_embeddings_cross \
    2025_01_01_H11_M14_S21 BSC MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    2025_01_01_H09_M00_S00 FAMILY MR_FILTER_NONE PARTITION_RULE_USE_ALL
```

Output is placed under the **input dataset** run folder, tagged with a compact model label (`cross_<MODEL_TAG>`):
```
*-cross_family-vectors_bio.tsv
*-cross_family-metadata.tsv
```

Pass `--cross <MODEL_TAG>` to Stage 4 to use these vectors in dimensionality reduction.

---

### Stage 3c — Combine real + control vectors (optional)

If you want to run dimensionality reduction on real and control proteins together (required for t-SNE, which needs all points to compute the joint projection), run this before Stage 4:

```bash
python -m model_eval.model_combine_datasets TIMESTAMP DATASET MR_FILTER PARTITION_RULE
```

Produces `*-combined-metadata.tsv` and `*-combined-vectors_bio.tsv` in the `vector_output` folder. Pass `--control` to Stage 4 and beyond to use the combined files.

---

### Stage 4 — Dimensionality reduction

Reduces protein vectors (typically 100D) to 2D or 3D for visualization. Supports **t-SNE** (sklearn / openTSNE), **UMAP**, and **densMAP** via a JSON experiment configuration.

**Experiment config** — create `experiment.json` under `vector_output/experiments/`:

```json
{
    "reduction_method": "tsne",
    "tsne_implementation": "sklearn",
    "pca_n_components": 50,
    "n_components": [2],
    "random_state": [0, 1000, 537],
    "method": ["barnes_hut"],
    "perplexity": [10, 30, 50],
    "learning_rate": ["auto"],
    "max_iter": [5000],
    "n_jobs": [8],
    "verbose": 1
}
```

For **UMAP / densMAP**, use `"reduction_method": "umap"` and add UMAP-specific keys:

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

Arrays drive parameter sweeps — the longest array sets how many runs execute. Single-element arrays (or scalars) are reused for every run.

```bash
python -m model_eval.model_dim_reduction TIMESTAMP DATASET MR_FILTER PARTITION_RULE EXPERIMENT_FILE \
    [--control] [--cross MODEL_TAG] [--filter-col COLUMN VALUE]
```

```bash
# Standard t-SNE sweep
python -m model_eval.model_dim_reduction 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment

# With combined control data
python -m model_eval.model_dim_reduction 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment --control

# Using cross-embedded vectors (from Stage 3b)
python -m model_eval.model_dim_reduction 2025_01_01_H11_M14_S21 BSC MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment --cross family

# Drop proteins where metadata column 'partition_type' == 'mr' before reducing
python -m model_eval.model_dim_reduction 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment \
    --filter-col partition_type mr
```

| Flag | Description |
|---|---|
| `--control` | Use combined real + control vectors |
| `--cross MODEL_TAG` | Use cross-embedded vectors tagged with `MODEL_TAG` |
| `--filter-col COLUMN VALUE` | Drop rows where `COLUMN == VALUE` before reducing |

**Output** under `vector_output/EXPERIMENT_FILE/`:
```
*-vectors_pca-50.tsv
*-vectors_tsne-2_barnes_hut_30_auto_5000_0.tsv
*-vectors_umap-2_n15_d0.1_42.tsv
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
python -m model_eval.model_metrics 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    my_experiment family_name

# With parallelization
python -m model_eval.model_metrics 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL \
    my_experiment family_name --max-workers 8
```

| Argument | Description |
|---|---|
| `CLASS_LABEL_COLUMN` | Metadata column to use as ground-truth class (e.g. `family_name`, `family_type`) |
| `--control` | Use combined dataset |
| `--max-workers N` | Parallel workers for metric computation |

---

### Stage 6 — Visualization

#### Static charts

Renders scatter plots for each experiment run, colored by `family_name` and `family_type`:

```bash
python -m model_viz.model_viz_enhanced TIMESTAMP DATASET MR_FILTER PARTITION_RULE EXPERIMENT_NAME \
    [--control] [--mode single|grid] [--chart-type name|type|overlap] [--grid-resolution N]
```

```bash
# Single chart, colored by family name
python -m model_viz.model_viz_enhanced 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment

# Grid of all experiment runs
python -m model_viz.model_viz_enhanced 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment \
    --mode grid --chart-type type
```

Output charts saved under `vector_output/EXPERIMENT_NAME/charts/`.

#### Interactive explorer (Dash web app)

A browser-based explorer that handles 1M+ points via WebGL. Features: distinct-color / highlight / density-heatmap rendering modes, searchable dropdowns, CATH hierarchical drill-down, hover tooltips, PNG export.

```bash
python -m model_viz.explorer_app
# Open http://localhost:8050
```

The app reads from the `vector_output` folder of the configured run; configure the Run ID via the UI dropdowns or environment variables.

#### TensorBoard Projector

Interactive 3D projector for exploring embeddings with zoom, filter, and nearest-neighbor search. Note: browser memory limits sampling to ~10K points.

```bash
# Set up projector files
python -m corpus_prep.tensorboard_setup tsne TIMESTAMP DATASET MR_FILTER PARTITION_RULE [--control]
# or
python -m corpus_prep.tensorboard_setup pca TIMESTAMP DATASET MR_FILTER PARTITION_RULE

# The script will print the exact tensorboard serve command to run
```

#### Jupyter exploration notebooks

- `corpus_prep/corpus_prep_exploration.ipynb` — explore Stage 1 BigQuery outputs
- `model_eval/model_eval_exploration.ipynb` — explore embeddings and reduction results
- `model_viz/interactive_viz_explorer.py` — widget-based notebook explorer

---

## End-to-end example (test dataset)

The `TEST_GROUP` dataset is small (~483 proteins) and fast to process — good for verifying your setup works before running on large datasets.

```bash
# 0. Activate environment
source venv/bin/activate

# 1. Corpus prep — loads MR data into BigQuery and computes the BioWord corpus
python -m corpus_prep.corpus_prep_pipeline TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL
# Note the printed Run ID, e.g. 2025_01_01_H11_M14_S21-testGroupDataset-filter_none-partition_use_all

# 2. Train FastText model on the corpus
python -m corpus_prep.corpus_train 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL

# 3. Compute protein embedding vectors
python -m model_eval.model_embeddings 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL

# 4. Dimensionality reduction — place experiment.json in vector_output/experiments/ first
python -m model_eval.model_dim_reduction 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment

# 5. Visualize
python -m model_viz.model_viz_enhanced 2025_01_01_H11_M14_S21 TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL my_experiment
# Or launch the interactive explorer:
python -m model_viz.explorer_app
```

---

## Output folder structure

```
processed_datasets/
└── testGroupDataset/
    ├── testGroupDataset_*_PATTERNS.csv          # input (Stage 1 input)
    ├── testGroupDataset_*_POSITIONS.csv         # input (Stage 1 input)
    ├── testGroupDataset_sequence_dataset.csv    # input (Stage 1 input)
    ├── 20250101/
    │   ├── models/
    │   │   └── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all.model
    │   │       *.model.syn1.npy  *.model.wv.vectors_ngrams.npy  *.model.wv.vectors_vocab.npy
    │   ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-s3_corpus_for_train_0.gz
    │   ├── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all-s3_corpus_for_train_joined.csv
    │   └── 20250101_11_14_21-testGroupDataset-filter_none-partition_use_all/
    │       └── vector_output/
    │           ├── *-metadata.tsv
    │           ├── *-vectors_bio.tsv
    │           ├── experiments/
    │           │   └── my_experiment.json
    │           └── my_experiment/
    │               ├── *-vectors_pca-50.tsv
    │               ├── *-vectors_tsne-2_barnes_hut_30_auto_5000_0.tsv
    │               └── charts/
    │                   └── *.png
    └── logs/                                    # stage run logs
```

---

## Scale & resource guide

| Dataset | Proteins | Unique MRs | MR positions | RAM (train) | Train time |
|---|---|---|---|---|---|
| TEST_GROUP | ~483 | ~110K | ~1M | < 8 GB | < 5 min |
| FAMILY | ~700K | ~103M | ~3B | ~150 GB | ~5 h (48 vCPU) |

- **Stage 1 (BigQuery):** scales to any size; cost is determined by GCP query bytes processed.
- **Stage 3 (vector compute):** iterating 700K proteins takes ~5 h; no parallelization currently.
- **Stage 4 (t-SNE `barnes_hut`):** recommended over `exact` for large datasets to avoid memory overflow.
- **UMAP / densMAP:** generally faster than t-SNE for large datasets; densMAP also preserves density information.
