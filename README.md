# thesis_exp

## Introduction

This project models a complex data pipeline to process an input protein dataset, create a BioWord representation of proteins based on their Maximal Repeats, then train a FastText model on a corpus of those representations and finally build a Protein vector based on the BioWord embeddings for each proteins. This vector space can be then reduced to 3 or 2 dimensions to analyze cluster, protein distances and extract other attributes from the built representations.

Throughout the pipeline, different tools and techniques are used:

- Python: main programming language and scripts
- Prefect: pipeline orchestration & monitoring for most stages
- Docker & PostgreSQL: prefect database
- Pandas Dataframes: data operations and I/O between stages
- Google BigQuery: fast, scalable and efficient data operations to compute BioWord representations
- Google Storage: I/O between BigQuery and local Python pipelines
- FastText: ML model to produce embeddings on Corpus
- PCA & tSNE: combination of dimensionality reduction techniques to visualize and explore results
- Tensorboard: interactive dashboard to play with reduction techniques, parameters and viz results in real time
- Seaborn: graph & chart results 
- Jupyter Notebooks: prototype, run experiments and build charts

## Dataset input

This project works in tandem with the output result of the Maximal Repeat calculations that can be found in:

After running that project on a certain protein dataset, the output should be stored in a directory like:

```
processed_datasets/
├── protein_dataset_large/
│   ├── protein_dataset_large_1_999999_1_PATTERNS.csv   # Dataset of all computed MRs
│   ├── protein_dataset_large_1_999999_1_POSITIONS.csv  # Dataset of all mr-protein relations
│   └── protein_dataset_large_sequence_dataset.csv      # Dataset of all proteins and their attributes
└── protein_dataset_small/              
```
When setting up the INPUT_DATA_ROOT_PATH it should point to the root processed_datasets/ to enable you to later choose on which dataset to execute the different pipeline stages. Also, each protein_dataset_name folder will be used as intermediate output result of the different pipeline stages.

## Setup

1. Create Python Virtual Environment venv

2. Install requirements

3. Create a Google Cloud project and create a Service account that has full access to BigQuery. Download key to use via API. Store the file in project root folder as ```"gcloud_saccount_key.json"```

4. Create .env file in project root folder with the following variables:

```
INPUT_DATA_ROOT_PATH=/full/path/to/processed_datasets
GCLOUD_SERVICE_ACCOUNT_KEY=/full/path/to/gcloud_account_key.json
GCLOUD_INPUT_BUCKET=protein_storage_bucket_name
GCLOUD_PROJECT_ID=gcloud_project_id
```

## Running Prefect

Most of the data pipelines included in this project run via Prefect. Before running them, you must start up the Prefect server from the root project folder:

```
./start_prefect.sh
```

This will spin up the PostgreSQL docker service that Prefect uses to store flow & task execution details. 

Once Prefect server is started, you may access it via ```https://localhost:4200```

You may always run this script as a replacement for ```prefect server start``` as it will ensure the right settings are active.