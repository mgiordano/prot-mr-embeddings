# thesis_exp

## Introduction

This project models a complex data pipeline to process an input protein dataset, create a BioWord representation of proteins based on their Maximal Repeats, then train a FastText model on a corpus of those representations and finally build a Protein vector based on the BioWord embeddings for each proteins. This vector space can be then reduced to 3 or 2 dimensions to analyze cluster, protein distances and extract other attributes from the built representations.

Throughout the pipeline, different tools and techniques are used:

- __Python__: main programming language and scripts
- __Prefect__: pipeline orchestration & monitoring for most stages
- __Docker & PostgreSQL__: prefect database
- __Pandas Dataframes__: data operations and I/O between stages
- __Google BigQuery__: fast, scalable and efficient data operations to compute BioWord representations
- __Google Storage__: I/O between BigQuery and local Python pipelines
- __FastText__: ML model to produce embeddings on Corpus
- __PCA & tSNE__: combination of dimensionality reduction techniques to visualize and explore results
- __Tensorboard__: interactive dashboard to play with reduction techniques, parameters and viz results in real time
- __Seaborn__: graph & chart results 
- __Jupyter Notebooks__: prototype, run experiments and build charts

## Dataset input

This project works in tandem with the output result of the Maximal Repeat calculations. The script to compute this can be found in the following repo, it's a modification of the original work done by Turjanski & Ferreiro:

https://github.com/mgiordano/prot-mr-generator

After running that project on a certain protein dataset, the output should be stored in a directory like:

```
processed_datasets/
├── protein_dataset_large/
│   ├── protein_dataset_large_1_999999_1_PATTERNS.csv           # Dataset of all computed MRs
│   ├── protein_dataset_large_1_999999_1_POSITIONS.csv          # Dataset of all mr-protein relations
│   ├── protein_dataset_large_sequence_dataset.csv              # Dataset of all proteins and their attributes
│   └── control_protein_dataset_large_sequence_dataset.csv      # Dataset of all control proteins and their attributes
└── protein_dataset_small/              
```
When setting up the `INPUT_DATA_ROOT_PATH` it should point to the root processed_datasets/ to enable you to later choose on which dataset to execute the different pipeline stages. Also, each protein_dataset_name folder will be used as intermediate output result of the different pipeline stages.

## Setup

1. Create Python Virtual Environment venv in project root directory:
```
python -m venv venv
```

2. Activate ennvironment and install requirements
```
source venv/bin/activate
pip install -r requirements.txt
```

3. Create a Google Cloud project and create a Service account that has full access to Cloud Storage and BigQuery. Download key to use via API. Store the file in project root folder as ```gcloud_saccount_key.json```

4. Create a storage bucket `protein_storage_bucket_name` that will be used both to upload initial input datasets into BQ and also serve as transient export location when downloading query results locally. For the latter case, in order not to incur in extra storage costs and to keep the bucket as clean as possible, you may set an Object lifecycle rule to delete when matching the following criteria (/20 being the prefix of the run timestamps used as folders for storing the exports):
```bash
1+ days since object was created
Name matches any prefix of: 'testGroupDataset/20', 'familyDataset/20', 'smallGroup/20'
```

4. Create .env file in project root folder with the following variables:


```bash
INPUT_DATA_ROOT_PATH="/full/path/to/processed_datasets"
GCLOUD_SERVICE_ACCOUNT_KEY="/full/path/to/gcloud_account_key.json"
GCLOUD_INPUT_BUCKET="protein_storage_bucket_name"
GCLOUD_PROJECT_ID="gcloud_project_id"
```

## Running Prefect

Most of the data pipelines included in this project run via Prefect. Before running them, you must start up the Prefect server from the root project folder:

```
./start_prefect.sh
```

This will spin up the PostgreSQL docker service that Prefect uses to store flow & task execution details. 

Once Prefect server is started, you may access it via ```https://localhost:4200```

You may always run this script as a replacement for ```prefect server start``` as it will ensure the right settings are active.

## Pipeline stages

The pipeline has many data processing stages that meet certain pre / post conditions, so in that sense they may be run separately and asynchronically, allowing for granular control on what to (re)compute.

### Stage 1: Corpus preparation pipeline
 This is the first stage in the global processing pipeline. Its job is to load the input datasets into BigQuery, apply a filtering rule to the Maximal Repeats and then compute the BioWord partition for each protein by following the specified partition rule. 

 __Why BigQuery?__
 When developing this solution, multiple data processing limits were hit when using larger input datasets. The Test Group dataset is very lightweight (has only 483 protein subjects, 110K unique maximal repeats and over 1M matched positions) and can be run locally just with a Dataframe. However, when trying with the Family dataset group (700K protein subjects, 100M unique MRs and 3B matching positions), dataframes were out of the question but not even a local SQL database (such as DuckDB) was able to do the job with 128 GB of RAM. BigQuery rose then as the perfect solution, as these data quantities are a piece of cake and the massive computation needed (joins especially) were done in a handful of minutes.

What to expect:
1. Within your Storage bucket, a folder matching `protein_dataset_name` will be created, inside of which you'll see the `*_sequence_dataset.csv`, `*_PATTERNS.csv` and `*_POSITIONS.csv`.

2. In BigQuery, corresponding `*_sequences`, `*_patterns` and `*_positions` tables will be created under a `protein_input` dataset.

3. Throughout the processing execution, different tables holding intermediate results will be stored under the `stage_results` dataset. Naming convention for these tables will be: `YYYY_MM_DD_Hh_mm_ss-protein_dataset_name-filter_rule_name-partition_rule_name-sN_stage_name` where stages could be `s1_filtered_mrs, s2_joined_mrs and s3_corpus`. Having intermediate results allows for better troubleshooting and data exploration of transformations performed before reaching the final s3 corpus state.

The unique key made up of the run timestamp, dataset name and rule names will be typically referred to as `Run ID` and should be referenced in subsequent stages in order to pin point exactly which output to perform the remaining training and evaluation transformations. This allows to run multiple corpus preparations (different datasets * different rules) and then build different models based off of them. Also, this ID will identify future pipeline stage outputs regardless of when they are run, as outputs will always refer to the original `Run ID` data used while additionaly adding other qualifiers of each stage.

 Ensure you have Prefect server up before running

```bash
./start_prefect.sh
```

Then, to execute this processing stage, run:

```bash
python -m corpus_prep.corpus_prep_pipeline DATASET MR_FILTER PARTITION_RULE [--dry-run]
```
The parameter values must match the attribute definitions in `utils/utils.py`. For example:

```bash
python -m corpus_prep.corpus_prep_pipeline TEST_GROUP MR_FILTER_NONE PARTITION_RULE_USE_ALL --dry-run
```

Dry runs only apply to stage transformations while initial input datasets will be loaded as usual. The stage results tables under a dry run will start with a `99_tmp-` prefix and will have `EXPIRATION=1 DAY` by default. This is useful when doing test runs or experimenting in order not to clutter your datasets with invalid or temporary results. 

You may monitor flow execution via the Prefect dashboard and also by looking at BigQuery jobs in your project. In order to explore results, please refer to `corpus_prep_exploration.ipnyb`.

### Stage 2: Corpus preparation - Model training

This is the second stage in the overall pipeline and its main responsibility is to train a FastText model on the previous stage corpus result. 

 Ensure you have Prefect server up before running

```bash
./start_prefect.sh
```

Then, to execute this processing stage, run:
```bash
python -m corpus_prep.corpus_train TIMESTAMP DATASET MR_FILTER PARTITION_RULE [--vector-size <N> --max-cpu <K>]
```

The `TIMESTAMP` argument should match the format `YYYY_MM_DD_Hh_Mm_ss` and, alongside the combination of the other parameters, correspond to an existing `Run ID` in the BigQuery datasets reflecting on the specific corpus instance to train this model.

First, this stage will trigger a BQ to Storage sharded export of the chosen corpus table, and subsequently will download all the shards locally. This export will be qualified with a `_for_train` suffix and will only include the BioWord partitions, without any header or other columns, as this is the raw corpus the embeddings model needs. 

The main training function of this stage needs to perform a single `.train()` call on the FastText model, so for that purpose all the shards will be assembled into a single joined file and a path to it will be provided to the model training (note there is an alternative function to do iterative `.train()` calls on each shard -this was built in order to overcome computing limitations but has since been discouraged from a theory point of view).

The `--vector-size` optional parameter should be an `int` representing the dimensional space of the target embeddings produced by the FastText model. Default is 100, following recommended standards. Other training parameters are fixed, such as `epochs`, `sg` (to use Skip-gram, better suited for sub-words) or the combination of `hs` and `negative` (currently, the GENSIM implementation of FastText being used only supports parallelization during training for the combination of `hs=1` and `n=0`, which are the defaults embedded in the code).

The `--max-cpu` optional parameter should be an `int` to limit the cores used in parallel training processing, as the default is set to use the maximum cpu core count.

- Gensim FastText implementation: https://radimrehurek.com/gensim/models/fasttext.html#introduction

- Facebook original implementation: https://fasttext.cc/docs/en/faqs.html

__Computation resources:__ do note that training a FastText model can be compute intensive as the size of the vocabulary grows. 150 GBs of RAM were needed to train on a corpus of 700K bioword proteins (sentences) and 103M MRs (vocabulary words) while it took 5+ hours to train on a 48 vCPU machine.

__Output files:__ All outputs will be saved under the `INPUT_DATA_ROOT_PATH/dataset_name` folder. For each distinct _date_ a sub-folder will be created and inside it all outputs belonging to runs performed that day will be organized (this will be the case for the remaining stages). On the one hand, the shards of the BQ Corpus export will be stored under the date folder while the outputs of the trained FastText model will be stored under the _models_ subfolder.

```
processed_datasets/
├── protein_dataset_large/
│   ├── 20250101/
│   │   └──models/
│   │       ├── 20250101_11_14_21-protein_dataset_large-filter_ne-partition_use_all.model
│   │       ├── 20250101_11_14_21-protein_dataset_large-filter_ne-partition_use_all.model.syn1.npy
│   │       ├── 20250101_11_14_21-protein_dataset_large-filter_ne-partition_use_all.model.wv.vectors_ngrams.npy
│   │       └── 20250101_11_14_21-protein_dataset_large-filter_ne-partition_use_all.model.wv.vectors_vocab.npy
│   ├── 20250101_11_14_21-protein_dataset_large-filter_ne-partition_use_all-s3_corpus_for_train_0.gz
│   ├── 20250101_11_14_21-protein_dataset_large-filter_ne-partition_use_all-s3_corpus_for_train_1.gz
│   └── 20250101_11_14_21-protein_dataset_large-filter_ne-partition_use_all-s3_corpus_for_train_joined.csv
└── protein_dataset_small/              
```

Notice that `.gz` is the compressed shard downloaded from BQ, the plain `.csv` shards will also be stored once decompressed as well as the fully joined corpus. Also, model output files may vary depending on the size of the model. Refer to the external documentation on how to properly interpret them.

### Control dataset corpus

Control data is useful for evaluating the resulting behaviour of the model. If a control dataset is present, it can be incorporated in the modeling but with a different treatment. Control data should not be used to calculate maximal repeats nor train the embeddings model, therefore it won't automatically fit with the previous stages. 

We would like instead to create protein vectors out of control data but simply using the MRs from the real dataset and by creating the embeddings out of the real protein corpus. Therefore, once previous stages have been fulfilled for a real protein dataset, you may run

```bash
python -m corpus_prep.corpus_prep_control TIMESTAMP DATASET MR_FILTER PARTITION_RULE
```

This script will download the unique MR vocabulary from the processed stages in BigQuery corresponding to Run ID and, for each control protein, it will locally run the Aho-Corasick algorithm to efficiently find pattern matches and build the control partition corpus. The resulting corpus file will be placed in the same corresponding folder with the `s3_corpus_control_for_eval` suffix.

Make sure to have the corresponding control dataset located in the input folder of the specified DATASET name.

### Stage 3: Model evaluation - Protein vectors creation

This is the third stage that comprises evaluating the trained model by computing the full vector for a protein based on the corresponding bioword (MRs) embeddings.

Because of limitations with how Prefect works (with serializing / deserializing objects between task calls), this stage is a plain Python script, so running a Prefect server is not needed for this case. However, log messages are stored in a `logs` folder at the root project in order to track run time, task progress, etc.

To execute this process stage run:

```bash
python -m model_eval.model_embeddings TIMESTAMP DATASET MR_FILTER PARTITION_RULE
```

By providing the `Run ID` parameter inputs, this stage will then proceed to load the outputted model from the previous step stored in the corresponding `models` folder, while also loading (and downloading if necessary) the corpus output from the provided Run ID (in this case, it will use the `_for_eval` variant, as headers and labels such as family_name, family_type, etc will be needed for future evaluation steps).

__Load times__: do note that loading a large trained model (such as the 700k protein case) could take up to 30 minutes and will use approx. 120 GB ram (using `mmap=true` did provide 30% RAM savings but increased load time by 4X).

Once all data is lodaded, the process will iterate each protein's BioWord partition, retrieve the embedding for each bioword and sum up all vectors, then divide by the total number of words, effectively building a vector representation for the protein that is the mean of its partition embeddings.

__Vector compute time__: currently, iterating over all 700k proteins and creating their vectors takes aprox 5 hours without any paralellization. This could be improved.

For future processing, a folder `vector_output` will be created under the corresponding `Run ID` folder path, storing the metadata (labels) as a `*-metadata.tsv` and the biovectors as a `*-vectors_bio.tsv`.

__Control data__: Use the `--control` flag to build the embeddings for the control dataset using the control corpus.

__Memory map__: Use the `--mmap` flag to set the memory mapping feature in the FastText model loading. Load time will be longer but less RAM will be used up. Also, memory mapping allows for thread sharing of the model memory space.

__Metadata__: Use the `--metadata` flag to only create the `metadata.tsv` output for the desired dataset.

### Stage 3 bis: Embeddings combination

If evaluating control data alongside real protein data, you will need to run the combination script before proceeding with the dimensionality reduction. This is because tSNE embeddings need to be calculated for both sets of data at the same time to properly compute lower dimensional groupings. Simply run:

```bash
python -m model_eval.model_combine_datasets TIMESTAMP DATASET MR_FILTER PARTITION_RULE
```

The `-combined` `*-metadata.tsv` and `*-vectors_bio.tsv` files will be placed in the corresponing `vector_output` folder.

### Stage 4: Model evaluation - Dimensionality reduction

The protein vectors created in the previous step will be encoded in the same dimension space than the bioword embeddings (usually 100 or more). In order to work, visualize or analyze the data in a more manageable space, this step will transform the data into N dimensions (usually 2 or 3) by applying PCA+TSNE.

For this stage, we're using tSNE as an aid to visualize the data in 2D/3D while preserving local vs. global distances. In order to properly interpret tSNE results, see: https://distill.pub/2016/misread-tsne/.

The proposed combination is first applying PCA to reduce to T dimensions (default recommendation is T=50) and then apply tSNE. Because this algorithm has a few important hyperparameters, this step can be parametrized with an `experiment.json` file that has the following structure:

```json
{
    "reduction_method" : "tsne",
    "tsne_implementation" : "sklearn",
    "pca_n_components" : 50,
    "n_components" : [2],
    "random_state" : [0, 1000, 537, 1281, 2, 100, 100, 1133, 208, 400007, 200322, 4294967295, 5643728],
    "method" : ["barnes_hut"],
    "perplexity" : [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "learning_rate" : ["auto"],
    "max_iter" : [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 10000, 5000, 5000],
    "n_jobs" : [16],
    "verbose" : 1
}
```

Essentially, the max length array will determine how many parameter combinations will be run (when array length is 1, all combinations will use that value). Parameter values not contained in an array are always global for all combinations.

- `tsne_implementation` specifies which t-SNE implementation to use: `"sklearn"` (default) or `"openTSNE"`. This parameter is stored in the experiment configuration and doesn't affect folder or file naming.
- `method`: determines the tSNE computation method. For sklearn implementation, `barnes_hut` greatly reduces memory requirements compared to the `exact` method (otherwise the 700k protein set can't be processed). For openTSNE, it will default to `auto` to let that framework decide based on dataset size.

This `.json` file should be placed under an `experiments` folder within the corresponding `vector_output` for the desired `Run ID`

Then, the whole script can be run as a plain python script:

```bash
python -m model_eval.model_dim_reduction TIMETSTAMP DATASET MR_FILTER PARTITION_RULE EXPERIMENT_JSON_FILE_NAME [--control]
```

The output will be placed under a folder named after the `EXPERIMENT_JSON_FILE_NAME` and will contain `.tsv` files for each parameter combination run of tSNE, following the naming:

```
*-vectors_pca-[pca_n_components].tsv
*-vectors_tsne-[n_components]_[method]_[perplexity]_[learning_rate]_[max_iter]_[random_state].tsv
```
### Stage 5: Model evaluation - Visualization

Effectively visualizing the results is a key part of interpreting the model and deriving some intuitions / conclusions. There are several tools and ways to achieve this.

#### TensorFlow Projector

This is an interactive tool provided by TensorFlow to visualize vector projections in 2 or 3 dimensions, allowing to zoom in / out, filter based on lables, highlighting searches, etc. It is very powerful to interactively play with the data and see how hyperparameter changes can affect the reults. However, due to memory limitations, it will not load a huge dataset, effectively sampling to as much as 10K subjects.

In order to set it up locally, you may run:

```bash
python -m corpus_eval.tensorboard_setup <tsne|pca> TIMESTAMP DATASET MR_FILTER PARTITION_RULE [--control]
```

By specifying either `tsne` or `pca` the script will look for the corresponding reduced projections, searching for the `.tsv` file under the `vector_output` folder for that `Run ID`. When running with `--control` the `-combined` embeddings data will be used. 

When the setup ends, it will print in the console the exact command to run the TensorBoard server locally, which can then be accessed via the corresponding localhost URL and PORT.

#### Chart visualizations

For complete visualizations of the data, charts can be rendered using this script by running:

```bash
python -m model_eval.model_visualization TIMETSTAMP DATASET MR_FILTER PARTITION_RULE EXPERIMENT_JSON_FILE_NAME [--control]
```

It will look in the experiment output folder for all `.tsv` files generated by each experiment and will render the data points by labeling both for family name and for family type. Results will be placed under the `charts` subfolder in the corresponding experiment location.