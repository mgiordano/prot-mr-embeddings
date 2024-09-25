from prefect import flow, tags, task
#from prefect_dask import DaskTaskRunner
import os
#import re
#import pandas as pd
#import numpy as np
#from itertools import chain
from dotenv import dotenv_values
from corpus_prep_utils import dataset_names, data_step_names, filters, partition_rules
from database.database_helper import DatabaseHelper
import corpus_prep_utils

cpu_count = os.cpu_count()
worker_count = cpu_count
stream_count = cpu_count

@task(log_prints=True)
def load_sequence_dataset(input_data_root_path: str, family_dataset_name: str, dataset_database_helper):
    file_name = family_dataset_name+"_sequence_dataset.csv"
    input_data_root_path = os.path.join(input_data_root_path,family_dataset_name)
    dataset_database_helper.load_sequences_dataset(input_data_root_path, file_name)
    return dataset_database_helper

@task(log_prints=True)
def load_patterns_dataset(input_data_root_path: str, family_dataset_name: str, dataset_database_helper):
    file_name = family_dataset_name+"_1_999999_1_PATTERNS.csv"
    input_patterns_dataset_path = os.path.join(input_data_root_path,family_dataset_name)
    dataset_database_helper.load_patterns_dataset(input_patterns_dataset_path, file_name)
    return dataset_database_helper

@task(log_prints=True)
def load_positions_dataset(input_data_root_path: str, family_dataset_name: str, dataset_database_helper):
    file_name = family_dataset_name+"_1_999999_1_POSITIONS.csv"
    input_positions_dataset_path = os.path.join(input_data_root_path,family_dataset_name)
    dataset_database_helper.load_positions_dataset(input_positions_dataset_path, file_name)
    return dataset_database_helper

@task(log_prints=True)
def save_results_from_db(db_helper, dataset_stage: str, is_temp=False, cluster_columns=[], filter_name = "", partition_rule_name = ""):
    table_name = db_helper.run_id + "-" + dataset_stage
    options = []
    if is_temp:
        table_name = "99_tmp-" + table_name
        expiration_option = {
            "key" : "expiration_timestamp",
            "value" : "TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)"
        }
        options.append(expiration_option)
    db_helper.save_to_table(table_name, cluster_columns, options)
    return table_name

@task(log_prints=True)
def save_results_from_local(db_helper, dataset_df, input_data_root_path: str, dataset_stage: str, filter_name = "", partition_rule_name = ""):
    run_id = db_helper.run_id + "-" + dataset_stage
    parent_folder_path = corpus_prep_utils.save_local(input_data_root_path, db_helper.dataset_name, run_id, dataset_df)
    file_name = run_id + ".csv"
    date = run_id.split("_")[0]
    gcs_folder = os.path.join(db_helper.dataset_name, date)
    # save part file in GCS
    gcs_uri = db_helper.storage_helper.upload_file(parent_folder_path, file_name, gcs_folder)
    # return file location for load job once all tasks have finished
    return gcs_uri

@task(log_prints=True)
def export_corpus_to_gcs(db_helper, table_name):
    gcs_root_path = os.path.join(db_helper.dataset_name, "corpus")

    # export corpus for training
    # by selecting only word partition column
    gcs_root_path_for_train = os.path.join(gcs_root_path, "for_train")
    export_columns = ["word_partition"]
    # file name suffix uses wildcard
    # to shard table export
    train_suffix = "_for_train_*"
    db_helper.export_table_to_gcs_as_csv(table_name, gcs_root_path_for_train, export_columns, train_suffix)

    # export corpus for evaluation
    # with all label columns and header
    gcs_root_path_for_eval = os.path.join(gcs_root_path, "for_eval")
    eval_suffix = "_for_eval_*"
    extra_options = [
        {
            "key" : "header",
            "value" : "TRUE"
        }
    ]
    db_helper.export_table_to_gcs_as_csv(table_name, gcs_root_path_for_eval, 
                                         file_name_suffix=eval_suffix, extra_options=extra_options)
    
# #######################################
#      SUB FLOW 2                       #
# #######################################

@task(log_prints=True)
def join_datasets(db_helper, pattern_source_table):
    
    # join patterns with the sequences they
    # appear in, providing a list of all instance positions
    db_helper.select_all_sequence_mrs(pattern_source_table)

    # save step 2 partial results
    # by executing chained db operations
    # and materializing results for next step
    # if dry run, save temp, if not persist
    cluster_columns = ["sequence_family_name"]
    table_name = save_results_from_db(db_helper, data_step_names.S2_JOINED_MR, 
                                      is_temp=db_helper.dry_run, cluster_columns=cluster_columns)

    return table_name

# #######################################
#      SF2 v REMOTE BQ                  #
# #######################################

@task(log_prints=True)
def compute_partition_matrix(db_helper, source_table, partition_rule):
    db_helper.select_bioword_partition(source_table)
    cluster_columns = ["sequence_family_name"]
    table_name = save_results_from_db(db_helper, data_step_names.S3_CORPUS, 
                                      is_temp=db_helper.dry_run, cluster_columns=cluster_columns)
    return table_name

@flow(name="Compute BioWord Partition - BQ", log_prints=True)
def compute_bioword_partition_bq(db_helper, patterns_source_table, partition_rule):
    joined_mrs_table_name = join_datasets(db_helper, patterns_source_table)
    bioword_partition_table_name = compute_partition_matrix(db_helper, joined_mrs_table_name, partition_rule)
    
    # export corpus to sharded files
    # in Google Cloud Storage
    # for further use in Corpus Train Pipeline
    if not db_helper.dry_run:
        export_corpus_to_gcs(db_helper, bioword_partition_table_name)

# #######################################
#     SF2 v LOCAL PARALLEL              #
# #######################################
"""
def compute_pattern_repeats_matrix(row):
    sequence = row["sequence"]
    sequence_length = len(sequence)
    pattern_list = row["pattern_positions"]

    # each row of the matrix will correspond
    # to the ith char position in the sequence string
    # initialize all rows with empty arrays
    partition_matrix = np.frompyfunc(list, 0, 1)(np.empty((sequence_length,), dtype=object))
    # iterate through patterns
    for pattern_position in pattern_list:
        pattern = pattern_position["pattern"]
        # insert pattern in ith row of the matrix
        for position in pattern_position["starting_positions"]:
            partition_matrix[position].append(pattern)

    # For non matching MRs, BQ will return an array with an empty struct
    if len(pattern_list) == 1 and len(pattern_list[0]["starting_positions"]) == 0:
        # if no matching patterns, use full sequence as only word
        partition_matrix[0].append(sequence)
    return partition_matrix

@task(log_prints=True)
def flatten_partition_matrix(corpus_matrix_df):
     corpus_matrix_df["word_partition"] = corpus_matrix_df["word_partition_matrix"].apply(lambda x: " ".join(list(chain.from_iterable(x))))
     corpus_matrix_df = corpus_matrix_df.drop(columns=["word_partition_matrix"])
     return corpus_matrix_df

@task(log_prints=True)
def compute_pattern_repeats_in_order(db_helper_init_params, stream_name, part):
    db_helper = DatabaseHelper(dataset_name=db_helper_init_params["dataset_name"], run_id=db_helper_init_params["run_id"],
                               dry_run=db_helper_init_params["dry_run"], input_data_root_path=db_helper_init_params["input_data_root_path"])
    processed_rows_dfs = []
    row_count = 0
    stream_reader = db_helper.read_rows_from_stream(stream_name)
    for page in stream_reader.rows().pages:
        rows_df = page.to_dataframe()
        rows_df["word_partition_matrix"] = rows_df.apply(compute_pattern_repeats_matrix, axis=1)
        rows_df = rows_df.drop(columns=["pattern_positions"])
        processed_rows_dfs.append(rows_df)
        row_count += page.num_items
    corpus_dataset = pd.concat(processed_rows_dfs, ignore_index=True)
    corpus_dataset = flatten_partition_matrix(corpus_dataset)
    return save_results_from_local(db_helper, corpus_dataset, db_helper.input_data_root_path, data_step_names.S3_CORPUS + "-part_" + str(part))

@flow(name="Compute BioWord Partition - Parallel", task_runner=DaskTaskRunner(cluster_kwargs={"n_workers": worker_count}), log_prints=True)
def compute_bioword_partition_parallel(db_helper_init_params, patterns_source_table, partition_rule):
    db_helper = DatabaseHelper(dataset_name=db_helper_init_params["dataset_name"], run_id=db_helper_init_params["run_id"],
                               dry_run=db_helper_init_params["dry_run"], input_data_root_path=db_helper_init_params["input_data_root_path"])
    joined_mrs_table_name = join_datasets(db_helper, patterns_source_table)

    # TODO: apply filters based on rules?

    read_session = db_helper.create_stream_session(joined_mrs_table_name, max_stream_count=stream_count)
    tasks = []
    i = 0
    for stream in read_session.streams:
        tasks.append(compute_pattern_repeats_in_order.submit(db_helper.get_init_params(), stream.name, i))
        i+=1
    gcs_uri = ""
    for task in tasks:
        task.wait()
    gcs_uri = tasks[0].result()
    file_part_suffix = r"part_\d+\.csv$"
    wildcard_suffix = "part_*.csv"
    gcs_uri = re.sub(file_part_suffix, wildcard_suffix, gcs_uri)
    db_helper.load_or_create_table("", "", "", db_helper.BQ_CORPUS_DATASET_NAME, run_id + "-" + data_step_names.S3_CORPUS, gcs_uri=gcs_uri)
"""

# #######################################
#      SUB FLOW 1                       #
# #######################################

def compute_probability(pattern: str, aminoacid_frequencies):
    probability = 1
    for char in pattern:
        probability *= aminoacid_frequencies[char]
    
    return probability

def calculate_number_of_patterns_of_length(pattern_length: int, sequence_dataset_df, possible_patterns_cache):
    # pattern_length - sequence_length + 1 is the total number of possible subsequences of pattern length in sequence
    # calculate for each sequence and then return global count
    if(pattern_length not in possible_patterns_cache):
        filtered_sequence_dataset_df = sequence_dataset_df[sequence_dataset_df["sequence"].str.len() >= pattern_length]
        possible_subsequences_number = filtered_sequence_dataset_df["sequence"].str.len() - pattern_length + 1
        count = int(possible_subsequences_number.sum()) * int(20**pattern_length)
        possible_patterns_cache[pattern_length] = count
        return count
    else:
        return possible_patterns_cache[pattern_length]

@task(log_prints=True)
def count_aminoacid_frequency(sequence_dataset_df):
    # total count of aminoacid residues
    total_length = sequence_dataset_df["sequence"].str.len().sum()
    
    aminoacid_letters = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
                         'U', 'O', 'X', 'B', 'Z', 'J']

    frequencies = {}
    for letter in aminoacid_letters:
        letter_counts = sequence_dataset_df['sequence'].str.count(letter)
        total_count = letter_counts.sum()
        frequency = total_count / total_length
        frequencies[letter] = frequency
    print(frequencies)
    return frequencies

@task(log_prints=True)
def compute_patterns_base_probability(pattern_dataset_df, aminoacid_frequencies):
    pattern_dataset_df["pattern_base_probability"] = pattern_dataset_df["pattern"].astype(str).apply(compute_probability, args=(aminoacid_frequencies,))
    return pattern_dataset_df

@task(log_prints=True)
def compute_possible_patterns_of_length(pattern_dataset_df, sequence_dataset_df):
    possible_patterns_cache = {}
    pattern_dataset_df["total_possible_patterns"] = pattern_dataset_df["pattern"].str.len().apply(calculate_number_of_patterns_of_length, args=(sequence_dataset_df,possible_patterns_cache))
    return pattern_dataset_df

@task(log_prints=True)
def filter_mrs_with_statistical_significance(sequence_dataset_df, mrs_dataset_df):

    # TODO: refactor to work withour mr df

    # compute each aminoacid frequency in the global sequence dataset
    aminoacid_frequencies = count_aminoacid_frequency(sequence_dataset_df)

    # extend the mr dataset by adding 2 columns:

    # 1 - compute patterns base probability by assuming aminoacid independence
    # and calculating the product of all the pattern residues freqs as measured above
    mrs_dataset_df = compute_patterns_base_probability(mrs_dataset_df, aminoacid_frequencies)

    # as an intermediate step, for each pattern compute all the possible patterns of length N
    # which will be the basis for the universe of all possible patterns of such length in the dataset 
    # given a specific pattern of length N
    mrs_dataset_df = compute_possible_patterns_of_length(mrs_dataset_df, sequence_dataset_df)

    # 2 - calculate pattern observed probability by dividing the total number of pattern instances
    # in the dataset over the total universe of possible patterns of same length in the dataset
    mrs_dataset_df["pattern_observed_probability"] = (mrs_dataset_df["instances"] / mrs_dataset_df["total_possible_patterns"]).astype(float)

    # Filter rows by only keeping the patterns whose observed prob is greater than the theoretical prob
    # this is a proxy for statistical significance: patterns that appear more than one would expect randomly
    # based on the dataset aminoacid distribution
    filtered_mrs_dataset_df = mrs_dataset_df[mrs_dataset_df["pattern_observed_probability"] > mrs_dataset_df["pattern_base_probability"]]
    #filtered_mrs_dataset_df = mrs_dataset_df[mrs_dataset_df["instances"] > (mrs_dataset_df["total_possible_patterns"] * mrs_dataset_df["pattern_base_probability"])]
    return filtered_mrs_dataset_df

@task(log_prints=True)
def filter_mrs_with_query(db_helper, filter):
    # escape string values with ''
    value = f"'{filter.value}'" if isinstance(filter.value, str) else filter.value
    filter = f"{filter.by} {filter.condition} {value}"
    return db_helper.select_patterns(filter)

@flow(name="Filter Maximal Repeats", log_prints=True)
def filter_maximal_repeats(db_helper, filter):

    if filter == filters.MR_FILTER_NONE:
       db_helper = db_helper.select_patterns()
    #elif filter == filters.MR_FILTER_KEEP_SIGNIFICANT:
        # TODO: review statistical filter
        #mr_dataset_df = filter_mrs_with_statistical_significance(db_helper)
    else:
        # if not special case, apply filter query with parameters
        db_helper = filter_mrs_with_query(db_helper, filter)
    
    # save step 1 partial results
    # by executing chained db operations
    # and materializing results
    table_name = save_results_from_db(db_helper, data_step_names.S1_FILTERED_MR, 
                                      is_temp=dry_run, cluster_columns=db_helper.patterns_table_cluster_columns)
    return table_name

# #######################################
#      MAIN FLOW                        #
# #######################################

@flow(name="Prepare BioWord Protein Corpus", log_prints=True)
def prepare_corpus(input_data_root_path: str, family_dataset_name: str, db_helper, filter, partition_rule):
    
    # Load data to create DB tables if not existent
    # and properly initialize db helper table references

    # sequences datasets are typically memory manageable
    # so it should be ok to execute and load df
    sequence_dataset_df = load_sequence_dataset(input_data_root_path, family_dataset_name, db_helper).select_sequences().execute()

    # patterns and positions are only loaded into DB
    # and references initialized within db helper
    load_patterns_dataset(input_data_root_path, family_dataset_name, db_helper)
    load_positions_dataset(input_data_root_path, family_dataset_name, db_helper)

    # SUB FLOW 1: Filter Maximal Repeats
    # execute flow by loading datasets and applying filter rule
    # pipeline operations are chained in the db helper
    # and materialized into a table
    filtered_mr_table_name = filter_maximal_repeats(db_helper, filter)

    # get db_helper init parameters to pass to parallel flow
    # and re instantiate db helper in each parallel task
    # to avoid serialization when passing full db_helper obj
    db_helper_init_params = db_helper.get_init_params()
    
    # SUB FLOW 2: Compute BioWord Partition
    # apply word partitioning rule to the sequence dataset
    # joined with the filtered MRs
    # return full corpus for next step flow
    corpus_dataset_df = compute_bioword_partition_bq(db_helper, filtered_mr_table_name, partition_rule)
    
    # save final step results
    #save_results_from_local(db_helper, corpus_dataset_df, input_data_root_path, data_step_names.S3_CORPUS)

# #######################################
#      MAIN                             #
# #######################################

# run the flow!
if __name__=="__main__":
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    print(dotenv_path)
    config = dotenv_values(dotenv_path)

    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    family_dataset_name = dataset_names.TEST_GROUP
    filter = filters.MR_FILTER_NONE
    partition_rule = partition_rules.PARTITION_RULE_USE_ALL
    dry_run = True

    run_id = corpus_prep_utils.create_run_id(family_dataset_name, filter.name, partition_rule["name"])
    dataset_database_helper = DatabaseHelper(family_dataset_name, input_data_root_path, run_id, dry_run)

    with tags(filter.name, partition_rule["name"]):
        prepare_corpus(input_data_root_path, family_dataset_name, dataset_database_helper, filter, partition_rule)