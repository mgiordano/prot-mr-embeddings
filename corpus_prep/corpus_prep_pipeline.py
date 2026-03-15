import os
import logging
import time
from dotenv import dotenv_values
import argparse
from utils.utils import dataset_names, data_step_names, filters, partition_rules
from utils.database.database_helper import DatabaseHelper
import utils.utils as corpus_prep_utils

cpu_count = os.cpu_count()
worker_count = cpu_count
stream_count = cpu_count

def load_sequence_dataset(input_data_root_path: str, family_dataset_name: str, dataset_database_helper):
    logging.info("START TASK - load_sequence_dataset")
    file_name = family_dataset_name+"_sequence_dataset.csv"
    input_data_root_path = os.path.join(input_data_root_path,family_dataset_name)
    dataset_database_helper.load_sequences_dataset(input_data_root_path, file_name)
    logging.info("END TASK - load_sequence_dataset")
    return dataset_database_helper

def load_patterns_dataset(input_data_root_path: str, family_dataset_name: str, dataset_database_helper):
    logging.info("START TASK - load_patterns_dataset")
    file_name = family_dataset_name+"_1_999999_1_PATTERNS.csv"
    input_patterns_dataset_path = os.path.join(input_data_root_path,family_dataset_name)
    dataset_database_helper.load_patterns_dataset(input_patterns_dataset_path, file_name)
    logging.info("END TASK - load_patterns_dataset")
    return dataset_database_helper

def load_positions_dataset(input_data_root_path: str, family_dataset_name: str, dataset_database_helper):
    logging.info("START TASK - load_positions_dataset")
    file_name = family_dataset_name+"_1_999999_1_POSITIONS.csv"
    input_positions_dataset_path = os.path.join(input_data_root_path,family_dataset_name)
    dataset_database_helper.load_positions_dataset(input_positions_dataset_path, file_name)
    logging.info("END TASK - load_positions_dataset")
    return dataset_database_helper

def save_results_from_db(db_helper, dataset_stage: str, is_temp=False, cluster_columns=[], filter_name = "", partition_rule_name = ""):
    logging.info(f"START TASK - save_results_from_db (stage={dataset_stage})")
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
    logging.info(f"END TASK - save_results_from_db (stage={dataset_stage})")
    return table_name

def save_results_from_local(db_helper, dataset_df, input_data_root_path: str, dataset_stage: str, filter_name = "", partition_rule_name = ""):
    logging.info(f"START TASK - save_results_from_local (stage={dataset_stage})")
    run_id = db_helper.run_id + "-" + dataset_stage
    parent_folder_path = corpus_prep_utils.save_local(input_data_root_path, db_helper.dataset_name, run_id, dataset_df)
    file_name = run_id + ".csv"
    date = run_id.split("_")[0]
    gcs_folder = os.path.join(db_helper.dataset_name, date)
    # save part file in GCS
    gcs_uri = db_helper.storage_helper.upload_file(parent_folder_path, file_name, gcs_folder)
    logging.info(f"END TASK - save_results_from_local (stage={dataset_stage})")
    # return file location for load job once all tasks have finished
    return gcs_uri

def export_corpus_to_gcs(db_helper, table_name):
    logging.info("START TASK - export_corpus_to_gcs")
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
    logging.info("END TASK - export_corpus_to_gcs")
    
# #######################################
#      SUB FLOW 2                       #
# #######################################

def join_datasets(db_helper, pattern_source_table):
    logging.info("START TASK - join_datasets")
    
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

    logging.info("END TASK - join_datasets")
    return table_name

# #######################################
#      SF2 v REMOTE BQ                  #
# #######################################

def compute_partition_matrix(db_helper, source_table, partition_rule, metadata_table=None, metadata_join_key=None):
    logging.info("START TASK - compute_partition_matrix")
    db_helper.select_bioword_partition(source_table, metadata_table=metadata_table, metadata_join_key=metadata_join_key)
    cluster_columns = ["sequence_family_name"] if not metadata_table else []
    table_name = save_results_from_db(db_helper, data_step_names.S3_CORPUS, 
                                      is_temp=db_helper.dry_run, cluster_columns=cluster_columns)
    logging.info("END TASK - compute_partition_matrix")
    return table_name

def compute_bioword_partition_bq(db_helper, patterns_source_table, partition_rule, metadata_table=None, metadata_join_key=None):
    logging.info("START FLOW - Compute BioWord Partition - BQ")
    joined_mrs_table_name = join_datasets(db_helper, patterns_source_table)
    bioword_partition_table_name = compute_partition_matrix(db_helper, joined_mrs_table_name, partition_rule,
                                                           metadata_table=metadata_table, metadata_join_key=metadata_join_key)
    
    # export corpus to sharded files
    # in Google Cloud Storage
    # for further use in Corpus Train Pipeline
    if not db_helper.dry_run:
        export_corpus_to_gcs(db_helper, bioword_partition_table_name)
    logging.info("END FLOW - Compute BioWord Partition - BQ")

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

def count_aminoacid_frequency(sequence_dataset_df):
    logging.info("START TASK - count_aminoacid_frequency")
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
    logging.info(f"Aminoacid frequencies: {frequencies}")
    logging.info("END TASK - count_aminoacid_frequency")
    return frequencies

def compute_patterns_base_probability(pattern_dataset_df, aminoacid_frequencies):
    logging.info("START TASK - compute_patterns_base_probability")
    pattern_dataset_df["pattern_base_probability"] = pattern_dataset_df["pattern"].astype(str).apply(compute_probability, args=(aminoacid_frequencies,))
    logging.info("END TASK - compute_patterns_base_probability")
    return pattern_dataset_df

def compute_possible_patterns_of_length(pattern_dataset_df, sequence_dataset_df):
    logging.info("START TASK - compute_possible_patterns_of_length")
    possible_patterns_cache = {}
    pattern_dataset_df["total_possible_patterns"] = pattern_dataset_df["pattern"].str.len().apply(calculate_number_of_patterns_of_length, args=(sequence_dataset_df,possible_patterns_cache))
    logging.info("END TASK - compute_possible_patterns_of_length")
    return pattern_dataset_df

def filter_mrs_with_statistical_significance(sequence_dataset_df, mrs_dataset_df):
    logging.info("START TASK - filter_mrs_with_statistical_significance")

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
    logging.info("END TASK - filter_mrs_with_statistical_significance")
    return filtered_mrs_dataset_df

def filter_mrs_with_query(db_helper, filter):
    logging.info("START TASK - filter_mrs_with_query")
    result = db_helper.select_patterns(filter.get_filter_for_query())
    logging.info("END TASK - filter_mrs_with_query")
    return result

def filter_maximal_repeats(db_helper, filter):
    logging.info("START FLOW - Filter Maximal Repeats")

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
    logging.info("END FLOW - Filter Maximal Repeats")
    return table_name

# #######################################
#      MAIN FLOW                        #
# #######################################

def prepare_corpus(input_data_root_path: str, family_dataset_name: str, db_helper, filter, partition_rule,
                   metadata_table=None, metadata_join_key=None):
    logging.info("START FLOW ******************* Prepare BioWord Protein Corpus *******************")
    
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

    # SUB FLOW 2: Compute BioWord Partition
    # apply word partitioning rule to the sequence dataset
    # joined with the filtered MRs
    # return full corpus for next step flow
    corpus_dataset_df = compute_bioword_partition_bq(db_helper, filtered_mr_table_name, partition_rule,
                                                     metadata_table=metadata_table, metadata_join_key=metadata_join_key)
    
    # save final step results
    #save_results_from_local(db_helper, corpus_dataset_df, input_data_root_path, data_step_names.S3_CORPUS)
    logging.info("END FLOW ******************* Prepare BioWord Protein Corpus *******************")

# #######################################
#      MAIN                             #
# #######################################

if __name__=="__main__":
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'corpus_prep_pipeline.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]

    # input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    parser = argparse.ArgumentParser(description='Perform BioWord Protein Corpus Preparation')
    parser.add_argument('dataset_name', help='Input protein dataset name')
    parser.add_argument('filter', help='MR filter name')
    parser.add_argument('partition_rule', help='MR partition rule')
    parser.add_argument('--dry-run', action='store_true', help='Run the script in dry-run mode.')
    parser.add_argument('--metadata-table', default=None,
                        help='Optional metadata table name in protein_input dataset (e.g. bscDataset_metadata)')
    parser.add_argument('--metadata-join-key', default=None,
                        help='Join key in format corpus_col:metadata_col (e.g. sequence_name:seq_id)')
    
    args = parser.parse_args()

    # Validate metadata args are provided together
    if bool(args.metadata_table) != bool(args.metadata_join_key):
        parser.error('--metadata-table and --metadata-join-key must be provided together')

    family_dataset_name = getattr(dataset_names, args.dataset_name)
    filter = getattr(filters, args.filter)
    partition_rule = getattr(partition_rules, args.partition_rule)
    dry_run = args.dry_run

    run_id = corpus_prep_utils.create_run_id(family_dataset_name, filter.name, partition_rule["name"])
    dataset_database_helper = DatabaseHelper(family_dataset_name, input_data_root_path, run_id, dry_run)

    overall_start_time = time.time()
    logging.info("Starting corpus preparation...")

    prepare_corpus(input_data_root_path, family_dataset_name, dataset_database_helper, filter, partition_rule,
                   metadata_table=args.metadata_table, metadata_join_key=args.metadata_join_key)

    logging.info(f"Script finished in {time.time() - overall_start_time:.2f} seconds.")