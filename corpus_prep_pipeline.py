from prefect import flow, task, tags
# from prefect.cache_policies import TASK_SOURCE, INPUTS
import os
import pandas as pd
import json
from itertools import chain
from datetime import datetime

# MR Filter types
MR_FILTER_NONE = {
    "by" : "none",
    "name" : "filter_none"
}

MR_FILTER_KEEP_SIGNIFICANT = {
    "condition" : "include",
    "by" : "significance",
    "name" : "filter_keep_significant"
}

MR_FILTER_DROP_SMR = {
    "condition" : "exclude",
    "by" : "type",
    "value" : "'SMR'",
    "name" : "filter_drop_smr"
}

MR_FILTER_DROP_NE = {
    "condition" : "exclude",
    "by" : "type",
    "value" : "'NE'",
    "name" : "filter_drop_ne"
}

MR_FILTER_DROP_NN = {
    "condition" : "!=",
    "by" : "type",
    "value" : "'NN'",
    "name" : "filter_drop_nn"
}

MR_FILTER_KEEP_LEN6PLUS = {
    "condition" : ">=",
    "by" : "length",
    "value" : 6,
    "name" : "filter_keep_length_6plus"
}

# Partition rules
PARTITION_RULE_USE_ALL = {
    "name" : "partition_use_all"
}

def custom_agg_list(series):
    if series.isnull().all():
        # avoid [NaN] for missing values on left join
        return []  # Return an empty list for all-NaN groups
    else:
        return list(series.dropna())  # Remove NaNs within the list

def compute_probability(pattern: str, aminoacid_frequencies):
    probability = 1
    for char in pattern:
        probability *= aminoacid_frequencies[char]
    
    return probability

#@task(cache_policy=TASK_SOURCE + INPUTS, log_prints=False)
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
    
def compute_pattern_repeats_matrix(row):
    sequence = row["sequence"]
    pattern_list = row["patterns"]
    partition_size = 0
    # each row of the matrix will correspond
    # to the ith char position in the sequence string
    partition_matrix = []
    for i in range(len(sequence) - 1):
        # initialize matrix row with empty array
        partition_matrix.append([])
        # then search for matching patterns that are
        # substrings of sequence at the ith position
        for pattern in pattern_list:
            if sequence.startswith(pattern, i):
                # if there's a match, add a pattern instance
                # to the ith row of the matrix
                partition_matrix[i].append(pattern)
                partition_size += 1
    if partition_size == 0:
        # if no matching patterns, use full sequence as only word
        partition_matrix[0].append(sequence)
    return partition_matrix
    
@task(log_prints=True)
def load_sequence_dataset(input_data_root_path: str, family_dataset_name: str):
    input_sequence_dataset_path = os.path.join(input_data_root_path,family_dataset_name, family_dataset_name+"_sequence_dataset.csv")
    sequence_dataset_df = pd.read_csv(input_sequence_dataset_path)
    return sequence_dataset_df

@task(log_prints=True)
def load_mr_dataset(input_data_root_path: str, family_dataset_name: str):
    input_mr_dataset_path = os.path.join(input_data_root_path,family_dataset_name, family_dataset_name+"_1_999999_1_ALL.json")

    with open(input_mr_dataset_path, 'r') as f:
        data = json.load(f)
    
    mr_dataset_df = pd.json_normalize(data, max_level=0) 
    return mr_dataset_df

@task(log_prints=True)
def save_corpus_dataset(input_data_root_path: str, family_dataset_name: str, corpus_dataset, filter_name: str, partition_rule_name: str):
    now_timestamp = datetime.now()
    date_formatted = now_timestamp.strftime("%Y%m%d")
    ts_formatted = now_timestamp.strftime("%Y%m%d_%H_%M_%S")
    parent_folder_path = os.path.join(input_data_root_path, family_dataset_name, date_formatted)
    os.makedirs(parent_folder_path, exist_ok=True)
    write_path = os.path.join(parent_folder_path, "corpus_full_"+family_dataset_name+"_"+filter_name+"_"+partition_rule_name+"_"+ts_formatted+".csv")
    corpus_dataset.to_csv(write_path, index=False)

@task(log_prints=True)
def count_aminoacid_frequency(sequence_dataset_df):
    # total count of aminoacid residues
    total_length = sequence_dataset_df["sequence"].str.len().sum()
    
    aminoacid_letters = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'U', 'O', 'X', 'B', 'Z', 'J']

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
def filter_mrs_with_query(mr_dataset_df, filter_attr, filter_condition, filter_value):
    mr_dataset_df = mr_dataset_df.query(f"{filter_attr} {filter_condition} {filter_value}")
    return mr_dataset_df

@flow(name="Filter Maximal Repeats", log_prints=True)
def filter_maximal_repeats(mr_dataset_df, sequence_dataset_df, filter):

    original_length = len(mr_dataset_df)

    if filter["by"] == "none":
        pass
    elif filter["by"] == "significance":
        mr_dataset_df = filter_mrs_with_statistical_significance(sequence_dataset_df, mr_dataset_df)
    else:
        mr_dataset_df = filter_mrs_with_query(mr_dataset_df, filter["by"], filter["condition"], filter["value"])
    print(len(mr_dataset_df) / original_length)
    print(mr_dataset_df.head(10))

    return mr_dataset_df

@task(log_prints=True)
def join_datasets(sequence_df, mr_df):

    # only keep columns relevant to the join
    mr_lean_df = mr_df[["pattern", "affected_protein_ids"]]

    # flatten the list of protein IDs for performing join
    mr_exploded_df = mr_lean_df.explode("affected_protein_ids")
    
    # rename columns for proper join key
    mr_exploded_df = mr_exploded_df.rename(columns={'affected_protein_ids': 'protein_id'})
    sequence_df = sequence_df.rename(columns={'id': 'protein_id'})

    print(len(sequence_df))
    print(len(mr_exploded_df))
    joined_df = sequence_df.join(mr_exploded_df.set_index("protein_id"), on="protein_id", validate="1:m")
    print(len(joined_df))

    # Set pattern as the aggregate column
    agg_columns = {'pattern': custom_agg_list}

    # Get all columns not in agg_columns and apply 'first' to them
    # as sequence dataset values will be the same across repeated rows
    default_agg = {col: 'first' for col in joined_df.columns if col not in agg_columns}

    # Apply groupby into new dataframe
    grouped_df = joined_df.groupby('protein_id').agg({**default_agg, **agg_columns}).reset_index(drop=True)

    # Rename the aggregated column
    grouped_df = grouped_df.rename(columns={'pattern': 'patterns'})

    # grouped_df has, for each sequence, a list of the filtered MRs patterns that belong to that seq
    return grouped_df

@task(log_prints=True)
def compute_pattern_repeats_in_order(joined_sequence_dataset_df):
    joined_sequence_dataset_df["word_partition_matrix"] = joined_sequence_dataset_df.apply(compute_pattern_repeats_matrix, axis=1)
    joined_sequence_dataset_df = joined_sequence_dataset_df.drop(columns=["patterns"])
    return joined_sequence_dataset_df

@task(log_prints=True)
def flatten_partition_matrix(corpus_matrix_df):
     corpus_matrix_df["word_partition"] = corpus_matrix_df["word_partition_matrix"].apply(lambda x: " ".join(list(chain.from_iterable(x))))
     corpus_matrix_df = corpus_matrix_df.drop(columns=["word_partition_matrix"])

     return corpus_matrix_df

@flow(name="Compute BioWord Partition", log_prints=True)
def compute_bioword_partition(sequence_df, mr_df, partition_rule):

    joined_sequence_dataset_df = join_datasets(sequence_df, mr_df)

    # TODO: apply filters based on rules?

    corpus_matrix_partition_df = compute_pattern_repeats_in_order(joined_sequence_dataset_df)

    corpus_df = flatten_partition_matrix(corpus_matrix_partition_df)
    print(corpus_df.head(10))
    return corpus_df

@flow(name="Prepare BioWord Protein Corpus", log_prints=True)
def prepare_corpus(input_data_root_path: str, family_dataset_name: str, filter, partition_rule):

    sequence_dataset_df = load_sequence_dataset(input_data_root_path, family_dataset_name)
    mr_dataset_df = load_mr_dataset(input_data_root_path, family_dataset_name)

    # SUB FLOW 1: Filter Maximal Repeats
    # execute flow by loading datasets and applying filter rule
    # return sequences and filtered MRs for next step flow
    filtered_mr_dataset = filter_maximal_repeats(mr_dataset_df, sequence_dataset_df, filter)

    # SUB FLOW 2: Compute BioWord Partition
    # apply word partitioning rule to the sequence dataset
    # based on the filtered MRs
    # return full corpus for next step flow
    corpus_dataset = compute_bioword_partition(sequence_dataset_df, filtered_mr_dataset, partition_rule)
    
    save_corpus_dataset(input_data_root_path, family_dataset_name, corpus_dataset, filter["name"], partition_rule["name"])


# run the flow!
if __name__=="__main__":
    # TODO: adapt dataset location to allow flexible setup
    # currently processed datasets holds the exact folder structure
    # as processed by mr-generator output
    input_data_root_path = os.path.join(os.path.dirname(os.getcwd()), "proteins_db", 'processed_datasets')
    family_dataset_name = "testGroupDataset"
    filter = MR_FILTER_KEEP_SIGNIFICANT
    partition_rule = PARTITION_RULE_USE_ALL

    with tags(filter["name"], partition_rule["name"]):
        prepare_corpus(input_data_root_path, family_dataset_name, filter, partition_rule)
