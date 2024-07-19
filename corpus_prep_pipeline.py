from prefect import flow, task, tags
# from prefect.cache_policies import TASK_SOURCE, INPUTS
import os
import pandas as pd
import json

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
    "name" : "filter_keep_len6plus"
}

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
        count = possible_subsequences_number.sum()
        possible_patterns_cache[pattern_length] = count
        return count
    else:
        return possible_patterns_cache[pattern_length]

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
    mrs_dataset_df["pattern_observed_probability"] = mrs_dataset_df["instances"] / mrs_dataset_df["total_possible_patterns"]

    # Filter rows by only keeping the patterns whose observed prob is greater than the theoretical prob
    # this is a proxy for statistical significance: patterns that appear more than one would expect randomly
    # based on the dataset aminoacid distribution
    filtered_mrs_dataset_df = mrs_dataset_df[mrs_dataset_df["pattern_observed_probability"] > mrs_dataset_df["pattern_base_probability"]]

    return filtered_mrs_dataset_df

@task(log_prints=True)
def filter_mrs_with_query(mr_dataset_df, filter_attr, filter_condition, filter_value):
    mr_dataset_df = mr_dataset_df.query(f"{filter_attr} {filter_condition} {filter_value}")
    return mr_dataset_df

@flow(name="Filter Maximal Repeats", log_prints=True)
def filter_maximal_repeats(input_data_root_path: str, family_dataset_name: str, filter):
    sequence_dataset_df = load_sequence_dataset(input_data_root_path, family_dataset_name)
    mr_dataset_df = load_mr_dataset(input_data_root_path, family_dataset_name)

    original_length = len(mr_dataset_df)

    if filter["by"] == "none":
        pass
    elif filter["by"] == "significance":
        mr_dataset_df = filter_mrs_with_statistical_significance(sequence_dataset_df, mr_dataset_df)
    else:
        mr_dataset_df = filter_mrs_with_query(mr_dataset_df, filter["by"], filter["condition"], filter["value"])
    print(len(mr_dataset_df) / original_length)
    datasets = {
        "mr_df" : mr_dataset_df,
        "sequence_df" :  sequence_dataset_df
    }
    return datasets

# run the flow!
if __name__=="__main__":

    # to do: adapt dataset location to allow flexible setup
    # currently processed datasets holds the exact folder structure
    # as processed by mr-generator output
    proteins_db_path = os.path.join(os.path.dirname(os.getcwd()), "proteins_db", 'processed_datasets')

    filter = MR_FILTER_KEEP_LEN6PLUS
    with tags(filter["name"]):

        # FLOW 1: Filter Maximal Repeats
        # execute flow by loading datasets and applying filter rule
        # return sequences and filtered MRs for next step flow
        datasets = filter_maximal_repeats(proteins_db_path, "testGroupDataset", filter)

        # FLOW 2: Compute BioWord Partition
        # apply word partitioning rule to the sequence dataset
        # based on the filtered MRs
        # return full corpus for next step flow
 
