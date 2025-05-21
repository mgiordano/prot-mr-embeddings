import pytest
import pandas as pd
import corpus_prep_pipeline as corpus_prep
from utils.utils import dataset_names, filters
from dotenv import dotenv_values

config = dotenv_values("../.env")
input_data_root_path = config["INPUT_DATA_ROOT_PATH"]

@pytest.fixture
def sequence_test_df():
    return corpus_prep.load_sequence_dataset(input_data_root_path, dataset_names.TEST_GROUP)

@pytest.fixture
def sequence_nano_df():
    return corpus_prep.load_sequence_dataset(input_data_root_path, dataset_names.NANO_GROUP)

@pytest.fixture
def mr_test_df():
    return corpus_prep.load_mr_dataset(input_data_root_path, dataset_names.TEST_GROUP)

@pytest.fixture
def mr_nano_df():
    return corpus_prep.load_mr_dataset(input_data_root_path, dataset_names.NANO_GROUP)

# #######################################
#      INPUT DATA VALIDATION TESTS      #
# #######################################

def check_affected_proteins(mr_row, sequence_test_df):
    '''For a given MR, check if pattern is a valid substring of affected protein'''
    pattern = mr_row["pattern"]
    instances = mr_row["instances"]
    affected_count = 0
    for affected_protein in mr_row["affected_proteins"]:
        # for each affected protein id, retrieve sequence in seq dataset
        # sequence id corresponds to row index
        protein_id = affected_protein["protein_id"]
        sequence_row = sequence_test_df.iloc[protein_id]
        sequence = sequence_row["sequence"]
        for position in affected_protein["starting_positions"]:
            # for each declared position, check if valid substring in sequence
            assert sequence.startswith(pattern, position)
            # accumulate instances accross all affected prots
            affected_count += 1
    # check global affected count vs instances
    assert instances == affected_count

def test_validate_mr_dataset_test(sequence_test_df, mr_test_df):
    validate_mr_dataset(sequence_test_df, mr_test_df)

def test_validate_mr_dataset_nano(sequence_nano_df, mr_nano_df):
    validate_mr_dataset(sequence_nano_df, mr_nano_df)

def validate_mr_dataset(sequence_df, mr_df):
    '''Test correctness of input MR dataset, mainly that for each pattern affected proteins correspond'''
    mr_df.apply(check_affected_proteins, args=[sequence_df,], axis=1)

# #######################################
#       MR FILTERS TESTS                #
# #######################################

@pytest.fixture
def sample_sequence_dataset_df():
    # 1. Create a sample dataframe
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'family_name': ['FamilyA', 'FamilyB', 'FamilyA', 'FamilyC', 'FamilyB', 'FamilyA', 'FamilyB', 'FamilyC', 'FamilyA', 'FamilyB'],
        'name': ['Protein1', 'Protein2', 'Protein3', 'Protein4', 'Protein5', 'Protein6', 'Protein7', 'Protein8', 'Protein9', 'Protein10'],
        'sequence': ['MAIVMGR', 'MSVPTDG', 'MKLFTYQ', 
                     'MKTYLKQ', 'MSVPTDG', 'MAIVMGR', 
                     'MSVPTDG', 'MKTYLKQ', 'MAIVMGR', 
                     'MSVPTDG']
    }
    return pd.DataFrame(data)

def test_count_aminoacid_frequency(sample_sequence_dataset_df):
    expected_result = {'A': 3/70, 'R': 3/70, 'N': 0, 'D': 4/70, 'C': 0, 
                       'E': 0, 'Q': 3/70, 'G': 7/70, 'H': 0, 'I': 3/70, 
                       'L': 3/70, 'K': 5/70, 'M': 13/70, 'F': 1/70, 'P': 4/70, 
                       'S': 4/70, 'T': 7/70, 'W': 0, 'Y': 3/70, 'V': 7/70, 
                       'U': 0.0, 'O': 0.0, 'X': 0.0, 'B': 0.0, 'Z': 0.0, 'J': 0.0}
    
    aminoacid_frequencies = corpus_prep.count_aminoacid_frequency(sample_sequence_dataset_df)
    sum = 0
    # check all letters are present in calculated result
    for letter in expected_result:
        calculated_freq = aminoacid_frequencies[letter]
        sum += calculated_freq
        assert expected_result[letter] == calculated_freq
    
    # check calculated result has no extra entries
    assert len(expected_result.keys()) == len(aminoacid_frequencies.keys())
    
    # check all relative frequencies sum 1 (except rounding error)
    assert sum - 1 < 0.00000000001

def check_filtered_values(df, filter):
    original_size = len(df)
    count_value = (df[filter.by] == filter.value).sum()
    filtered_mr_df = corpus_prep.filter_mrs_with_query(df, filter)
    assert len(filtered_mr_df) == original_size - count_value
    assert (filtered_mr_df[filter.by] == filter.value).sum() == 0

def test_filter_with_query_types_test(mr_test_df):
    validate_filter_with_query_types(mr_test_df)

def test_filter_with_query_types_nano(mr_nano_df):
    validate_filter_with_query_types(mr_nano_df)

def validate_filter_with_query_types(mr_df):
    ## Drop SMR case
    check_filtered_values(mr_df, filters.MR_FILTER_DROP_SMR)
    ## Drop NN case
    check_filtered_values(mr_df, filters.MR_FILTER_DROP_NN)
    ## Drop NE case
    check_filtered_values(mr_df, filters.MR_FILTER_DROP_NE)

def test_filter_with_query_len_test(mr_test_df):
    validate_filter_with_query_len(mr_test_df)

def test_filter_with_query_len_nano(mr_nano_df):
    validate_filter_with_query_len(mr_nano_df)

def validate_filter_with_query_len(mr_df):
    ## Keep length >= 6 case
    filter = filters.MR_FILTER_KEEP_LEN6PLUS
    original_size = len(mr_df)
    count_value = (mr_df[filter.by] < filter.value).sum()
    filtered_mr_df = corpus_prep.filter_mrs_with_query(mr_df, filter)
    assert len(filtered_mr_df) == original_size - count_value
    assert (filtered_mr_df[filter.by] < filter.value).sum() == 0

# #######################################
#       PARTITION TESTS                 #
# #######################################

def check_matching_mrs(row, joined_df):
    protein_id = row["protein_id"]
    pattern = row["pattern"]

    join_matching_prot_df = joined_df[(joined_df["protein_id"] == protein_id)]
    exploded_join_matching_prot_df = join_matching_prot_df.explode("pattern_positions")
    exploded_join_matching_prot_df['pattern'] = exploded_join_matching_prot_df["pattern_positions"].apply(lambda x: None if pd.isna(x) else x['pattern'])
    exploded_join_matching_prot_df['starting_positions'] = exploded_join_matching_prot_df["pattern_positions"].apply(lambda x: [] if pd.isna(x) else x['starting_positions'])
    exploded_join_matching_df = exploded_join_matching_prot_df[exploded_join_matching_prot_df["pattern"] == pattern]
    # check there's only one row corresponding to the pattern-protein match
    assert len(exploded_join_matching_df) == 1
    # check index positions are well preserved
    assert row["starting_positions"] == exploded_join_matching_df["starting_positions"].iloc[0]

def test_join_test(sequence_test_df, mr_test_df):
    # get random sample
    sample_mr_df = mr_test_df.sample(frac=0.05)
    validate_join(sequence_test_df, sample_mr_df)

def test_join_nano(sequence_nano_df, mr_nano_df):
    validate_join(sequence_nano_df, mr_nano_df)

def validate_join(sequence_df, mr_df):
    '''Pre condition is all MRs belong to at least one sequence but not vice versa'''
    joined_df = corpus_prep.join_datasets(sequence_df, mr_df)

    # check grouped join row count equals sequences
    assert len(joined_df) == len(sequence_df)
    
    # check all sequence IDs are present after join
    assert sequence_df["id"].nunique() == joined_df["protein_id"].nunique()

    # for sequences with no matching MRs, check for 
    # single sequence row with empty pattern_positions
    
    affected_protein_ids = {elem['protein_id'] for list in mr_df['affected_proteins'] for elem in list if isinstance(elem, dict)}
    unmatched_sequence_df = sequence_df[~sequence_df["id"].isin(affected_protein_ids)]
    assert len(unmatched_sequence_df) == joined_df["pattern_positions"].apply(lambda x: 1 if x == [] else 0).sum()

    # expand and normalize joined and mr datasets
    exploded_joined_df = joined_df.explode("pattern_positions")
    exploded_mr_df = mr_df.explode("affected_proteins")
    # extract columns and check for NaNs in case the join didnt match sequences with MRs
    exploded_mr_df['protein_id'] = exploded_mr_df["affected_proteins"].apply(lambda x: None if pd.isna(x) else x['protein_id'])
    exploded_mr_df['starting_positions'] = exploded_mr_df["affected_proteins"].apply(lambda x: [] if pd.isna(x) else x['starting_positions'])

    # check all MRs are present and consistent
    # TODO: improve check efficiency 
    exploded_mr_df.apply(check_matching_mrs, args=(joined_df,), axis=1)

    # check there are no extra rows in joined result
    # joined dataset is at least all mr-protein combinations
    # and at most the extra of one row per unmatched sequence
    assert len(exploded_joined_df) == len(exploded_mr_df) + len(unmatched_sequence_df)


def check_word_partition_order(row):
    sequence = row["sequence"]
    partition_matrix = row["word_partition_matrix"]

    # check all positions are valid in the sequence
    for i, pattern_positions in enumerate(partition_matrix):
        for pattern in pattern_positions:
            assert sequence.startswith(pattern, i)

def test_compute_pattern_repeats_in_order_test(sequence_test_df, mr_test_df):
    validate_compute_pattern_repeats_in_order(sequence_test_df, mr_test_df)

def test_compute_pattern_repeats_in_order_nano(sequence_nano_df, mr_nano_df):
    validate_compute_pattern_repeats_in_order(sequence_nano_df, mr_nano_df)

def validate_compute_pattern_repeats_in_order(sequence_df, mr_df):
    joined_df = corpus_prep.join_datasets(sequence_df, mr_df)
    with pd.option_context("display.max_columns", None):
        print(joined_df)
    corpus_matrix_partition_df = corpus_prep.compute_pattern_repeats_in_order(joined_df)
    corpus_matrix_partition_df.apply(check_word_partition_order, axis=1)
