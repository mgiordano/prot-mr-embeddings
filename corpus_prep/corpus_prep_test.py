import pytest
import pandas as pd
import corpus_prep_pipeline as corpus_prep
from dotenv import dotenv_values

config = dotenv_values("../.env")
input_data_root_path = config["INPUT_DATA_ROOT_PATH"]

@pytest.fixture
def sequence_test_df():
    return corpus_prep.load_sequence_dataset(input_data_root_path, corpus_prep.TEST_GROUP_DATASET_NAME)

@pytest.fixture
def mr_test_df():
    return corpus_prep.load_mr_dataset(input_data_root_path, corpus_prep.TEST_GROUP_DATASET_NAME)

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

def test_validate_mr_dataset(sequence_test_df, mr_test_df):
    '''Test correctness of input MR dataset, mainly that for each pattern affected proteins correspond'''
    mr_test_df.apply(check_affected_proteins, args=[sequence_test_df,], axis=1)

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

