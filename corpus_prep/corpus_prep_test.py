import pytest
import pandas as pd
import corpus_prep_pipeline as corpus_prep

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

