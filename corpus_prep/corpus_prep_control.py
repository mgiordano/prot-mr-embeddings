import ahocorasick
import pandas as pd
import time
import os
from dotenv import dotenv_values
import argparse
import logging
from utils.utils import dataset_names, data_step_names, filters, partition_rules
import utils.utils as corpus_prep_utils

cpu_count = os.cpu_count()

# --- Aho-Corasick Automaton Building ---
def build_automaton(patterns_list):
    """Builds and finalizes an Aho-Corasick automaton."""
    A = ahocorasick.Automaton(ahocorasick.STORE_ANY)
    for idx, pattern_str in enumerate(patterns_list):
        if pattern_str and isinstance(pattern_str, str):
            A.add_word(pattern_str, pattern_str) # Store pattern string itself
        else:
            logging.warning(f"Skipping invalid pattern: {pattern_str}")
    A.make_automaton()
    return A

# --- Main Processing Function ---
def process_sequences(sequences_series, automaton): 
    """
    Processes protein sequences against the pre-built automaton.
    Returns a list of strings, where each string is the ordered matched patterns for a sequence.
    """
    output_matches_list = []
    total_sequences = len(sequences_series)
    start_time_processing = time.time()

    for i, sequence in enumerate(sequences_series):
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time_processing
            rate = (i + 1) / elapsed if elapsed > 0 else float('inf')
            logging.info(f"Processed {i+1}/{total_sequences} sequences... ({elapsed:.2f}s elapsed, {rate:.0f} sequences/sec)")

        if not isinstance(sequence, str) or not sequence: # Handle NaN/None/empty strings
            output_matches_list.append('')
            continue

        found_matches_with_positions = []
        for end_index, matched_pattern_value in automaton.iter(sequence):
            start_index = end_index - len(matched_pattern_value) + 1
            found_matches_with_positions.append({'pattern': matched_pattern_value, 'start': start_index})

        if not found_matches_with_positions:
            output_matches_list.append('')
            continue

        # Sort matches by their start position
        found_matches_with_positions.sort(key=lambda x: x['start'])
        
        # Join patterns based on start order
        ordered_patterns_str = " ".join([match['pattern'] for match in found_matches_with_positions])
        output_matches_list.append(ordered_patterns_str)

    return output_matches_list

def get_control_sequences_file_path_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str):
    """Get the file path for control sequences dataset."""
    return corpus_prep_utils.get_control_sequences_file_path(input_data_root_path, family_dataset_name, timestamp)

def load_control_sequences(control_sequences_file_path: str):
    """Load control sequences from the processed datasets folder."""
    if not os.path.exists(control_sequences_file_path):
        raise FileNotFoundError(f"Control sequences file not found at {control_sequences_file_path}")
    
    sequences_df = pd.read_csv(control_sequences_file_path, dtype={'sequence': str})
    sequences_df['sequence'] = sequences_df['sequence'].fillna('')
    
    logging.info(f"Loaded {len(sequences_df)} control sequences from {control_sequences_file_path}")
    return sequences_df

def get_filtered_patterns_file_path_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    """Get or create the joined filtered patterns file from a run."""
    return corpus_prep_utils.get_filtered_patterns_file_path_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)

def load_filtered_patterns(patterns_file_path: str):
    """Load patterns from the joined CSV file."""
    try:
        patterns_df = pd.read_csv(patterns_file_path, usecols=["pattern"], dtype={"pattern": str})
        patterns_list = patterns_df["pattern"].dropna().unique().tolist()
        if not patterns_list:
            logging.error("No valid patterns found. Exiting.")
            return []
        logging.info(f"Loaded {len(patterns_list)} unique patterns from {patterns_file_path}")
        return patterns_list
    except Exception as e:
        logging.error(f"An error occurred loading patterns: {e}")
        return []

def save_control_results(input_data_root_path: str, family_dataset_name: str, timestamp: str, 
                        filter_name: str, partition_rule_name: str, sequences_df, matched_results_list):
    """Save the control results with the same naming convention."""
    # Create output DataFrame with the same structure as the main corpus output
    output_df = pd.DataFrame({
        'sequence_family_name': sequences_df.get('family_name', sequences_df.get('sequence_family_name', '')),
        'sequence_family_type': sequences_df.get('family_type', sequences_df.get('sequence_family_type', '')),
        'sequence_name': sequences_df.get('name', sequences_df.get('sequence_name', '')),
        'sequence': sequences_df['sequence'],
        'word_partition': matched_results_list
    })
    
    return corpus_prep_utils.save_control_results(input_data_root_path, family_dataset_name, timestamp, 
                                                  filter_name, partition_rule_name, output_df)

# #######################################
#      MAIN PROCESSING                  #
# #######################################

def process_control_sequences(input_data_root_path: str, family_dataset_name: str, timestamp: str, 
                             filter_name: str, partition_rule_name: str):
    """Main function to process control sequences with filtered patterns."""
    logging.info("START FLOW - Process BioWord Control Sequences")
    
    # 1. Get control sequences file path and load sequences
    logging.info("Loading control sequences...")
    control_sequences_file_path = get_control_sequences_file_path_from_run(input_data_root_path, family_dataset_name, timestamp)
    sequences_df = load_control_sequences(control_sequences_file_path)
    
    # 2. Get filtered patterns file path and load patterns
    logging.info("Loading filtered patterns...")
    patterns_file_path = get_filtered_patterns_file_path_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    patterns_list = load_filtered_patterns(patterns_file_path)
    if not patterns_list:
        logging.error("No patterns loaded. Exiting.")
        return
    
    # 3. Build Aho-Corasick automaton
    logging.info("Building Aho-Corasick automaton...")
    start_time_build = time.time()
    automaton = build_automaton(patterns_list)
    logging.info(f"Automaton built in {time.time() - start_time_build:.2f} seconds.")
    
    # 4. Process sequences
    logging.info("Processing control sequences...")
    matched_results_list = process_sequences(sequences_df['sequence'], automaton)
    
    # 5. Save results
    logging.info("Saving results...")
    output_path = save_control_results(input_data_root_path, family_dataset_name, timestamp, 
                                      filter_name, partition_rule_name, sequences_df, matched_results_list)
    
    logging.info(f"Control sequence processing completed. Results saved to {output_path}")
    logging.info("END FLOW - Process BioWord Control Sequences")
    return output_path

# #######################################
#      MAIN                             #
# #######################################

if __name__ == "__main__":
    # Load environment configuration
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    config = dotenv_values(dotenv_path)
    
    # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'corpus_prep_control.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    parser = argparse.ArgumentParser(description='Process BioWord Control Sequences with Filtered Patterns')
    # Corpus input arguments (same interface as corpus_train.py)
    parser.add_argument('timestamp', help='Run timestamp')
    parser.add_argument('dataset_name', help='Input protein dataset name')
    parser.add_argument('filter', help='MR filter')
    parser.add_argument('partition_rule', help='MR partition rule')
    
    args = parser.parse_args()

    # Input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    timestamp = args.timestamp
    family_dataset_name = getattr(dataset_names, args.dataset_name)
    filter_name = getattr(filters, args.filter).name
    partition_rule_name = getattr(partition_rules, args.partition_rule)["name"]

    overall_start_time = time.time()
    logging.info("Starting control sequence processing...")

    process_control_sequences(input_data_root_path, family_dataset_name, timestamp, 
                             filter_name, partition_rule_name)
    
    logging.info(f"Script finished in {time.time() - overall_start_time:.2f} seconds.")