import numpy as np
import pandas as pd
from dotenv import dotenv_values
import os
import sys
import logging
import argparse
from utils.utils import dataset_names, filters, partition_rules
import utils.utils as utils

def load_metadata_file(file_path):
    """Load metadata TSV file"""
    logging.info(f"Loading metadata from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    
    df = pd.read_csv(file_path, sep='\t')
    logging.info(f"Loaded {len(df)} metadata records")
    return df

def load_vectors_file(file_path):
    """Load vectors TSV file"""
    logging.info(f"Loading vectors from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vectors file not found: {file_path}")
    
    vectors = np.loadtxt(file_path, delimiter='\t', dtype=np.float32)
    logging.info(f"Loaded vectors with shape: {vectors.shape}")
    return vectors

def combine_datasets(vector_out_folder_path, run_id):
    """Combine original and control datasets into unified metadata and bio_vectors files"""
    logging.info("START TASK - combine_datasets")
    
    # Define file paths for original dataset
    original_metadata_file = os.path.join(vector_out_folder_path, f"{run_id}-metadata.tsv")
    original_vectors_file = os.path.join(vector_out_folder_path, f"{run_id}-vectors_bio.tsv")
    
    # Define file paths for control dataset
    control_metadata_file = os.path.join(vector_out_folder_path, f"{run_id}-control-metadata.tsv")
    control_vectors_file = os.path.join(vector_out_folder_path, f"{run_id}-control-vectors_bio.tsv")
    
    # Load original dataset files
    try:
        original_metadata = load_metadata_file(original_metadata_file)
        original_vectors = load_vectors_file(original_vectors_file)
    except FileNotFoundError as e:
        logging.error(f"Original dataset files not found: {e}")
        raise
    
    # Load control dataset files
    try:
        control_metadata = load_metadata_file(control_metadata_file)
        control_vectors = load_vectors_file(control_vectors_file)
    except FileNotFoundError as e:
        logging.error(f"Control dataset files not found: {e}")
        raise
    
    # Verify that metadata and vectors have matching row counts
    if len(original_metadata) != original_vectors.shape[0]:
        raise ValueError(f"Original metadata ({len(original_metadata)}) and vectors ({original_vectors.shape[0]}) row count mismatch")
    
    if len(control_metadata) != control_vectors.shape[0]:
        raise ValueError(f"Control metadata ({len(control_metadata)}) and vectors ({control_vectors.shape[0]}) row count mismatch")
    
    # Verify that vectors have the same number of dimensions
    if original_vectors.shape[1] != control_vectors.shape[1]:
        raise ValueError(f"Vector dimension mismatch: original {original_vectors.shape[1]} vs control {control_vectors.shape[1]}")
    
    logging.info(f"Original dataset: {len(original_metadata)} sequences")
    logging.info(f"Control dataset: {len(control_metadata)} sequences")
    
    # Ensure partition_type column exists in both datasets, fill with empty strings if missing
    if 'partition_type' not in original_metadata.columns:
        original_metadata = original_metadata.assign(partition_type="")
    if 'partition_type' not in control_metadata.columns:
        control_metadata = control_metadata.assign(partition_type="")

    # Combine metadata (original first, then control)
    combined_metadata = pd.concat([original_metadata, control_metadata], ignore_index=True)
    logging.info(f"Combined metadata: {len(combined_metadata)} sequences")
    
    # Combine vectors (original first, then control)
    combined_vectors = np.vstack([original_vectors, control_vectors])
    logging.info(f"Combined vectors shape: {combined_vectors.shape}")
    
    # Save combined metadata
    combined_metadata_file = os.path.join(vector_out_folder_path, f"{run_id}-combined-metadata.tsv")
    combined_metadata.to_csv(combined_metadata_file, sep='\t', index=False)
    logging.info(f"Saved combined metadata to: {combined_metadata_file}")
    
    # Save combined vectors
    combined_vectors_file = os.path.join(vector_out_folder_path, f"{run_id}-combined-vectors_bio.tsv")
    np.savetxt(combined_vectors_file, combined_vectors, delimiter='\t', fmt='%.20f')
    logging.info(f"Saved combined vectors to: {combined_vectors_file}")
    
    logging.info("END TASK - combine_datasets")
    return combined_metadata_file, combined_vectors_file

def main():
    """Main function to combine datasets"""
    logging.info("START FLOW ******************* Combine Original and Control Datasets *******************")
    
    # Parse arguments (same signature as model_dim_reduction.py)
    parser = argparse.ArgumentParser(description='Combine original and control protein embeddings')
    parser.add_argument('timestamp',
                        help='Run timestamp')
    parser.add_argument('dataset_name', 
                        help='Input protein dataset name')
    parser.add_argument('filter',
                        help='MR Filter')
    parser.add_argument('partition_rule',
                        help='MR partition rule')

    args = parser.parse_args()
    
    # Load environment configuration
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    config = dotenv_values(dotenv_path)
    
    # Input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    family_dataset_name = getattr(dataset_names, args.dataset_name)
    timestamp = args.timestamp
    filter_name = getattr(filters, args.filter).name
    partition_rule_name = getattr(partition_rules, args.partition_rule)["name"]
    
    # Create output structure
    date = utils.get_date_from_formatted_ts(timestamp)
    vector_out_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    run_id = timestamp + "-" + family_dataset_name + "-" + filter_name + "-" + partition_rule_name
    
    # Verify output folder exists
    if not os.path.exists(vector_out_folder_path):
        raise FileNotFoundError(f"Vector output folder not found: {vector_out_folder_path}")
    
    # Combine datasets
    combined_metadata_file, combined_vectors_file = combine_datasets(vector_out_folder_path, run_id)
    
    logging.info(f"Successfully combined datasets:")
    logging.info(f"  - Combined metadata: {combined_metadata_file}")
    logging.info(f"  - Combined vectors: {combined_vectors_file}")
    
    logging.info("END FLOW ******************* Combine Original and Control Datasets *******************")

if __name__ == "__main__":
    # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'model_combine_datasets.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    
    main() 