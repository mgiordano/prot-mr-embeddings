import tensorflow as tf
import numpy as np
from tensorboard.plugins import projector
import json
import os
import sys
import argparse
from dotenv import dotenv_values
from utils import utils
from utils.utils import dataset_names, filters, partition_rules
import glob

def setup_tensorboard_projector(vectors_path, metadata_path, log_dir, max_points=10000):
    """
    Set up TensorBoard projector with existing vectors.tsv and metadata.tsv files.
    
    Args:
        vectors_path: Path to vectors.tsv containing embeddings
        metadata_path: Path to metadata.tsv containing labels
        log_dir: Directory to save TensorBoard logs
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Load the embeddings from vectors.tsv
    embeddings = np.loadtxt(vectors_path, delimiter='\t', dtype=np.float32)
    
    # Convert embeddings to tensor
    embedding_var = tf.Variable(embeddings, name='embeddings')
    
    # Create checkpoint
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    
    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = os.path.abspath(metadata_path)
    
    # Save the config
    projector.visualize_embeddings(log_dir, config)

    # Create custom TensorBoard configuration to increase point limit
    tb_config = {
        "tensorboard-plugin-projector": {
            "webGLPointLimit": max_points
        }
    }
    
    # Save custom config
    with open(os.path.join(log_dir, 'plugin_config.json'), 'w') as f:
        json.dump(tb_config, f)
    
    print(f"Setup complete! To view projections, run:")
    print(f"tensorboard --logdir={log_dir}  --bind_all")
    return log_dir

# Example usage
if __name__ == "__main__":
    # travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Setup TensorBoard projector for protein embeddings')
    parser.add_argument('timestamp',
                        help='Run timestamp')
    parser.add_argument('dataset_name', 
                        help='Input protein dataset name')
    parser.add_argument('filter',
                        help='MR Filter')
    parser.add_argument('partition_rule',
                        help='MR partition rule')
    parser.add_argument('reduce_method',
                        help='Reduction method (pca|tsne)')
    parser.add_argument('experiment_name',
                        help='Experiment folder name (without .json extension)')
    parser.add_argument('--control', 
                        action='store_true',
                        help='Use combined dataset (original + control) for TensorBoard projection')
    
    args = parser.parse_args()

    # input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    # set run data to work on
    reduce_method = args.reduce_method
    family_dataset_name = getattr(dataset_names, args.dataset_name)
    timestamp = args.timestamp
    filter_name = getattr(filters, args.filter).name
    partition_rule_name = getattr(partition_rules, args.partition_rule)["name"]
    experiment_name = args.experiment_name

    date = utils.get_date_from_formatted_ts(timestamp)
    vector_output_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    
    # Determine run ID and experiment folder name based on whether we're using combined data
    run_id = timestamp + "-" + family_dataset_name + "-" + filter_name + "-" + partition_rule_name
    
    if args.control:
        run_id_for_files = run_id + "-combined"
        experiment_folder_name = experiment_name + "-combined"
    else:
        run_id_for_files = run_id
        experiment_folder_name = experiment_name
    
    # Locate experiment folder (same logic as model_visualization.py)
    experiments_folder_path = os.path.join(vector_output_folder_path, "experiments")
    experiment_folder_path = os.path.join(experiments_folder_path, experiment_folder_name)
    
    if not os.path.exists(experiment_folder_path):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_folder_path}")
    
    # Find the vectors file in the experiment folder (handles parameter suffixes like -50)
    vectors_pattern = os.path.join(experiment_folder_path, f"{run_id_for_files}-vectors_{reduce_method}*.tsv")
    vectors_files = glob.glob(vectors_pattern)
    
    if not vectors_files:
        if args.control:
            raise FileNotFoundError(f"No combined {reduce_method} vectors files found in {experiment_folder_path}. Please run model_dim_reduction.py with --control flag first.")
        else:
            raise FileNotFoundError(f"No {reduce_method} vectors files found in {experiment_folder_path}. Please run model_dim_reduction.py first.")
    
    # Use the first matching vectors file
    vectors_file = vectors_files[0]
    
    # Metadata file is in the parent vector_output folder
    metadata_file = os.path.join(vector_output_folder_path, run_id_for_files + "-metadata.tsv")
    
    # Check if metadata file exists
    if not os.path.exists(metadata_file):
        if args.control:
            raise FileNotFoundError(f"Combined metadata file not found: {metadata_file}. Please run model_combine_datasets.py first.")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    print(f"Using vectors file: {vectors_file}")
    print(f"Using metadata file: {metadata_file}")
    
    log_path = os.path.join(vector_output_folder_path, "logs", "projection")
    log_dir = setup_tensorboard_projector(vectors_file, metadata_file, log_path, 700000)