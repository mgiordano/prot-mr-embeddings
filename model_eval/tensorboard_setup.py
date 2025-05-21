import tensorflow as tf
import numpy as np
from tensorboard.plugins import projector
import json
import os
import sys
from dotenv import dotenv_values
from utils import utils

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
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # arguments
    arguments = len(sys.argv) - 1
    if(arguments!=5):
        print("Usage: python model_eval.tensorboard_setup <pca|tnse> <family_dataset_name> <timestamp> <filter_name> <partition_rule_name>") 
        quit()

    # input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    # set run data to work on
    reduce_method = sys.argv[1]
    family_dataset_name = sys.argv[2]
    timestamp = sys.argv[3]
    filter_name = sys.argv[4]
    partition_rule_name = sys.argv[5]

    date = utils.get_date_from_formatted_ts(timestamp)
    parent_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    filename_prefix = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name
    vectors_file_suffix = f"-vectors_{reduce_method}.tsv"
    vectors_file = os.path.join(parent_folder_path, filename_prefix+vectors_file_suffix)  # Path to your vectors.tsv
    metadata_file = os.path.join(parent_folder_path, filename_prefix+"-metadata.tsv")  # Path to your metadata.tsv
    log_path = os.path.join(parent_folder_path, "logs", "projection")
    log_dir = setup_tensorboard_projector(vectors_file, metadata_file, log_path, 700000)