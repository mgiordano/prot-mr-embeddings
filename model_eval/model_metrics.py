import os
import sys
import argparse
import logging
import glob
import pandas as pd
import numpy as np
from dotenv import dotenv_values
from sklearn.neighbors import NearestNeighbors
import time

from utils.utils import dataset_names, filters, partition_rules
import utils.utils as utils

def parse_filename_parameters(filename):
    """
    Parse experiment parameters from filename
    """
    # Remove file extension and get the base filename
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Split parameters by '-'
    params = base_filename.split('-')
    
    # Extract timestamp and parameters
    timestamp = params[0]
    dataset = params[1]
    filter_type = params[2].replace('filter_', '')
    partition = params[3].replace('partition_', '')
    vectors_method = params[4].replace('vectors_', '')
    method = params[5]
    perplexity = params[6]
    learning_rate = params[7]
    max_iterations = params[8]
    random_state = params[9]
    
    return {
        'timestamp': timestamp,
        'dataset': dataset,
        'filter': filter_type,
        'partition': partition,
        'vectors_method': vectors_method,
        'method': method,
        'perplexity': perplexity,
        'learning_rate': learning_rate,
        'max_iterations': max_iterations,
        'random_state': random_state
    }

def find_k_nearest_neighbors(data, k):
    """
    Find k nearest neighbors for all points in the dataset
    Returns indices of neighbors for each point
    """
    logging.info(f"Computing {k} nearest neighbors for {data.shape[0]} points in {data.shape[1]}D space")
    
    # Use sklearn's NearestNeighbors for efficient computation
    # n_neighbors = k + 1 because the first neighbor is the point itself
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
    nbrs.fit(data)
    
    # Find neighbors (including self as first neighbor)
    distances, indices = nbrs.kneighbors(data)
    
    # Remove self from neighbors (first column)
    neighbor_indices = indices[:, 1:]
    
    return neighbor_indices

def calculate_knn_preservation(high_dim_neighbors, low_dim_data, k=10):
    """
    Calculate K-Nearest Neighbors preservation metric using pre-computed high-dimensional neighbors
    
    This metric evaluates how well the local structure is preserved.
    For each point, it compares the k nearest neighbors in high-dimensional 
    space with the k nearest neighbors in the t-SNE embedding.
    
    Args:
        high_dim_neighbors: numpy array of shape (n_samples, k) with neighbor indices
        low_dim_data: numpy array of shape (n_samples, 2) - t-SNE embedding
        k: number of nearest neighbors to consider (default=10)
    
    Returns:
        float: KNN preservation score (0 to 1, higher is better)
    """
    logging.info(f"START TASK - calculate_knn_preservation with k={k}")
    start_time = time.time()
    
    num_points = low_dim_data.shape[0]
    
    if k >= num_points:
        logging.warning(f"k={k} is >= number of points ({num_points}). Setting k to {num_points-1}")
        k = num_points - 1
    
    # Find neighbors in low-dimensional space only (high-dim already computed)
    logging.info("Finding neighbors in low-dimensional space")
    low_dim_neighbors = find_k_nearest_neighbors(low_dim_data, k)
    
    # Calculate preservation for each point
    logging.info("Calculating neighborhood preservation")
    total_preservation = 0
    
    for i in range(num_points):
        # Get neighbor sets for point i
        high_neighbors_set = set(high_dim_neighbors[i])
        low_neighbors_set = set(low_dim_neighbors[i])
        
        # Find intersection
        preserved_neighbors = high_neighbors_set.intersection(low_neighbors_set)
        num_preserved = len(preserved_neighbors)
        
        # Add fraction of preserved neighbors
        total_preservation += num_preserved / k
    
    # Calculate average preservation
    knn_preservation = total_preservation / num_points
    
    elapsed_time = time.time() - start_time
    logging.info(f"END TASK - calculate_knn_preservation: {knn_preservation:.4f} (computed in {elapsed_time:.2f}s)")
    
    return knn_preservation

def load_vector_data(vector_file_path):
    """Load vector data from TSV file"""
    logging.info(f"Loading vector data from: {vector_file_path}")
    vectors = np.loadtxt(vector_file_path, delimiter='\t', dtype=np.float32)
    logging.info(f"Loaded vectors with shape: {vectors.shape}")
    return vectors

def load_high_dim_data(experiment_folder_path, run_id, use_combined=False):
    """
    Load the high-dimensional data (PCA-reduced vectors) that was used as input to t-SNE
    """
    # Determine the input filename based on combined flag
    if use_combined:
        high_dim_filename = f"{run_id}-combined-vectors_pca-50.tsv"
    else:
        high_dim_filename = f"{run_id}-vectors_pca-50.tsv"
    
    high_dim_path = os.path.join(experiment_folder_path, high_dim_filename)
    
    if not os.path.exists(high_dim_path):
        # Try alternative naming pattern
        if use_combined:
            alt_filename = f"{run_id}-vectors_pca-50.tsv"
        else:
            alt_filename = f"{run_id}-combined-vectors_pca-50.tsv"
        alt_path = os.path.join(experiment_folder_path, alt_filename)
        
        if os.path.exists(alt_path):
            high_dim_path = alt_path
        else:
            raise FileNotFoundError(f"High-dimensional data file not found: {high_dim_path}")
    
    return load_vector_data(high_dim_path)

def analyze_tsne_metrics(experiment_folder_path, run_id, use_combined=False):
    """
    Analyze metrics for all t-SNE files in the experiment folder
    """
    logging.info("START TASK - analyze_tsne_metrics")
    
    # Find all t-SNE TSV files in experiment folder
    tsv_pattern = os.path.join(experiment_folder_path, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    
    # Filter to get only t-SNE vector files
    tsne_files = [f for f in tsv_files if 'vectors_tsne' in os.path.basename(f)]
    
    if not tsne_files:
        logging.warning(f"No t-SNE vector files found in {experiment_folder_path}")
        return pd.DataFrame()
    
    logging.info(f"Found {len(tsne_files)} t-SNE vector files to analyze")
    
    # Load high-dimensional data once (it's the same for all t-SNE runs)
    try:
        high_dim_data = load_high_dim_data(experiment_folder_path, run_id, use_combined)
    except FileNotFoundError as e:
        logging.error(f"Could not load high-dimensional data: {str(e)}")
        return pd.DataFrame()
    
    # Calculate high-dimensional neighbors ONCE for all experiments (optimization)
    logging.info("Computing high-dimensional neighbors once for all experiments")
    high_dim_neighbors = find_k_nearest_neighbors(high_dim_data, k=10)
    
    results = []
    
    for tsne_file in tsne_files:
        logging.info(f"Processing t-SNE file: {os.path.basename(tsne_file)}")
        
        try:
            # Parse parameters from filename
            params = parse_filename_parameters(tsne_file)
            
            # Load t-SNE vectors (2D)
            low_dim_data = load_vector_data(tsne_file)
            
            # Check data consistency
            if high_dim_data.shape[0] != low_dim_data.shape[0]:
                logging.error(f"Dimension mismatch: high_dim={high_dim_data.shape[0]}, low_dim={low_dim_data.shape[0]}")
                continue
            
            # Calculate KNN preservation metric (using pre-computed high-dim neighbors)
            knn_preservation = calculate_knn_preservation(high_dim_neighbors, low_dim_data, k=10)
            
            # Create result row
            result_row = {
                'filename': os.path.basename(tsne_file),
                'timestamp': params['timestamp'],
                'dataset': params['dataset'],
                'filter': params['filter'],
                'partition': params['partition'],
                'vectors_method': params['vectors_method'],
                'method': params['method'],
                'perplexity': int(params['perplexity']),
                'learning_rate': params['learning_rate'],  # Keep as string to handle 'auto'
                'max_iterations': int(params['max_iterations']),
                'random_state': int(params['random_state']),
                'knn_preservation_k10': knn_preservation,
                'num_points': high_dim_data.shape[0],
                'high_dim_features': high_dim_data.shape[1],
                'low_dim_features': low_dim_data.shape[1]
            }
            
            results.append(result_row)
            
        except Exception as e:
            logging.error(f"Error processing {tsne_file}: {str(e)}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Sort by perplexity and learning rate for better organization
        results_df = results_df.sort_values(['perplexity', 'learning_rate', 'max_iterations'])
    
    logging.info("END TASK - analyze_tsne_metrics")
    return results_df

def create_metrics_for_experiment(experiment_folder_path, run_id, use_combined=False):
    """Create metrics analysis for all TSV files in the experiment folder"""
    logging.info("START TASK - create_metrics_for_experiment")
    
    # Create metrics subfolder inside the experiment folder
    metrics_folder = os.path.join(experiment_folder_path, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Analyze t-SNE metrics
    metrics_df = analyze_tsne_metrics(experiment_folder_path, run_id, use_combined)
    
    if metrics_df.empty:
        logging.warning("No metrics computed - no valid t-SNE files found")
        return
    
    # Save metrics to CSV
    metrics_filename = "tsne_metrics.csv"
    metrics_output_path = os.path.join(metrics_folder, metrics_filename)
    
    metrics_df.to_csv(metrics_output_path, index=False)
    logging.info(f"Metrics saved to: {metrics_output_path}")
    
    # Log summary statistics
    logging.info(f"Computed metrics for {len(metrics_df)} t-SNE configurations")
    if 'knn_preservation_k10' in metrics_df.columns:
        knn_mean = metrics_df['knn_preservation_k10'].mean()
        knn_std = metrics_df['knn_preservation_k10'].std()
        knn_min = metrics_df['knn_preservation_k10'].min()
        knn_max = metrics_df['knn_preservation_k10'].max()
        logging.info(f"KNN Preservation (k=10) - Mean: {knn_mean:.4f}, Std: {knn_std:.4f}, Min: {knn_min:.4f}, Max: {knn_max:.4f}")
    
    logging.info("END TASK - create_metrics_for_experiment")

def create_tsne_metrics(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, experiment_name, use_combined=False):
    """Main flow to create metrics analysis for experiment results"""
    logging.info("START FLOW ******************* Create t-SNE Metrics *******************")
    
    # Create output structure
    date = utils.get_date_from_formatted_ts(timestamp)
    vector_output_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    
    # Determine run ID
    run_id = timestamp + "-" + family_dataset_name + "-" + filter_name + "-" + partition_rule_name
    
    if use_combined:
        experiment_folder_name = experiment_name + "-combined"
    else:
        experiment_folder_name = experiment_name
    
    # Locate experiment folder
    experiments_folder_path = os.path.join(vector_output_folder_path, "experiments")
    experiment_folder_path = os.path.join(experiments_folder_path, experiment_folder_name)
    
    if not os.path.exists(experiment_folder_path):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_folder_path}")
    
    logging.info(f"Processing experiment folder: {experiment_folder_path}")
    
    # Create metrics for all vector files in the experiment
    create_metrics_for_experiment(experiment_folder_path, run_id, use_combined)
    
    logging.info("END FLOW ******************* Create t-SNE Metrics *******************")

# #######################################
#      MAIN                             #
# #######################################

if __name__ == "__main__":
    # Travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'model_metrics.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    
    parser = argparse.ArgumentParser(description='Create metrics analysis from experiment t-SNE results')
    parser.add_argument('timestamp',
                        help='Run timestamp')
    parser.add_argument('dataset_name', 
                        help='Input protein dataset name')
    parser.add_argument('filter',
                        help='MR Filter')
    parser.add_argument('partition_rule',
                        help='MR partition rule')
    parser.add_argument('experiment_name',
                        help='Experiment folder name (without .json extension)')
    parser.add_argument('--control', 
                        action='store_true',
                        help='Use combined dataset (original + control) for metrics analysis')
    
    # Parse arguments
    args = parser.parse_args()

    # Input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    
    # Set run data to work on
    family_dataset_name = getattr(dataset_names, args.dataset_name)
    timestamp = args.timestamp
    filter_name = getattr(filters, args.filter).name
    partition_rule_name = getattr(partition_rules, args.partition_rule)["name"]
    experiment_name = args.experiment_name
    
    # Create metrics analysis
    create_tsne_metrics(
        input_data_root_path, 
        family_dataset_name, 
        timestamp, 
        filter_name, 
        partition_rule_name, 
        experiment_name, 
        args.control
    ) 