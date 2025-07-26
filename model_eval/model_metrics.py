import os
import sys
import argparse
import logging
import glob
import pandas as pd
import numpy as np
from dotenv import dotenv_values
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples
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

def calculate_cpd(high_dim_distances, low_dim_data, sample_indices):
    """
    Calculate Correlation of Pairwise Distances (CPD) metric using pre-computed high-dimensional distances
    
    This metric evaluates how well the global structure is preserved.
    It compares pairwise distances in high-dimensional space with those in 
    the t-SNE embedding using Spearman correlation.
    
    Args:
        high_dim_distances: pre-computed pairwise distances from high-dimensional space
        low_dim_data: numpy array of shape (n_samples, 2) - t-SNE embedding
        sample_indices: indices of the sampled points
    
    Returns:
        float: Spearman correlation coefficient (-1 to 1, higher is better)
    """
    logging.info("START TASK - calculate_cpd")
    start_time = time.time()
    
    # Create subsampled low-dimensional data
    low_dim_sample = low_dim_data[sample_indices]
    
    logging.info(f"Computing low-dimensional pairwise distances for {len(sample_indices)} sampled points")
    
    # Calculate pairwise distances for low-dimensional data only
    # (high-dimensional distances are pre-computed)
    low_dim_distances = pdist(low_dim_sample, metric='euclidean')
    
    # Calculate Spearman correlation between distance arrays
    logging.info("Computing Spearman correlation")
    correlation, p_value = spearmanr(high_dim_distances, low_dim_distances)
    
    elapsed_time = time.time() - start_time
    logging.info(f"END TASK - calculate_cpd: {correlation:.4f} (p-value: {p_value:.6f}, computed in {elapsed_time:.2f}s)")
    
    return correlation

def calculate_class_centroids(data, class_labels, unique_classes):
    """
    Calculate centroid (mean position) for each class
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        class_labels: array-like of class labels for each sample
        unique_classes: list of unique class names
    
    Returns:
        numpy array of shape (n_classes, n_features) with class centroids
    """
    centroids = []
    for class_name in unique_classes:
        class_mask = class_labels == class_name
        class_data = data[class_mask]
        centroid = np.mean(class_data, axis=0)
        centroids.append(centroid)
    
    return np.array(centroids)

def calculate_knc_preservation(high_dim_class_neighbors, low_dim_data, class_labels, unique_classes, k=10):
    """
    Calculate K-Nearest Classes (KNC) preservation metric using pre-computed high-dimensional class neighbors
    
    This metric evaluates how well the mesoscopic (class-level) structure is preserved.
    It compares the k nearest class centroids in high-dimensional space with those 
    in the t-SNE embedding.
    
    Args:
        high_dim_class_neighbors: pre-computed neighbor indices for high-dimensional class centroids
        low_dim_data: numpy array of shape (n_samples, 2) - t-SNE embedding
        class_labels: array-like of class labels for each sample
        unique_classes: array of unique class names
        k: number of nearest class centroids to consider (default=10)
    
    Returns:
        float: KNC preservation score (0 to 1, higher is better)
    """
    logging.info(f"START TASK - calculate_knc_preservation with k={k}")
    start_time = time.time()
    
    num_classes = len(unique_classes)
    
    if k >= num_classes:
        logging.warning(f"k={k} is >= number of classes ({num_classes}). Setting k to {num_classes-1}")
        k = num_classes - 1
    
    if k <= 0:
        logging.warning(f"Not enough classes for KNC analysis (need at least 2 classes, found {num_classes})")
        return np.nan
    
    # Calculate class centroids in low-dimensional space only (high-dim already computed)
    logging.info("Computing class centroids in low-dimensional space")
    low_dim_centroids = calculate_class_centroids(low_dim_data, class_labels, unique_classes)
    
    # Find k nearest centroids for each class in low-dimensional space
    logging.info("Finding nearest class centroids in low-dimensional space")
    low_dim_class_neighbors = find_k_nearest_neighbors(low_dim_centroids, k)
    
    # Calculate preservation for each class
    logging.info("Calculating class neighborhood preservation")
    total_preservation = 0
    
    for i in range(num_classes):
        # Get neighbor sets for class i
        high_neighbors_set = set(high_dim_class_neighbors[i])
        low_neighbors_set = set(low_dim_class_neighbors[i])
        
        # Find intersection
        preserved_neighbors = high_neighbors_set.intersection(low_neighbors_set)
        num_preserved = len(preserved_neighbors)
        
        # Add fraction of preserved neighbors
        total_preservation += num_preserved / k
    
    # Calculate average preservation across all classes
    knc_preservation = total_preservation / num_classes
    
    elapsed_time = time.time() - start_time
    logging.info(f"END TASK - calculate_knc_preservation: {knc_preservation:.4f} (computed in {elapsed_time:.2f}s)")
    
    return knc_preservation

def calculate_class_silhouette_scores(data, class_labels):
    """
    Calculate normalized silhouette scores for each class using sklearn's efficient implementation
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        class_labels: array-like of class labels for each sample
    
    Returns:
        dict: class_name -> normalized silhouette score (0 to 1, higher is better)
    """
    logging.info("START TASK - calculate_class_silhouette_scores")
    start_time = time.time()
    
    # Use sklearn's efficient silhouette implementation
    silhouette_scores = silhouette_samples(data, class_labels, metric='euclidean')
    
    # Calculate average silhouette score for each class
    unique_classes = np.unique(class_labels)
    class_scores = {}
    
    for class_name in unique_classes:
        class_mask = class_labels == class_name
        class_silhouette_scores = silhouette_scores[class_mask]
        avg_silhouette = np.mean(class_silhouette_scores)
        
        # Normalize from [-1, 1] to [0, 1]
        normalized_score = (avg_silhouette + 1) / 2
        class_scores[class_name] = normalized_score
    
    elapsed_time = time.time() - start_time
    logging.info(f"END TASK - calculate_class_silhouette_scores: computed for {len(unique_classes)} classes in {elapsed_time:.2f}s")
    
    return class_scores

def centroids_to_string_array(centroids):
    """
    Convert centroid coordinates to string representation for CSV storage
    
    Args:
        centroids: numpy array of shape (n_classes, n_features)
    
    Returns:
        dict: class_index -> string representation of centroid
    """
    centroid_strings = {}
    for i, centroid in enumerate(centroids):
        # Convert to string with reasonable precision
        centroid_str = '[' + ','.join([f'{x:.6f}' for x in centroid]) + ']'
        centroid_strings[i] = centroid_str
    
    return centroid_strings

def analyze_cluster_metrics(experiment_folder_path, run_id, metadata_file_path, class_label_column, 
                          high_dim_data, class_labels, unique_classes, high_dim_centroids, use_combined=False):
    """
    Analyze cluster-level metrics for all t-SNE files in the experiment folder
    """
    logging.info("START TASK - analyze_cluster_metrics")
    
    # Find all t-SNE TSV files in experiment folder
    tsv_pattern = os.path.join(experiment_folder_path, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    
    # Filter to get only t-SNE vector files
    tsne_files = [f for f in tsv_files if 'vectors_tsne' in os.path.basename(f)]
    
    if not tsne_files:
        logging.warning(f"No t-SNE vector files found in {experiment_folder_path}")
        return pd.DataFrame()
    
    logging.info(f"Found {len(tsne_files)} t-SNE vector files for cluster analysis")
    
    # Calculate high-dimensional silhouette scores ONCE (optimization)
    logging.info("Computing high-dimensional silhouette scores once for all experiments")
    high_dim_silhouette_scores = calculate_class_silhouette_scores(high_dim_data, class_labels)
    
    # Convert high-dimensional centroids to string format ONCE (optimization)
    high_dim_centroid_strings = centroids_to_string_array(high_dim_centroids)
    
    results = []
    
    for tsne_file in tsne_files:
        logging.info(f"Processing cluster metrics for t-SNE file: {os.path.basename(tsne_file)}")
        
        try:
            # Parse parameters from filename
            params = parse_filename_parameters(tsne_file)
            
            # Load t-SNE vectors (2D)
            low_dim_data = load_vector_data(tsne_file)
            
            # Check data consistency
            if high_dim_data.shape[0] != low_dim_data.shape[0]:
                logging.error(f"Dimension mismatch: high_dim={high_dim_data.shape[0]}, low_dim={low_dim_data.shape[0]}")
                continue
            
            # Calculate low-dimensional silhouette scores
            low_dim_silhouette_scores = calculate_class_silhouette_scores(low_dim_data, class_labels)
            
            # Calculate low-dimensional centroids
            low_dim_centroids = calculate_class_centroids(low_dim_data, class_labels, unique_classes)
            low_dim_centroid_strings = centroids_to_string_array(low_dim_centroids)
            
            # Create results for each class
            for i, class_name in enumerate(unique_classes):
                result_row = {
                    'filename': os.path.basename(tsne_file),
                    'timestamp': params['timestamp'],
                    'dataset': params['dataset'],
                    'filter': params['filter'],
                    'partition': params['partition'],
                    'vectors_method': params['vectors_method'],
                    'method': params['method'],
                    'perplexity': int(params['perplexity']),
                    'learning_rate': params['learning_rate'],
                    'max_iterations': int(params['max_iterations']),
                    'random_state': int(params['random_state']),
                    'class_name': class_name,
                    'high_dim_centroid': high_dim_centroid_strings[i],
                    'low_dim_centroid': low_dim_centroid_strings[i],
                    'high_dim_silhouette': high_dim_silhouette_scores[class_name],
                    'low_dim_silhouette': low_dim_silhouette_scores[class_name],
                    'silhouette_preservation': low_dim_silhouette_scores[class_name] / high_dim_silhouette_scores[class_name] if high_dim_silhouette_scores[class_name] > 0 else np.nan,
                    'num_class_points': np.sum(class_labels == class_name)
                }
                
                results.append(result_row)
                
        except Exception as e:
            logging.error(f"Error processing cluster metrics for {tsne_file}: {str(e)}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Sort by class name and experiment parameters for better organization
        results_df = results_df.sort_values(['class_name', 'perplexity', 'learning_rate', 'max_iterations'])
    
    logging.info("END TASK - analyze_cluster_metrics")
    return results_df

def load_vector_data(vector_file_path):
    """Load vector data from TSV file"""
    logging.info(f"Loading vector data from: {vector_file_path}")
    vectors = np.loadtxt(vector_file_path, delimiter='\t', dtype=np.float32)
    logging.info(f"Loaded vectors with shape: {vectors.shape}")
    return vectors

def load_metadata(metadata_file_path):
    """Load metadata from TSV file"""
    logging.info(f"Loading metadata from: {metadata_file_path}")
    metadata_df = pd.read_csv(metadata_file_path, sep='\t')
    logging.info(f"Loaded metadata with shape: {metadata_df.shape}")
    logging.info(f"Metadata columns: {list(metadata_df.columns)}")
    return metadata_df

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

def analyze_tsne_metrics(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined=False):
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
    
    # Load metadata once (it's the same for all t-SNE runs)
    try:
        metadata_df = load_metadata(metadata_file_path)
        if class_label_column not in metadata_df.columns:
            logging.error(f"Class label column '{class_label_column}' not found in metadata. Available columns: {list(metadata_df.columns)}")
            return pd.DataFrame()
        class_labels = metadata_df[class_label_column].values
        logging.info(f"Using '{class_label_column}' as class labels")
    except FileNotFoundError as e:
        logging.error(f"Could not load metadata: {str(e)}")
        return pd.DataFrame()
    
    # Calculate high-dimensional neighbors ONCE for all experiments (optimization)
    logging.info("Computing high-dimensional neighbors once for all experiments")
    high_dim_neighbors = find_k_nearest_neighbors(high_dim_data, k=10)
    
    # Calculate sampling and high-dimensional distances ONCE for CPD metric (optimization)
    num_points = high_dim_data.shape[0]
    sample_size = min(1000, num_points)  # Use smaller sample if dataset is small
    if sample_size < 1000:
        logging.warning(f"Dataset has only {num_points} points, using sample_size={sample_size}")
    
    # Set random seed for reproducible sampling across all experiments
    np.random.seed(42)
    sample_indices = np.random.choice(num_points, size=sample_size, replace=False)
    
    logging.info(f"Computing high-dimensional pairwise distances once for {sample_size} sampled points")
    high_dim_sample = high_dim_data[sample_indices]
    high_dim_distances = pdist(high_dim_sample, metric='euclidean')
    
    # Calculate class-related data ONCE for KNC metric (optimization)
    unique_classes = np.unique(class_labels)
    num_classes = len(unique_classes)
    logging.info(f"Found {num_classes} unique classes for KNC analysis")
    
    # Compute high-dimensional class centroids and neighbors once
    if num_classes > 1:  # Only compute if we have multiple classes
        logging.info("Computing high-dimensional class centroids once for all experiments")
        high_dim_centroids = calculate_class_centroids(high_dim_data, class_labels, unique_classes)
        
        knc_k = min(10, num_classes - 1)  # Adjust k for available classes
        if knc_k > 0:
            logging.info("Computing high-dimensional class neighbors once for all experiments")
            high_dim_class_neighbors = find_k_nearest_neighbors(high_dim_centroids, knc_k)
        else:
            high_dim_class_neighbors = None
    else:
        high_dim_class_neighbors = None
        knc_k = 0
    
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
            
            # Calculate CPD metric (global structure preservation) using pre-computed distances
            cpd_correlation = calculate_cpd(high_dim_distances, low_dim_data, sample_indices)
            
            # Calculate KNC metric (class-level structure preservation) using pre-computed data
            if high_dim_class_neighbors is not None and knc_k > 0:
                knc_preservation = calculate_knc_preservation(high_dim_class_neighbors, low_dim_data, class_labels, unique_classes, k=knc_k)
            else:
                knc_preservation = np.nan  # Not enough classes for KNC analysis
            
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
                'cpd_correlation': cpd_correlation,
                'knc_preservation_k10': knc_preservation,
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

def create_metrics_for_experiment(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined=False):
    """Create metrics analysis for all TSV files in the experiment folder"""
    logging.info("START TASK - create_metrics_for_experiment")
    
    # Create metrics subfolder inside the experiment folder
    metrics_folder = os.path.join(experiment_folder_path, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Analyze t-SNE metrics
    metrics_df = analyze_tsne_metrics(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined)
    
    if metrics_df.empty:
        logging.warning("No metrics computed - no valid t-SNE files found")
        return
    
    # Calculate and add summary statistics to the CSV BEFORE saving
    summary_stats = {}
    
    # Calculate summary statistics for each metric
    if 'knn_preservation_k10' in metrics_df.columns:
        knn_stats = metrics_df['knn_preservation_k10'].describe()
        summary_stats.update({
            'knn_mean': knn_stats['mean'],
            'knn_std': knn_stats['std'],
            'knn_min': knn_stats['min'],
            'knn_max': knn_stats['max'],
            'knn_25th_percentile': knn_stats['25%'],
            'knn_50th_percentile': knn_stats['50%'],
            'knn_75th_percentile': knn_stats['75%']
        })
    
    if 'cpd_correlation' in metrics_df.columns:
        cpd_stats = metrics_df['cpd_correlation'].describe()
        summary_stats.update({
            'cpd_mean': cpd_stats['mean'],
            'cpd_std': cpd_stats['std'],
            'cpd_min': cpd_stats['min'],
            'cpd_max': cpd_stats['max'],
            'cpd_25th_percentile': cpd_stats['25%'],
            'cpd_50th_percentile': cpd_stats['50%'],
            'cpd_75th_percentile': cpd_stats['75%']
        })
    
    if 'knc_preservation_k10' in metrics_df.columns:
        knc_stats = metrics_df['knc_preservation_k10'].describe()
        summary_stats.update({
            'knc_mean': knc_stats['mean'],
            'knc_std': knc_stats['std'],
            'knc_min': knc_stats['min'],
            'knc_max': knc_stats['max'],
            'knc_25th_percentile': knc_stats['25%'],
            'knc_50th_percentile': knc_stats['50%'],
            'knc_75th_percentile': knc_stats['75%']
        })
    
    # Add summary statistics as new columns to each row
    for col, value in summary_stats.items():
        metrics_df[col] = value
    
    # Save metrics to CSV (now includes summary statistics)
    metrics_filename = "tsne_metrics.csv"
    metrics_output_path = os.path.join(metrics_folder, metrics_filename)
    
    metrics_df.to_csv(metrics_output_path, index=False)
    logging.info(f"Metrics saved to: {metrics_output_path}")
    
    # Analyze cluster-level metrics and save to separate CSV
    logging.info("Starting cluster-level silhouette analysis")
    
    # Load the data we need for cluster analysis (reuse computed data from analyze_tsne_metrics)
    # We need to reload some data since it's not passed back from analyze_tsne_metrics
    try:
        high_dim_data = load_high_dim_data(experiment_folder_path, run_id, use_combined)
        metadata_df = load_metadata(metadata_file_path)
        class_labels = metadata_df[class_label_column].values
        unique_classes = np.unique(class_labels)
        
        if len(unique_classes) > 1:
            high_dim_centroids = calculate_class_centroids(high_dim_data, class_labels, unique_classes)
            
            cluster_df = analyze_cluster_metrics(
                experiment_folder_path, run_id, metadata_file_path, class_label_column,
                high_dim_data, class_labels, unique_classes, high_dim_centroids, use_combined
            )
            
            if not cluster_df.empty:
                cluster_filename = "cluster_metrics.csv"
                cluster_output_path = os.path.join(metrics_folder, cluster_filename)
                cluster_df.to_csv(cluster_output_path, index=False)
                logging.info(f"Cluster metrics saved to: {cluster_output_path}")
                logging.info(f"Computed cluster metrics for {len(unique_classes)} classes across {len(cluster_df) // len(unique_classes)} t-SNE configurations")
            else:
                logging.warning("No cluster metrics computed")
        else:
            logging.warning("Skipping cluster analysis - need at least 2 classes")
            
    except Exception as e:
        logging.error(f"Error during cluster analysis: {str(e)}")
    
    # Log summary statistics
    logging.info(f"Computed metrics for {len(metrics_df)} t-SNE configurations")
    if 'knn_preservation_k10' in metrics_df.columns:
        logging.info(f"KNN Preservation (k=10) - Mean: {summary_stats.get('knn_mean', 0):.4f}, Std: {summary_stats.get('knn_std', 0):.4f}, Min: {summary_stats.get('knn_min', 0):.4f}, Max: {summary_stats.get('knn_max', 0):.4f}")
    
    if 'cpd_correlation' in metrics_df.columns:
        logging.info(f"CPD Correlation - Mean: {summary_stats.get('cpd_mean', 0):.4f}, Std: {summary_stats.get('cpd_std', 0):.4f}, Min: {summary_stats.get('cpd_min', 0):.4f}, Max: {summary_stats.get('cpd_max', 0):.4f}")
    
    if 'knc_preservation_k10' in metrics_df.columns:
        logging.info(f"KNC Preservation (k=10) - Mean: {summary_stats.get('knc_mean', 0):.4f}, Std: {summary_stats.get('knc_std', 0):.4f}, Min: {summary_stats.get('knc_min', 0):.4f}, Max: {summary_stats.get('knc_max', 0):.4f}")
    
    logging.info("END TASK - create_metrics_for_experiment")

def create_tsne_metrics(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, experiment_name, class_label_column, use_combined=False):
    """Main flow to create metrics analysis for experiment results"""
    logging.info("START FLOW ******************* Create t-SNE Metrics *******************")
    
    # Create output structure
    date = utils.get_date_from_formatted_ts(timestamp)
    vector_output_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    
    # Determine run ID and metadata file path
    run_id = timestamp + "-" + family_dataset_name + "-" + filter_name + "-" + partition_rule_name
    
    if use_combined:
        run_id_for_metadata = run_id + "-combined"
        metadata_filename = run_id_for_metadata + "-metadata.tsv"
        experiment_folder_name = experiment_name + "-combined"
    else:
        metadata_filename = run_id + "-metadata.tsv"
        experiment_folder_name = experiment_name
    
    metadata_file_path = os.path.join(vector_output_folder_path, metadata_filename)
    
    # Check if metadata file exists
    if not os.path.exists(metadata_file_path):
        if use_combined:
            raise FileNotFoundError(f"Combined metadata file not found: {metadata_file_path}. Please run model_combine_datasets.py first.")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
    
    # Locate experiment folder
    experiments_folder_path = os.path.join(vector_output_folder_path, "experiments")
    experiment_folder_path = os.path.join(experiments_folder_path, experiment_folder_name)
    
    if not os.path.exists(experiment_folder_path):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_folder_path}")
    
    logging.info(f"Processing experiment folder: {experiment_folder_path}")
    logging.info(f"Using metadata file: {metadata_file_path}")
    logging.info(f"Using class label column: {class_label_column}")
    
    # Create metrics for all vector files in the experiment
    create_metrics_for_experiment(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined)
    
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
    parser.add_argument('class_label_column',
                        help='Metadata column name to use as class labels (e.g., sequence_family_name, sequence_family_type)')
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
    class_label_column = args.class_label_column
    
    # Create metrics analysis
    create_tsne_metrics(
        input_data_root_path, 
        family_dataset_name, 
        timestamp, 
        filter_name, 
        partition_rule_name, 
        experiment_name,
        class_label_column,
        args.control
    ) 