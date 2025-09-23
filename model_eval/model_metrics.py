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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import concurrent.futures
import threading
from functools import partial

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
    Calculate silhouette scores for each class using sklearn's efficient implementation
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        class_labels: array-like of class labels for each sample
    
    Returns:
        dict: class_name -> silhouette score (-1 to 1, higher is better)
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
        
        # Keep original silhouette range [-1, 1]
        class_scores[class_name] = avg_silhouette
    
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

# Thread-safe CSV writer for incremental results
_csv_write_lock = threading.Lock()

def append_to_csv_threadsafe(results_list, output_path, write_headers=False):
    """
    Thread-safe function to append results to CSV file
    
    Args:
        results_list: List of result dictionaries to write
        output_path: Path to output CSV file
        write_headers: Whether to write headers (first time writing)
    """
    if not results_list:
        return
    
    # Convert to DataFrame for easier CSV writing
    df = pd.DataFrame(results_list)
    
    with _csv_write_lock:
        # Check if file exists and whether we need headers
        file_exists = os.path.exists(output_path)
        write_header = write_headers and not file_exists
        
        # Append to CSV file
        df.to_csv(output_path, mode='a', index=False, header=write_header)
        
        logging.info(f"Appended {len(results_list)} results to {output_path}")

def process_single_vector_file(file_info):
    """
    Process a single vector file for cluster metrics analysis
    
    Args:
        file_info: Dictionary containing file processing information
    
    Returns:
        list: Results for this file
    """
    try:
        file_path = file_info['file_path']
        file_type = file_info['file_type']  # 'bio', 'pca', or 'tsne'
        class_labels = file_info['class_labels']
        unique_classes = file_info['unique_classes']
        run_id = file_info['run_id']
        
        logging.info(f"Processing {file_type} file: {os.path.basename(file_path)}")
        start_time = time.time()
        
        # Load vector data
        vectors = load_vector_data(file_path)
        
        # Calculate silhouette scores and centroids
        silhouette_scores = calculate_class_silhouette_scores(vectors, class_labels)
        centroids = calculate_class_centroids(vectors, class_labels, unique_classes)
        centroid_strings = centroids_to_string_array(centroids)
        
        # Create results for each class
        results = []
        
        if file_type == 'bio':
            # Bio vectors - original space
            for i, class_name in enumerate(unique_classes):
                result_row = {
                    'filename': os.path.basename(file_path),
                    'timestamp': run_id.split('-')[0],
                    'dataset': run_id.split('-')[1],
                    'filter': run_id.split('-')[2],
                    'partition': run_id.split('-')[3],
                    'technique': 'none',
                    'method': '',
                    'perplexity': np.nan,
                    'learning_rate': '',
                    'max_iterations': np.nan,
                    'random_state': np.nan,
                    'num_dimensions': vectors.shape[1],
                    'class_name': class_name,
                    'centroid': centroid_strings[i],
                    'silhouette_score': silhouette_scores[class_name],
                    'num_class_points': np.sum(class_labels == class_name)
                }
                results.append(result_row)
                
        elif file_type == 'pca':
            # PCA vectors
            for i, class_name in enumerate(unique_classes):
                result_row = {
                    'filename': os.path.basename(file_path),
                    'timestamp': run_id.split('-')[0],
                    'dataset': run_id.split('-')[1],
                    'filter': run_id.split('-')[2],
                    'partition': run_id.split('-')[3],
                    'technique': 'pca',
                    'method': '',
                    'perplexity': np.nan,
                    'learning_rate': '',
                    'max_iterations': np.nan,
                    'random_state': np.nan,
                    'num_dimensions': vectors.shape[1],
                    'class_name': class_name,
                    'centroid': centroid_strings[i],
                    'silhouette_score': silhouette_scores[class_name],
                    'num_class_points': np.sum(class_labels == class_name)
                }
                results.append(result_row)
                
        elif file_type == 'tsne':
            # t-SNE vectors - parse parameters from filename
            params = parse_filename_parameters(file_path)
            
            for i, class_name in enumerate(unique_classes):
                result_row = {
                    'filename': os.path.basename(file_path),
                    'timestamp': params['timestamp'],
                    'dataset': params['dataset'],
                    'filter': params['filter'],
                    'partition': params['partition'],
                    'technique': 'tsne',
                    'method': params['method'],
                    'perplexity': int(params['perplexity']),
                    'learning_rate': params['learning_rate'],
                    'max_iterations': int(params['max_iterations']),
                    'random_state': int(params['random_state']),
                    'num_dimensions': vectors.shape[1],
                    'class_name': class_name,
                    'centroid': centroid_strings[i],
                    'silhouette_score': silhouette_scores[class_name],
                    'num_class_points': np.sum(class_labels == class_name)
                }
                results.append(result_row)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Completed {file_type} file {os.path.basename(file_path)} in {elapsed_time:.2f}s")
        
        return results
        
    except Exception as e:
        logging.error(f"Error processing file {file_info.get('file_path', 'unknown')}: {str(e)}")
        return []

def analyze_cluster_metrics_parallel(experiment_folder_path, vector_output_folder_path, run_id, metadata_file_path, class_label_column, use_combined=False, max_workers=None):
    """
    Analyze cluster-level metrics using parallel processing and incremental CSV writing
    
    Args:
        experiment_folder_path: Path to experiment folder
        vector_output_folder_path: Path to vector output folder
        run_id: Run identifier
        metadata_file_path: Path to metadata file
        class_label_column: Column name for class labels
        use_combined: Whether using combined dataset
        max_workers: Maximum number of parallel workers (None = auto)
    
    Returns:
        str: Path to output CSV file
    """
    logging.info("START TASK - analyze_cluster_metrics_parallel")
    
    # Load metadata once (shared across all workers)
    metadata_df = load_metadata(metadata_file_path)
    class_labels = metadata_df[class_label_column].values
    unique_classes = np.unique(class_labels)
    
    # Prepare output file path
    metrics_folder = os.path.join(experiment_folder_path, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)
    cluster_output_path = os.path.join(metrics_folder, "cluster_metrics.csv")
    
    # Remove existing output file to start fresh
    if os.path.exists(cluster_output_path):
        os.remove(cluster_output_path)
    
    # Collect all files to process
    files_to_process = []
    
    # 1. Bio vectors file
    if use_combined:
        bio_vectors_filename = f"{run_id}-combined-vectors_bio.tsv"
    else:
        bio_vectors_filename = f"{run_id}-vectors_bio.tsv"
    
    bio_vectors_path = os.path.join(vector_output_folder_path, bio_vectors_filename)
    if os.path.exists(bio_vectors_path):
        files_to_process.append({
            'file_path': bio_vectors_path,
            'file_type': 'bio',
            'class_labels': class_labels,
            'unique_classes': unique_classes,
            'run_id': run_id
        })
    else:
        logging.warning(f"Bio vectors file not found: {bio_vectors_path}")
    
    # 2. PCA files
    tsv_pattern = os.path.join(experiment_folder_path, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    pca_files = [f for f in tsv_files if 'vectors_pca' in os.path.basename(f)]
    
    for pca_file in pca_files:
        files_to_process.append({
            'file_path': pca_file,
            'file_type': 'pca',
            'class_labels': class_labels,
            'unique_classes': unique_classes,
            'run_id': run_id
        })
    
    # 3. t-SNE files
    tsne_files = [f for f in tsv_files if 'vectors_tsne' in os.path.basename(f)]
    
    for tsne_file in tsne_files:
        files_to_process.append({
            'file_path': tsne_file,
            'file_type': 'tsne',
            'class_labels': class_labels,
            'unique_classes': unique_classes,
            'run_id': run_id
        })
    
    logging.info(f"Found {len(files_to_process)} files to process ({len(pca_files)} PCA, {len(tsne_files)} t-SNE, {1 if bio_vectors_path and os.path.exists(bio_vectors_path) else 0} bio)")
    
    if not files_to_process:
        logging.warning("No vector files found to process")
        return cluster_output_path
    
    # Set default max_workers based on available cores and file count
    if max_workers is None:
        import multiprocessing
        max_workers = min(len(files_to_process), multiprocessing.cpu_count())
    
    logging.info(f"Using {max_workers} parallel workers")
    
    # Process files in parallel using ProcessPoolExecutor
    # (better than ThreadPoolExecutor for CPU-bound numpy operations due to GIL)
    total_results_written = 0
    headers_written = False
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(process_single_vector_file, file_info): file_info 
                         for file_info in files_to_process}
        
        # Process completed jobs and write results incrementally
        for future in concurrent.futures.as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                results = future.result()
                if results:
                    # Write results immediately to CSV
                    append_to_csv_threadsafe(results, cluster_output_path, write_headers=not headers_written)
                    headers_written = True
                    total_results_written += len(results)
                    logging.info(f"Completed processing {file_info['file_type']} file: {os.path.basename(file_info['file_path'])}")
                else:
                    logging.warning(f"No results from {file_info['file_type']} file: {os.path.basename(file_info['file_path'])}")
                    
            except Exception as e:
                logging.error(f"Error processing {file_info['file_type']} file {os.path.basename(file_info['file_path'])}: {str(e)}")
    
    logging.info(f"END TASK - analyze_cluster_metrics_parallel: {total_results_written} total results written to {cluster_output_path}")
    
    return cluster_output_path

def analyze_cluster_metrics(experiment_folder_path, vector_output_folder_path, run_id, metadata_file_path, class_label_column, use_combined=False, max_workers=None):
    """
    Backward compatibility wrapper for analyze_cluster_metrics_parallel
    Returns a DataFrame like the original function for compatibility
    """
    logging.info("Using parallel cluster metrics analysis for better performance")
    
    # Call the parallel version
    csv_path = analyze_cluster_metrics_parallel(
        experiment_folder_path, vector_output_folder_path, run_id, 
        metadata_file_path, class_label_column, use_combined, max_workers
    )
    
    # Read the CSV back into a DataFrame for compatibility
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        
        if not results_df.empty:
            # Sort by technique, dimensions, class name, and experiment parameters
            results_df = results_df.sort_values(['technique', 'num_dimensions', 'class_name', 'perplexity', 'learning_rate'])
        
        return results_df
    else:
        return pd.DataFrame()

def calculate_scree_data(high_dim_data):
    """
    Calculate explained variance for each PCA component (scree plot data)
    
    Args:
        high_dim_data: numpy array of shape (n_samples, n_features)
    
    Returns:
        list: List of dictionaries with component analysis data
    """
    logging.info("START TASK - calculate_scree_data")
    start_time = time.time()
    
    # Step 1: Standardize the data (center to mean 0, scale to unit variance)
    logging.info("Standardizing high-dimensional data")
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(high_dim_data)
    
    # Step 2: Perform PCA to get the explained variance for each component
    logging.info(f"Performing PCA on {high_dim_data.shape[0]} samples with {high_dim_data.shape[1]} features")
    pca = PCA()
    pca.fit(standardized_data)
    
    # Get explained variance for each component
    explained_variances = pca.explained_variance_
    
    # Step 3: Calculate total variance to find percentages
    total_variance = np.sum(explained_variances)
    
    # Step 4: Store the output data in a structured format
    logging.info("Computing component statistics")
    scree_data_table = []
    cumulative_variance = 0
    
    for i in range(len(explained_variances)):
        variance = explained_variances[i]
        percentage_variance = (variance / total_variance) * 100
        cumulative_variance += percentage_variance
        
        record = {
            "component_number": i + 1,
            "explained_variance": variance,
            "percentage_variance": percentage_variance,
            "cumulative_percentage_variance": cumulative_variance
        }
        scree_data_table.append(record)
    
    elapsed_time = time.time() - start_time
    logging.info(f"END TASK - calculate_scree_data: computed {len(explained_variances)} components in {elapsed_time:.2f}s")
    
    return scree_data_table

def analyze_scree_data(vector_output_folder_path, run_id, use_combined=False):
    """
    Analyze PCA explained variance for the high-dimensional bio vectors
    """
    logging.info("START TASK - analyze_scree_data")
    
    # Determine input filename based on combined flag
    if use_combined:
        vectors_input_filename = run_id + "-combined-vectors_bio.tsv"
    else:
        vectors_input_filename = run_id + "-vectors_bio.tsv"
    
    vectors_path = os.path.join(vector_output_folder_path, vectors_input_filename)
    
    # Check if the input file exists
    if not os.path.exists(vectors_path):
        if use_combined:
            raise FileNotFoundError(f"Combined bio vectors file not found: {vectors_path}. Please run model_combine_datasets.py first.")
        else:
            raise FileNotFoundError(f"Bio vectors file not found: {vectors_path}")
    
    logging.info(f"Loading bio vectors from: {vectors_path}")
    
    # Load high-dimensional bio vectors
    bio_vectors = np.loadtxt(vectors_path, delimiter='\t', dtype=np.float32)
    logging.info(f"Loaded bio vectors with shape: {bio_vectors.shape}")
    
    # Calculate scree data
    scree_data = calculate_scree_data(bio_vectors)
    
    # Convert to DataFrame
    scree_df = pd.DataFrame(scree_data)
    
    logging.info("END TASK - analyze_scree_data")
    return scree_df

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

def analyze_tsne_metrics_parallel(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined=False, max_workers=None):
    """
    Parallel version of analyze_tsne_metrics using ProcessPoolExecutor for better performance
    """
    logging.info("START TASK - analyze_tsne_metrics_parallel")
    
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
    
    # Pre-compute shared data for all experiments (optimization)
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
    
    # Prepare data packages for parallel processing
    file_packages = []
    for tsne_file in tsne_files:
        file_packages.append({
            'tsne_file': tsne_file,
            'high_dim_data': high_dim_data,
            'high_dim_neighbors': high_dim_neighbors,
            'high_dim_distances': high_dim_distances,
            'sample_indices': sample_indices,
            'class_labels': class_labels,
            'unique_classes': unique_classes,
            'high_dim_class_neighbors': high_dim_class_neighbors,
            'knc_k': knc_k
        })
    
    # Set default max_workers based on available cores and file count
    if max_workers is None:
        import multiprocessing
        max_workers = min(len(tsne_files), multiprocessing.cpu_count())
    
    logging.info(f"Using {max_workers} parallel workers for t-SNE metrics")
    
    results = []
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(process_single_tsne_file, package): package 
                         for package in file_packages}
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_file):
            package = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logging.info(f"Completed t-SNE metrics for: {os.path.basename(package['tsne_file'])}")
                else:
                    logging.warning(f"No results from: {os.path.basename(package['tsne_file'])}")
                    
            except Exception as e:
                logging.error(f"Error processing {os.path.basename(package['tsne_file'])}: {str(e)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Sort by perplexity and learning rate for better organization
        results_df = results_df.sort_values(['perplexity', 'learning_rate', 'max_iterations'])
    
    logging.info("END TASK - analyze_tsne_metrics_parallel")
    return results_df

def process_single_tsne_file(package):
    """
    Process a single t-SNE file for metrics analysis
    
    Args:
        package: Dictionary containing all necessary data for processing
    
    Returns:
        dict: Result row for this t-SNE file
    """
    try:
        tsne_file = package['tsne_file']
        high_dim_data = package['high_dim_data']
        high_dim_neighbors = package['high_dim_neighbors']
        high_dim_distances = package['high_dim_distances']
        sample_indices = package['sample_indices']
        class_labels = package['class_labels']
        unique_classes = package['unique_classes']
        high_dim_class_neighbors = package['high_dim_class_neighbors']
        knc_k = package['knc_k']
        
        logging.info(f"Processing t-SNE file: {os.path.basename(tsne_file)}")
        
        # Parse parameters from filename
        params = parse_filename_parameters(tsne_file)
        
        # Load t-SNE vectors (2D)
        low_dim_data = load_vector_data(tsne_file)
        
        # Check data consistency
        if high_dim_data.shape[0] != low_dim_data.shape[0]:
            logging.error(f"Dimension mismatch: high_dim={high_dim_data.shape[0]}, low_dim={low_dim_data.shape[0]}")
            return None
        
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
        
        return result_row
        
    except Exception as e:
        logging.error(f"Error processing t-SNE file {package.get('tsne_file', 'unknown')}: {str(e)}")
        return None

def analyze_tsne_metrics(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined=False):
    """
    Backward compatibility wrapper for analyze_tsne_metrics_parallel
    """
    logging.info("Using parallel t-SNE metrics analysis for better performance")
    return analyze_tsne_metrics_parallel(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined)

def create_metrics_for_experiment(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined=False, max_workers=None):
    """Create metrics analysis for all TSV files in the experiment folder"""
    logging.info("START TASK - create_metrics_for_experiment")
    
    # Create metrics subfolder inside the experiment folder
    metrics_folder = os.path.join(experiment_folder_path, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Analyze t-SNE metrics
    metrics_df = analyze_tsne_metrics_parallel(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined, max_workers)
    
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
            # Get vector output folder path
            vector_output_folder_path = os.path.dirname(experiment_folder_path).replace('/experiments', '')
            
            cluster_df = analyze_cluster_metrics(
                experiment_folder_path, vector_output_folder_path, run_id, metadata_file_path, class_label_column, use_combined, max_workers
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
    
    # Analyze scree plot data (PCA explained variance) and save to separate CSV
    logging.info("Starting scree plot analysis (PCA explained variance)")
    
    try:
        # Get the vector output folder path to access the bio vectors file
        date = utils.get_date_from_formatted_ts(run_id.split('-')[0])  # Extract timestamp from run_id
        family_dataset_name = run_id.split('-')[1]  # Extract dataset name from run_id
        vector_output_folder_path = os.path.join(os.path.dirname(experiment_folder_path).replace('/experiments', ''))
        
        scree_df = analyze_scree_data(vector_output_folder_path, run_id, use_combined)
        
        if not scree_df.empty:
            scree_filename = "scree_data_table.csv"
            scree_output_path = os.path.join(metrics_folder, scree_filename)
            scree_df.to_csv(scree_output_path, index=False)
            logging.info(f"Scree data saved to: {scree_output_path}")
            
            # Log some key statistics
            total_components = len(scree_df)
            variance_90_components = len(scree_df[scree_df['cumulative_percentage_variance'] <= 90.0])
            variance_95_components = len(scree_df[scree_df['cumulative_percentage_variance'] <= 95.0])
            variance_99_components = len(scree_df[scree_df['cumulative_percentage_variance'] <= 99.0])
            
            logging.info(f"PCA Analysis Summary:")
            logging.info(f"Total components: {total_components}")
            logging.info(f"Components for 90% variance: {variance_90_components}")
            logging.info(f"Components for 95% variance: {variance_95_components}")
            logging.info(f"Components for 99% variance: {variance_99_components}")
        else:
            logging.warning("No scree data computed")
            
    except Exception as e:
        logging.error(f"Error during scree analysis: {str(e)}")
    
    # Log summary statistics
    logging.info(f"Computed metrics for {len(metrics_df)} t-SNE configurations")
    if 'knn_preservation_k10' in metrics_df.columns:
        logging.info(f"KNN Preservation (k=10) - Mean: {summary_stats.get('knn_mean', 0):.4f}, Std: {summary_stats.get('knn_std', 0):.4f}, Min: {summary_stats.get('knn_min', 0):.4f}, Max: {summary_stats.get('knn_max', 0):.4f}")
    
    if 'cpd_correlation' in metrics_df.columns:
        logging.info(f"CPD Correlation - Mean: {summary_stats.get('cpd_mean', 0):.4f}, Std: {summary_stats.get('cpd_std', 0):.4f}, Min: {summary_stats.get('cpd_min', 0):.4f}, Max: {summary_stats.get('cpd_max', 0):.4f}")
    
    if 'knc_preservation_k10' in metrics_df.columns:
        logging.info(f"KNC Preservation (k=10) - Mean: {summary_stats.get('knc_mean', 0):.4f}, Std: {summary_stats.get('knc_std', 0):.4f}, Min: {summary_stats.get('knc_min', 0):.4f}, Max: {summary_stats.get('knc_max', 0):.4f}")
    
    logging.info("END TASK - create_metrics_for_experiment")

def create_tsne_metrics(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, experiment_name, class_label_column, use_combined=False, max_workers=None):
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
    create_metrics_for_experiment(experiment_folder_path, run_id, metadata_file_path, class_label_column, use_combined, max_workers)
    
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
    parser.add_argument('--max-workers', 
                        type=int,
                        default=None,
                        help='Maximum number of parallel workers for cluster analysis (default: auto-detect based on CPU cores)')
    
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
        args.control,
        args.max_workers
    ) 