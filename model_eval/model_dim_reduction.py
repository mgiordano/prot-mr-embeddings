import numpy as np
import pandas as pd
from dotenv import dotenv_values
import os
import sys
# TSNE imports are now handled dynamically in reduce_with_tsne function
from sklearn.decomposition import PCA
# from prefect import flow, tags, task
import logging
import argparse
import json
from utils.utils import dataset_names, filters, partition_rules
import utils.utils as utils

def get_param_value(param_array, index):
    """Get parameter value handling array length mismatch"""
    if index < len(param_array):
        return param_array[index]
    return param_array[-1]  # Use last element if index out of bounds   

def map_tsne_parameters(tsne_parameters, implementation, iteration_index):
    """Map parameters between different TSNE implementations"""
    
    # Get base parameters
    n_components = get_param_value(tsne_parameters["n_components"], iteration_index)
    random_state = get_param_value(tsne_parameters["random_state"], iteration_index)
    perplexity = get_param_value(tsne_parameters["perplexity"], iteration_index)
    learning_rate = get_param_value(tsne_parameters["learning_rate"], iteration_index)
    max_iter = get_param_value(tsne_parameters["max_iter"], iteration_index)
    method = get_param_value(tsne_parameters["method"], iteration_index)
    exaggeration = get_param_value(tsne_parameters["exaggeration"], iteration_index)
    
    # Handle optional n_jobs parameter - default to max CPU count if not specified
    if "n_jobs" in tsne_parameters and tsne_parameters["n_jobs"] is not None:
        n_jobs = get_param_value(tsne_parameters["n_jobs"], iteration_index)
    else:
        n_jobs = os.cpu_count()  # Use all available CPUs
        logging.info(f"n_jobs not specified in parameters, defaulting to max CPU count: {n_jobs}")
    
    params = {
        "n_components": n_components,
        "random_state": random_state,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "n_jobs": n_jobs,
        "exaggeration": exaggeration,
    }
    if implementation == "openTSNE":
        # openTSNE parameter mapping
        params["negative_gradient_method"] = "fft"
        method_name = params["negative_gradient_method"]
        params["n_iter"] = max_iter
    elif implementation == "sklearn":
        # sklearn.manifold.TSNE parameter mapping
        params["max_iter"] = max_iter
        params["method"] = "barnes_hut" if method == "barnes_hut" else "exact"
        method_name = params["method"]
    else:
        raise ValueError(f"Unsupported TSNE implementation: {implementation}")
    
    return params, method_name

def create_standardized_filename_suffix(params, method_name, implementation):
    """Create standardized filename suffix for both implementations"""
    max_iter_value = params.get('max_iter', params.get('n_iter'))
    return f"-vectors_tsne-{method_name}-{params['perplexity']}-{params['learning_rate']}-{max_iter_value}-{params['random_state']}"

def log_tsne_iteration_start(params, method_name, implementation):
    """Standardized logging for TSNE iteration start"""
    max_iter_key = 'max_iter' if 'max_iter' in params else 'n_iter'
    logging.info(f"START TASK - TSNE iteration ({implementation}) - "
                f"perplexity: {params['perplexity']} - "
                f"learning_rate: {params['learning_rate']} - "
                f"method: {method_name} - "
                f"max_iter: {params[max_iter_key]} - "
                f"random_state: {params['random_state']} - "
                f"n_jobs: {params['n_jobs']}"
                f" - exaggeration: {params['exaggeration']}")

#@task(log_prints=True)
def compute_pca(vector_list, pca_parameters):
    logging.info("START TASK - compute_pca")
    # Apply PCA for initial dimensionality reduction
    pca = PCA(n_components=pca_parameters["n_components"])
    reduced_vectors = pca.fit_transform(vector_list)
    
    logging.info("END TASK - compute_pca")
    return reduced_vectors

#@task(log_prints=True)
def reduce_with_tsne(vectors, run_id, tsne_parameters, implementation="openTSNE"):
    logging.info(f"START TASK - reduce_with_tsne using {implementation}")

    # Import the appropriate TSNE implementation
    if implementation == "openTSNE":
        from openTSNE.sklearn import TSNE
        logging.info("Using openTSNE implementation")
    elif implementation == "sklearn":
        from sklearn.manifold import TSNE
        logging.info("Using sklearn.manifold TSNE implementation")
    else:
        raise ValueError(f"Unsupported TSNE implementation: {implementation}")

    experiment_out_path = tsne_parameters["experiment_out_path"]
    os.makedirs(experiment_out_path, exist_ok=True)

    # Create PCA baseline for tsne comparison
    pca_n_components = tsne_parameters["n_components"][0]
    pca_parameters = {"n_components" : pca_n_components}
    pca_baseline_vectors = compute_pca(vectors, pca_parameters)
    utils.save_vectors_to_tsv(pca_baseline_vectors, run_id, "-vectors_pca-"+str(pca_n_components), experiment_out_path)
    
    # Apply PCA to do a more efficient and first dimensionality reduction
    pca_n_components = tsne_parameters["pca_n_components"]
    pca_parameters = {"n_components" : pca_n_components}
    pca_reduced_vectors = compute_pca(vectors, pca_parameters)
    utils.save_vectors_to_tsv(pca_reduced_vectors, run_id, "-vectors_pca-"+str(pca_n_components), experiment_out_path)

    # Apply TSNE to do final dimensionality reduction to 2D
    max_iterations = max(len(tsne_parameters[key]) if isinstance(tsne_parameters[key], list) else 0 for key in tsne_parameters.keys())
    for i in range(max_iterations):
        # Map parameters for the specific implementation
        params, method_name = map_tsne_parameters(tsne_parameters, implementation, i)
        
        # Create TSNE instance
        tsne = TSNE(**params)
        
        # Log iteration start with standardized format
        log_tsne_iteration_start(params, method_name, implementation)
        
        # Apply t-SNE to reduce dimensionality
        reduced_vectors = tsne.fit_transform(pca_reduced_vectors)
        
        # Create standardized filename suffix
        out_file_suffix = create_standardized_filename_suffix(params, method_name, implementation)
        
        logging.info(f"START TASK - save {out_file_suffix}.tsv")
        
        utils.save_vectors_to_tsv(reduced_vectors, run_id, out_file_suffix, experiment_out_path)
        
        logging.info(f"END TASK - save {out_file_suffix}.tsv")
        logging.info("END TASK - TSNE iteration")
    
    logging.info(f"END TASK - reduce_with_tsne using {implementation}")
    return reduced_vectors

#@flow(name="Reduce embedding dimensions", log_prints=True)
def reduce_embedding_dimensions(vector_out_folder_path, run_id, reduction_parameters,
                                use_combined=False, cross_model_tag=None,
                                filter_col=None):
    """Reduce the dimensionality of the embedding vectors using t-SNE+PCA or UMAP.

    Args:
        use_combined:    Load the combined (original + control) vectors file.
        cross_model_tag: When set, load the cross-embedded vectors produced by
                         model_embeddings_cross.py for the given model tag.
                         Mutually exclusive with use_combined.
        filter_col:      Optional (column_name, value) tuple.  When provided,
                         rows whose *column_name* equals *value* in the
                         corresponding metadata file are **dropped** before
                         computing the dimensionality reduction.  Filtered
                         metadata and vectors are saved alongside the
                         originals.
    """

    # Determine input filename and output run_id qualifier
    if cross_model_tag:
        suffix = f"-cross_{cross_model_tag}"
        vectors_input_filename = run_id + suffix + "-vectors_bio.tsv"
        metadata_input_filename = run_id + suffix + "-metadata.tsv"
        run_id = run_id + suffix
    elif use_combined:
        vectors_input_filename = run_id + "-combined-vectors_bio.tsv"
        metadata_input_filename = run_id + "-combined-metadata.tsv"
        run_id = run_id + "-combined"
    else:
        vectors_input_filename = run_id + "-vectors_bio.tsv"
        metadata_input_filename = run_id + "-metadata.tsv"

    vectors_path = os.path.join(vector_out_folder_path, vectors_input_filename)

    # Check if the input file exists
    if not os.path.exists(vectors_path):
        if cross_model_tag:
            raise FileNotFoundError(
                f"Cross-embedded vectors file not found: {vectors_path}. "
                f"Please run model_embeddings_cross.py first."
            )
        elif use_combined:
            raise FileNotFoundError(
                f"Combined vectors file not found: {vectors_path}. "
                f"Please run model_combine_datasets.py first."
            )
        else:
            raise FileNotFoundError(f"Vectors file not found: {vectors_path}")

    logging.info(f"Loading vectors from: {vectors_path}")
    vectors = np.loadtxt(vectors_path, delimiter='\t', dtype=np.float32)
    logging.info(f"Loaded vectors with shape: {vectors.shape}")

    # ── Optional metadata-based filtering ──────────────────────────────
    if filter_col is not None:
        col_name, col_value = filter_col
        metadata_path = os.path.join(vector_out_folder_path, metadata_input_filename)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file required for --filter-col not found: {metadata_path}"
            )

        metadata_df = pd.read_csv(metadata_path, sep='\t')
        if col_name not in metadata_df.columns:
            raise ValueError(
                f"Column '{col_name}' not found in metadata. "
                f"Available columns: {list(metadata_df.columns)}"
            )

        if len(metadata_df) != vectors.shape[0]:
            raise ValueError(
                f"Metadata rows ({len(metadata_df)}) and vector rows "
                f"({vectors.shape[0]}) mismatch."
            )

        # Build mask: keep rows that do NOT match the column value
        keep_mask = metadata_df[col_name].astype(str) != str(col_value)
        n_dropped = int((~keep_mask).sum())
        logging.info(
            f"filter-col: dropping {n_dropped} / {len(metadata_df)} rows "
            f"where '{col_name}' == '{col_value}'"
        )

        # Filter vectors and metadata
        vectors = vectors[keep_mask.values]
        metadata_df = metadata_df[keep_mask].reset_index(drop=True)
        logging.info(f"Vectors after filtering: {vectors.shape}")

        # Persist filtered artefacts so downstream tools can use them
        filtered_vectors_path = os.path.join(
            vector_out_folder_path, run_id + "-filtered-vectors_bio.tsv"
        )
        np.savetxt(filtered_vectors_path, vectors, delimiter='\t', fmt='%.20f')
        logging.info(f"Saved filtered vectors to: {filtered_vectors_path}")

        filtered_metadata_path = os.path.join(
            vector_out_folder_path, run_id + "-filtered-metadata.tsv"
        )
        metadata_df.to_csv(filtered_metadata_path, sep='\t', index=False)
        logging.info(f"Saved filtered metadata to: {filtered_metadata_path}")

        # Tag the run_id so output files don't overwrite unfiltered results
        run_id = run_id + "-filtered"

    if reduction_parameters["reduction_method"] == "tsne":
        # Get TSNE implementation from experiment parameters, default to sklearn
        tsne_implementation = reduction_parameters.get("tsne_implementation", "sklearn")
        logging.info(f"Using TSNE implementation from experiment config: {tsne_implementation}")
        reduce_with_tsne(vectors, run_id, reduction_parameters, tsne_implementation)

# run the flow!
if __name__=="__main__":
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    config = dotenv_values(dotenv_path)
    
   # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir,'model_dim_reduction.log'),
        level=logging.INFO,  # Adjust log level as needed
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    parser = argparse.ArgumentParser(description='Compute dimensionality reduction on protein embeddings')
    parser.add_argument('timestamp',
                        help='Run timestamp')
    parser.add_argument('dataset_name', 
                        help='Input protein dataset name')
    parser.add_argument('filter',
                        help='MR Filter')
    parser.add_argument('partition_rule',
                        help='MR partition rule')
    parser.add_argument('experiment_file',
                    help='Experiment .json file with reduction parameters')
    parser.add_argument('--control',
                        action='store_true',
                        help='Use combined dataset (original + control) for dimensionality reduction')
    parser.add_argument('--cross',
                        metavar='MODEL_TAG',
                        default=None,
                        help=(
                            'Use cross-embedded vectors produced by model_embeddings_cross.py. '
                            'Provide the compact model tag used during embedding '
                            '(e.g. "bsc"). The script will look for '
                            '<run_id>-cross_<MODEL_TAG>-vectors_bio.tsv and tag all '
                            'output files with the same suffix.'
                        ))
    parser.add_argument('--filter-col',
                        nargs=2,
                        metavar=('COLUMN', 'VALUE'),
                        default=None,
                        help=(
                            'Drop vectors whose metadata COLUMN equals VALUE '
                            'before computing dimensionality reduction.  '
                            'Example: --filter-col partition_type mr'
                        ))

    # Parse arguments
    args = parser.parse_args()

    if args.control and args.cross:
        parser.error("--control and --cross are mutually exclusive.")

    # input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    # set run data to work on
    family_dataset_name = getattr(dataset_names, args.dataset_name)
    timestamp = args.timestamp
    filter_name = getattr(filters, args.filter).name
    partition_rule_name = getattr(partition_rules, args.partition_rule)["name"]
    reduction_parameters_file_name = sys.argv[5]

    # create output structure
    date = utils.get_date_from_formatted_ts(timestamp)
    vector_out_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    run_id = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name
    experiments_in_folder_path = os.path.join(vector_out_folder_path, "experiments")
    experiments_file_path = os.path.join(experiments_in_folder_path, reduction_parameters_file_name)

    with open(experiments_file_path, 'r') as file:
        reduction_parameters = json.load(file)
    #with tags(reduction_parameters["reduction_method"]):
        # Create experiment subfolder name based on JSON filename
        experiment_folder_name = reduction_parameters_file_name[:reduction_parameters_file_name.rindex('.')]

        # Tag folder name to match the vector variant being reduced
        if args.cross:
            experiment_folder_name += "-cross"
        elif args.control:
            experiment_folder_name += "-combined"
        if args.filter_col:
            experiment_folder_name += "-filtered"

        experiments_out_folder_path = os.path.join(experiments_in_folder_path, experiment_folder_name)
        reduction_parameters["experiment_out_path"] = experiments_out_folder_path

        # Build filter_col tuple if flag was provided
        filter_col = tuple(args.filter_col) if args.filter_col else None

        reduce_embedding_dimensions(
            vector_out_folder_path, run_id, reduction_parameters,
            use_combined=args.control,
            cross_model_tag=args.cross,
            filter_col=filter_col,
        )