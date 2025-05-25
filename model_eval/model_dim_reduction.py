import numpy as np
from dotenv import dotenv_values
import os
import sys
#from sklearn.manifold import TSNE
from openTSNE.sklearn import TSNE
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

#@task(log_prints=True)
def compute_pca(vector_list, pca_parameters):
    logging.info("START TASK - compute_pca")
    # Apply PCA for initial dimensionality reduction
    pca = PCA(n_components=pca_parameters["n_components"])
    reduced_vectors = pca.fit_transform(vector_list)
    
    logging.info("END TASK - compute_pca")
    return reduced_vectors

#@task(log_prints=True)
def reduce_with_tsne(vectors, run_id, tsne_parameters):
    logging.info("START TASK - reduce_with_tsne")

    experiment_out_path = tsne_parameters["experiment_out_path"]
    os.makedirs(experiment_out_path, exist_ok=True)
    
    # Apply PCA to do a more efficient and first dimensionality reduction
    pca_n_componenets = tsne_parameters["pca_n_components"]
    pca_parameters = {"n_components" : pca_n_componenets}
    pca_reduced_vectors = compute_pca(vectors, pca_parameters)
    utils.save_vectors_to_tsv(pca_reduced_vectors, run_id, "-vectors_pca-"+str(pca_n_componenets), experiment_out_path)

    # Apply TSNE to do final dimensionality reduction to 2D
    max_iterations = max(len(tsne_parameters[key]) if isinstance(tsne_parameters[key], list) else 0 for key in tsne_parameters.keys())
    for i in range(max_iterations):
        params = {
            "n_components" : get_param_value(tsne_parameters["n_components"], i), 
            "random_state" : get_param_value(tsne_parameters["random_state"], i),
            # method for sklearn TSNE
            "negative_gradient_method" : "bh" if get_param_value(tsne_parameters["method"], i) == "barnes_hut" else "auto", 
            "perplexity" : get_param_value(tsne_parameters["perplexity"], i), 
            "learning_rate" : get_param_value(tsne_parameters["learning_rate"], i),
            # max-iter for sklearn TSNE
            "n_iter" : get_param_value(tsne_parameters["max_iter"], i), 
            "n_jobs" : get_param_value(tsne_parameters["n_jobs"], i)
        }
        
        tsne = TSNE(**params)
        logging.info("START TASK -  TSNE iteration \
                    - perplexity: " + str(params["perplexity"]) 
                    + " - lrate: " + str(params["learning_rate"]) 
                    + " - method: " + str(params["negative_gradient_method"]) 
                    + " - maxiter: " + str(params["n_iter"]) 
                    + " - random_state: " + str(params["random_state"]))
        
        # Apply t-SNE to reduce dimensionality
        reduced_vectors = tsne.fit_transform(pca_reduced_vectors)
        
        out_file_suffix = f"-vectors_tsne-{params['negative_gradient_method']}-{params['perplexity']}-{params['learning_rate']}-{params['n_iter']}-{params['random_state']}"
        
        logging.info("START TASK -  save "+out_file_suffix+".tsv")
        
        utils.save_vectors_to_tsv(reduced_vectors, run_id, out_file_suffix, experiment_out_path)
        
        logging.info("END TASK -  save "+out_file_suffix+".tsv")
        logging.info("END TASK -  TSNE iteration")
    
    logging.info("END TASK - reduce_with_tsne")
    return reduced_vectors

#@flow(name="Reduce embedding dimensions", log_prints=True)
def reduce_embedding_dimensions(vector_out_folder_path, run_id, reduction_parameters, use_combined=False):
    """Reduce the dimensionality of the embedding vectors using t-SNE+PCA or UMAP"""

    # Determine input filename based on whether we're using combined data
    if use_combined:
        vectors_input_filename = run_id + "-combined-vectors_bio.tsv"
        # Update run_id to include combined qualifier for output files
        run_id = run_id + "-combined"
    else:
        vectors_input_filename = run_id + "-vectors_bio.tsv"
    
    vectors_path = os.path.join(vector_out_folder_path, vectors_input_filename)
    
    # Check if the input file exists
    if not os.path.exists(vectors_path):
        if use_combined:
            raise FileNotFoundError(f"Combined vectors file not found: {vectors_path}. Please run model_combine_datasets.py first.")
        else:
            raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
    
    logging.info(f"Loading vectors from: {vectors_path}")
    vectors = np.loadtxt(vectors_path, delimiter='\t', dtype=np.float32)
    logging.info(f"Loaded vectors with shape: {vectors.shape}")

    if reduction_parameters["reduction_method"] == "tsne":
        reduce_with_tsne(vectors, run_id, reduction_parameters)

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
    # Parse arguments
    args = parser.parse_args()

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
        
        # Add combined suffix to folder name if using control flag
        if args.control:
            experiment_folder_name += "-combined"
        
        experiments_out_folder_path = os.path.join(experiments_in_folder_path, experiment_folder_name)
        reduction_parameters["experiment_out_path"] = experiments_out_folder_path
        reduce_embedding_dimensions(vector_out_folder_path, run_id, reduction_parameters, args.control)