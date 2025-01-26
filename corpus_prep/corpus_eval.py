import sys
import os
from dotenv import dotenv_values
import logging
# from prefect import flow, tags, task
from gensim.models.fasttext import FastText
from gensim.utils import tokenize
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from corpus_prep_utils import dataset_names, filters, partition_rules, bioword_rules
import corpus_prep_utils

VECTOR_SIZE = 100

# #######################################
#      SUB FLOW 1                       #
# #######################################

def save_vectors_to_tsv(vectors, filename_prefix, filename_suffix, parent_folder_path):
    logging.info("START TASK -  save "+filename_suffix+".tsv")
    # Save reduced vectors as intermediate .tsv result
    filename = filename_prefix + filename_suffix + ".tsv"
    out_path = os.path.join(parent_folder_path, filename)
    np.savetxt(out_path, vectors, delimiter='\t', fmt='%.20f')
    logging.info("END TASK -  save "+filename_suffix+".tsv")

#@task(log_prints=True)
def compute_tsne(vectors, tsne_parameters):
    logging.info("START TASK - compute_tsne")
    # Initialize t-SNE
    tsne = TSNE(n_components=tsne_parameters["n_components"], random_state=tsne_parameters["random_state"], 
                method=tsne_parameters["method"], perplexity=tsne_parameters["perplexity"], learning_rate=tsne_parameters["learning_rate"],
                max_iter=tsne_parameters["max_iter"])

    # Apply t-SNE to reduce dimensionality
    reduced_vectors = tsne.fit_transform(vectors)
    logging.info("END TASK - compute_tsne")
    return reduced_vectors
    

#@task(log_prints=True)
def compute_pca(vector_list, pca_parameters):
    logging.info("START TASK - compute_pca")
    # Apply PCA for initial dimensionality reduction
    pca = PCA(n_components=pca_parameters["n_components"])
    reduced_vectors = pca.fit_transform(vector_list)
    
    logging.info("END TASK - compute_pca")
    return reduced_vectors

def get_vector_for_bioword_partition(bioword_partition, model):
    """Estimates an embedding vector for sequence based on its bioword representation"""
    
    # Create an empty vector the size of the trained FastText model
    vector = np.zeros(VECTOR_SIZE, dtype=np.float64)

    # Tokenize the bioword partition into a list of biowords
    bioword_partition = list(tokenize(bioword_partition))
    
    # Retrieve the embedding for each word and accumulate in main vector
    for i in range(0, len(bioword_partition)):
        vector += model.wv[bioword_partition[i]]
    
    # Take the mean of all bioword vectors
    vector = vector / len(bioword_partition)
    return vector

#@task(log_prints=True)
def compute_sequence_vectors(corpus_df, model, bioword_rule_column="word_partition"):
    logging.info("START TASK - compute_sequence_vectors")
    corpus_df["bio_vector"] = corpus_df[bioword_rule_column].astype(str).apply(get_vector_for_bioword_partition, args=(model,))
    logging.info("END TASK - compute_sequence_vectors")
    return corpus_df

#@task(log_prints=True)
def load_corpus_eval_to_df(corpus_path: str):
    logging.info("START TASK - load_corpus_eval_to_df")
    df = pd.read_csv(corpus_path, encoding='utf-8')
    logging.info("END TASK - load_corpus_eval_to_df")
    return df

#@task(log_prints=True)
def get_corpus_eval_file_path_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    logging.info("START TASK - get_corpus_eval_file_path_from_run")
    corpus_file_iterator = corpus_prep_utils.get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, False)
    corpus_path = corpus_prep_utils.create_or_load_joined_corpus_file(corpus_file_iterator)
    logging.info("END TASK - get_corpus_eval_file_path_from_run")
    return corpus_path

#@task(log_prints=True)
def load_model(model_path: str):
    logging.info("START TASK - load_model")
    model = FastText.load(model_path, mmap='r')
    logging.info("END TASK - load_model")
    return model

# #######################################
#      MAIN FLOW                        #
# #######################################

#@flow(name="Evaluate BioWord Protein Model", log_prints=True)
def eval_corpus(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, bioword_rule_column, pca_parameters, tsne_parameters):
    logging.info("START FLOW ******************* Evaluate BioWord Protein Model *******************")

    # Load FastText trained model
    model_path = corpus_prep_utils.get_model_path_by_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    model = load_model(model_path)

    # Load evaluation corpus export (bioword partition with seq info and labels)
    corpus_path = get_corpus_eval_file_path_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    corpus_df = load_corpus_eval_to_df(corpus_path)

    # SUB FLOW 1: Compute bioword vectors
    # based on model embeddings and apply
    # dimensionality reduction techniques to obtain
    # n_dim representation (usually 2D)
    
    # Compute vector representation for sequence bioword representation
    corpus_df = compute_sequence_vectors(corpus_df, model, bioword_rule_column)

    # free up memory
    del model
    corpus_df.drop('word_partition', axis=1, inplace=True)

    logging.info("START TASK -  save metadata.tsv")
    # save metadata tsv
    metadata_df = corpus_df[["sequence", "sequence_family_name", "sequence_family_type"]]
    metadata_filename = filename_prefix + "-metadata.tsv"
    metadata_out_path = os.path.join(parent_folder_path, metadata_filename)
    metadata_df.to_csv(metadata_out_path, sep='\t', index=False)
    logging.info("END TASK -  save metadata.tsv")

    logging.info("START TASK -  save vectors_bio.tsv")
    # Create biovectors dataframe by unnesting the vector column
    biovector_list = corpus_df["bio_vector"].tolist()
    biovectors_df = pd.DataFrame(
        biovector_list,
        columns=[f'dim_{i}' for i in range(len(corpus_df["bio_vector"].iloc[0]))]
    )
    biovector_filename = filename_prefix + "-vectors_bio.tsv"
    biovectors_out_path = os.path.join(parent_folder_path, biovector_filename)
    biovectors_df.to_csv(biovectors_out_path, sep='\t', header=False, index=False, float_format='%.20f')
    del biovectors_df
    logging.info("END TASK -  save vectors_bio.tsv")

    # Apply PCA to do a more efficient and first dimensionality reduction
    pca_reduced_vectors = compute_pca(biovector_list, pca_parameters)

    save_vectors_to_tsv(pca_reduced_vectors, filename_prefix, "-vectors_pca", parent_folder_path)
    
    # Apply TSNE to do final dimensionality reduction to 2D
    tsne_reduced_vectors = compute_tsne(pca_reduced_vectors, tsne_parameters)

    save_vectors_to_tsv(tsne_reduced_vectors, filename_prefix, "-vectors_tsne", parent_folder_path)

    corpus_df[['reduced_vector_d1', 'reduced_vector_d2']] = tsne_reduced_vectors

    # save df
    df_out_file_suffix = f"-s4_corpus_bio_vector_pca_{pca_parameters['n_components']}_tsne_{tsne_parameters['perplexity']}_{tsne_parameters['learning_rate']}.csv"
    df_out_filename = filename_prefix + df_out_file_suffix
    df_out_path = os.path.join(parent_folder_path, df_out_filename)
    corpus_df.to_csv(df_out_path, float_format='%.20f', index=False)
    del corpus_df
    
    logging.info("END FLOW ******************* Evaluate BioWord Protein Model *******************")

# #######################################
#      MAIN                             #
# #######################################

# run the flow!
if __name__=="__main__":
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # arguments
    arguments = len(sys.argv) - 1
    if(arguments!=4):
        print("Usage: python corpus_eval.py <family_dataset_name> <timestamp> <filter_name> <partition_rule_name>") 
        quit()

    # Configure logging
    logging.basicConfig(
        filename='corpus_eval.log',
        level=logging.INFO,  # Adjust log level as needed
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    # input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    # set run data to work on
    family_dataset_name = sys.argv[1]
    timestamp = sys.argv[2]
    filter_name = sys.argv[3]
    partition_rule_name = sys.argv[4]

    # create output structure
    date = corpus_prep_utils.get_date_from_formatted_ts(timestamp)
    parent_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    os.makedirs(parent_folder_path, exist_ok=True)
    filename_prefix = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name

    # model evaluation parameters
    bioword_rule_column = bioword_rules.BIOWORD_RULE_PARTITION_COLUMN
    pca_parameters = {
        "n_components" : int(VECTOR_SIZE / 2) # 50
    }
    tsne_parameters = {
        "n_components" : 2,
        "random_state" : 42,
        "method" : "exact",
        "perplexity" : 2,
        "learning_rate" : 10,
        "max_iter" : 5000 
    }
    
    # with tags("eval"):
    eval_corpus(input_data_root_path, family_dataset_name, timestamp, 
                filter_name, partition_rule_name, bioword_rule_column,
                pca_parameters, tsne_parameters)