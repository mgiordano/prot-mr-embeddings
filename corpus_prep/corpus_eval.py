import os
from dotenv import dotenv_values
from prefect import flow, tags, task
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
#      SUB FLOW 2                       #
# #######################################

# #######################################
#      SUB FLOW 1                       #
# #######################################

def compute_tsne(vectors, tsne_parameters):
    # Initialize t-SNE
    tsne = TSNE(n_components=tsne_parameters["n_components"], random_state=tsne_parameters["random_state"], 
                method=tsne_parameters["method"], perplexity=tsne_parameters["perplexity"], 
                max_iter=tsne_parameters["max_iter"])

    # Apply t-SNE to reduce dimensionality
    reduced_vectors = tsne.fit_transform(vectors)
    return reduced_vectors
    

@task(log_prints=True)
def compute_pca(corpus_df, pca_parameters):
    # Apply PCA for initial dimensionality reduction
    pca = PCA(n_components=pca_parameters["n_componenets"])
    reduced_vectors = pca.fit_transform(corpus_df['bio_vector'].values.tolist())
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

@task(log_prints=True)
def compute_sequence_vectors(corpus_df, model, bioword_rule_column="word_partition"):
    corpus_df["bio_vector"] = corpus_df[bioword_rule_column].astype(str).apply(get_vector_for_bioword_partition, args=(model,))
    return corpus_df

@task(log_prints=True)
def load_corpus_eval_to_df(corpus_path: str):
    return pd.read_csv(corpus_path, encoding='utf-8')

@task(log_prints=True)
def get_corpus_eval_file_path_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    corpus_file_iterator = corpus_prep_utils.get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, False)
    corpus_path = corpus_prep_utils.create_or_load_joined_corpus_file(corpus_file_iterator)
    return corpus_path

@task(log_prints=True)
def load_model(model_path: str):
    return FastText.load(model_path)

@flow(name="Compute Protein BioWord vectors", log_prints=True)
def compute_and_reduce_bioword_vectors(corpus_df, model, bioword_rule_column, pca_parameters, tsne_parameters):

    # Compute vector representation for sequence bioword representation
    corpus_df = compute_sequence_vectors(corpus_df, model, bioword_rule_column)

    # free up memory
    del model

    # Apply PCA to do a more efficient and first dimensionality reduction
    pca_reduced_vectors = compute_pca(corpus_df, pca_parameters)
    
    # Apply TSNE to do final dimensionality reduction to 2D
    tsne_reduced_vectors = compute_tsne(pca_reduced_vectors, tsne_parameters)

    corpus_df[['reduced_vector_d1', 'reduced_vector_d2']] = tsne_reduced_vectors
    
    return corpus_df

# #######################################
#      MAIN FLOW                        #
# #######################################

@flow(name="Evaluate BioWord Protein Model", log_prints=True)
def eval_corpus(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, bioword_rule_column, pca_parameters, tsne_parameters):
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
    corpus_df = compute_and_reduce_bioword_vectors(corpus_df, model, bioword_rule_column, pca_parameters, tsne_parameters)

    # create intermediate outputs
    date = corpus_prep_utils.get_date_from_formatted_ts(timestamp)
    parent_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    os.makedirs(parent_folder_path, exist_ok=True)
    corpus_df.drop('word_partition', axis=1, inplace=True)

    # save df
    df_out_path = os.path.join(parent_folder_path, "output.csv")
    corpus_df.to_csv(df_out_path, index=False)

    # Create vectors dataframe by unnesting the vector column
    vectors_df = pd.DataFrame(
        corpus_df["bio_vector"].tolist(),
        columns=[f'dim_{i}' for i in range(len(corpus_df["bio_vector"].iloc[0]))]
    )
    metadata_df = corpus_df[["sequence", "sequence_family_name", "sequence_family_type"]]
    del corpus_df
    metadata_out_path = os.path.join(parent_folder_path, "metadata.tsv")
    vectors_out_path = os.path.join(parent_folder_path, "vectors.tsv")
    metadata_df.to_csv(metadata_out_path, sep='\t', index=False)
    vectors_df.to_csv(vectors_out_path, sep='\t', header=False, index=False)


# #######################################
#      MAIN                             #
# #######################################

# run the flow!
if __name__=="__main__":
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    # set run data to work on
    family_dataset_name = dataset_names.TEST_GROUP
    #timestamp = "20241030_11_14_21"
    timestamp = "20241029_15_41_02"
    filter_name = filters.MR_FILTER_NONE.name
    partition_rule_name = partition_rules.PARTITION_RULE_USE_ALL["name"]

    # model evaluation parameters
    bioword_rule_column = bioword_rules.BIOWORD_RULE_PARTITION_COLUMN
    pca_parameters = {
        "n_componenets" : 2
    }
    tsne_parameters = {
        "n_components" : 2,
        "random_state" : 42,
        "method" : "exact",
        "perplexity" : 2,
        "max_iter" : 10000 
    }
    
    with tags("eval"):
        eval_corpus(input_data_root_path, family_dataset_name, timestamp, 
                    filter_name, partition_rule_name, bioword_rule_column,
                    pca_parameters, tsne_parameters)