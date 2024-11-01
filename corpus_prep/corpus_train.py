import os
from dotenv import dotenv_values
from pprint import pprint as print
from gensim.models.fasttext import FastText
from prefect import flow, tags, task
from corpus_prep_utils import dataset_names, data_step_names, filters, partition_rules, RunFilesCorpus
from database.storage_helper import StorageHelper
import corpus_prep_utils

cpu_count = os.cpu_count()

@task(log_prints=True)
def get_corpus_file_iterator_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    date = corpus_prep_utils.get_date_from_formatted_ts(timestamp)
    file_name_prefix = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name+"-"+data_step_names.S3_CORPUS+"_for_train_"
    parent_path = os.path.join(input_data_root_path, family_dataset_name, date)
    storage_helper = StorageHelper()
    storage_helper.download_files_matching_prefix("processed_proteins", file_name_prefix, family_dataset_name+"/corpus/for_train", parent_path)
    return RunFilesCorpus(parent_path, file_name_prefix)

@task(log_prints=True)
def save_model(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str, model):
    date = corpus_prep_utils.get_date_from_formatted_ts(timestamp)
    parent_path = os.path.join(input_data_root_path, family_dataset_name, date, "models")
    os.makedirs(parent_path, exist_ok=True)
    run_id = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name
    file_path = os.path.join(parent_path, run_id+".model")
    model.save(file_path)
    return file_path

@task(log_prints=True)
def build_model_vocabulary(model, corpus):
    model.build_vocab(corpus_iterable=corpus)
    return model

@task(log_prints=True)
def train_model(model, corpus):
    model.train(
        corpus_iterable=corpus, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words
    )
    print(model)
    return model


# #######################################
#      MAIN FLOW                        #
# #######################################

@flow(name="Train BioWord Protein Corpus", log_prints=True)
def train_corpus(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    
    # get file corpus iterable for desired run output 
    corpus = get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)

    # it seems paralell training works only with hs=1 and negative=0
    model = FastText(vector_size=100, workers=cpu_count, hs=1, negative=0)
    
    model = build_model_vocabulary(model, corpus)

    path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)

    #model_path = corpus_prep_utils.get_model_path_by_run(input_data_root_path, dataset_names.TEST_GROUP, "20241029_15_41_02", 
    #                                                 filters.MR_FILTER_NONE.name, partition_rules.PARTITION_RULE_USE_ALL["name"] )
    #model = FastText.load(model_path)

    model = train_model(model, corpus)

    path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)

# #######################################
#      MAIN                             #
# #######################################

# run the flow!
if __name__=="__main__":
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    config = dotenv_values(dotenv_path)
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    
    # set run data to work on
    family_dataset_name = dataset_names.FAMILY
    timestamp = "20241030_11_14_21"
    filter_name = filters.MR_FILTER_NONE.name
    partition_rule_name = partition_rules.PARTITION_RULE_USE_ALL["name"]
    
    with tags("train"):
        train_corpus(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)