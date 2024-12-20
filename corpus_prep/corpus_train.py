import os
from dotenv import dotenv_values
from gensim.models.fasttext import FastText
from prefect import flow, tags, task
from corpus_prep_utils import dataset_names, filters, partition_rules
import corpus_prep_utils

cpu_count = os.cpu_count()

@task(log_prints=True)
def get_corpus_file_iterator_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    return corpus_prep_utils.get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)

@task(log_prints=True)
def create_or_load_joined_corpus_file(run_files_iterator):
    return corpus_prep_utils.create_or_load_joined_corpus_file(run_files_iterator)

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
def count_file_sentences_words(file_path):
    return corpus_prep_utils.count_lines(file_path)

@task(log_prints=True)
def build_model_vocabulary(corpus_file_path, model, is_update=False):
    model.build_vocab(corpus_file=corpus_file_path, update=is_update)
    return model

@task(log_prints=True)
def train_model(corpus_file_path, model, epochs=5, total_examples_count=0, total_words_count=0):
    total_examples_count = model.corpus_count if total_examples_count == 0 else total_examples_count
    total_words_count = model.corpus_total_words if total_words_count == 0 else total_words_count
    print(f"Starting training. Total examples seen: {model.corpus_count}, Total words seen: {model.corpus_total_words}")
    trained_word_count, raw_word_count = model.train(
        corpus_file=corpus_file_path, epochs=epochs,
        total_words=total_words_count, total_examples=total_examples_count
    )
    print(f"Trained {trained_word_count} words. Raw word count {raw_word_count}")
    return model

# #######################################
#      MAIN FLOW                        #
# #######################################

@flow(name="Train BioWord Protein Corpus", log_prints=True)
def train_corpus(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    # get file corpus iterable for desired run output 
    corpus_file_iterator = get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    
    corpus_path = create_or_load_joined_corpus_file(corpus_file_iterator)
    
    # it seems paralell training works only with hs=1 and negative=0
    model = FastText(vector_size=100, workers=cpu_count, hs=1, negative=0)

    model = build_model_vocabulary(corpus_path, model)
    
    path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)
    
    model = train_model(corpus_path, model)

    path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)

    print(f"Saved trained model {path}")

@flow(name="Train BioWord Protein Corpus", log_prints=True)
def train_corpus_iteratively(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    
    # get file corpus iterable for desired run output 
    corpus = get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    print("Loaded corpus iterable")
    i = 0
    path = ''
    # it seems paralell training works only with hs=1 and negative=0
    model = FastText(vector_size=100, workers=cpu_count, hs=1, negative=0)
    for file in corpus:
        print(f"Processing corpus file {i}")
        update = False if i == 0 else True
        model = model if i == 0 else FastText.load(path)
        total_examples, total_words = count_file_sentences_words(file)
        print(f"{total_examples} examples and {total_words} unique words in corpus file {i}")
        model = build_model_vocabulary(file, model, update)
        model = train_model(file, total_examples, total_words, model)
        path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)
        i+=1
    print(f"Finished training. Total examples seen: {model.corpus_count}, Total words seen: {model.corpus_total_words}")
    return model

    #model_path = corpus_prep_utils.get_model_path_by_run(input_data_root_path, dataset_names.TEST_GROUP, "20241029_15_41_02", 
    #                                                 filters.MR_FILTER_NONE.name, partition_rules.PARTITION_RULE_USE_ALL["name"] )
    #model = FastText.load(model_path)

    #path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)

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
    #timestamp = "20241029_15_41_02"
    filter_name = filters.MR_FILTER_NONE.name
    partition_rule_name = partition_rules.PARTITION_RULE_USE_ALL["name"]
    
    with tags("train"):
        train_corpus(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)