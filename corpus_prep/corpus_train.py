import os
import logging
import time
from dotenv import dotenv_values
import argparse
from gensim.models.fasttext import FastText
from utils.utils import dataset_names, filters, partition_rules
import utils.utils as corpus_prep_utils

cpu_count = os.cpu_count()
VECTOR_SIZE = 100

def get_corpus_train_file_path_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    logging.info("START TASK - get_corpus_train_file_path_from_run")
    corpus_file_iterator = corpus_prep_utils.get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    corpus_path = corpus_prep_utils.create_or_load_joined_corpus_file(corpus_file_iterator)
    logging.info("END TASK - get_corpus_train_file_path_from_run")
    return corpus_path

def get_corpus_file_iterator_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    logging.info("START TASK - get_corpus_file_iterator_from_run")
    result = corpus_prep_utils.get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    logging.info("END TASK - get_corpus_file_iterator_from_run")
    return result

def create_or_load_joined_corpus_file(run_files_iterator):
    logging.info("START TASK - create_or_load_joined_corpus_file")
    result = corpus_prep_utils.create_or_load_joined_corpus_file(run_files_iterator)
    logging.info("END TASK - create_or_load_joined_corpus_file")
    return result

def save_model(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str, model):
    logging.info("START TASK - save_model")
    date = corpus_prep_utils.get_date_from_formatted_ts(timestamp)
    parent_path = os.path.join(input_data_root_path, family_dataset_name, date, "models")
    os.makedirs(parent_path, exist_ok=True)
    run_id = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name
    file_path = os.path.join(parent_path, run_id+".model")
    model.save(file_path)
    logging.info(f"END TASK - save_model (saved to {file_path})")
    return file_path

def count_file_sentences_words(file_path):
    logging.info(f"START TASK - count_file_sentences_words ({file_path})")
    result = corpus_prep_utils.count_lines(file_path)
    logging.info(f"END TASK - count_file_sentences_words ({file_path})")
    return result

def build_model_vocabulary(corpus_file_path, model, is_update=False):
    logging.info(f"START TASK - build_model_vocabulary (update={is_update})")
    model.build_vocab(corpus_file=corpus_file_path, update=is_update)
    logging.info("END TASK - build_model_vocabulary")
    return model

def train_model(corpus_file_path, model, epochs=5, total_examples_count=0, total_words_count=0):
    logging.info("START TASK - train_model")
    total_examples_count = model.corpus_count if total_examples_count == 0 else total_examples_count
    total_words_count = model.corpus_total_words if total_words_count == 0 else total_words_count
    logging.info(f"Starting training. Total examples seen: {model.corpus_count}, Total words seen: {model.corpus_total_words}")
    trained_word_count, raw_word_count = model.train(
        corpus_file=corpus_file_path, epochs=epochs,
        total_words=total_words_count, total_examples=total_examples_count
    )
    logging.info(f"Trained {trained_word_count} words. Raw word count {raw_word_count}")
    logging.info("END TASK - train_model")
    return model

# #######################################
#      MAIN FLOW                        #
# #######################################

def train_corpus(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str, vector_size: int = VECTOR_SIZE, workers: int = cpu_count):
    logging.info("START FLOW ******************* Train BioWord Protein Corpus (Single) *******************")
    # get file corpus iterable for desired run output 
    corpus_path = get_corpus_train_file_path_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    
    # it seems paralell training works only with hs=1 and negative=0
    # sg=1 uses skip-gram which has been shown to be better for subword info
    model = FastText(vector_size=vector_size, workers=workers, hs=1, negative=0, sg=1)

    model = build_model_vocabulary(corpus_path, model)
    
    # optional intermediate save as checkpoint for built vocabulary
    #path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)
    
    model = train_model(corpus_path, model)

    path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)

    logging.info(f"Saved trained model {path}")
    logging.info("END FLOW ******************* Train BioWord Protein Corpus (Single) *******************")

def train_corpus_iteratively(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    logging.info("START FLOW ******************* Train BioWord Protein Corpus (Iteratively) *******************")
    
    # get file corpus iterable for desired run output 
    corpus = get_corpus_file_iterator_from_run(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name)
    logging.info("Loaded corpus iterable")
    i = 0
    path = ''
    # it seems paralell training works only with hs=1 and negative=0
    model = FastText(vector_size=100, workers=cpu_count, hs=1, negative=0)
    for file in corpus:
        logging.info(f"Processing corpus file {i}")
        update = False if i == 0 else True
        model = model if i == 0 else FastText.load(path)
        total_examples, total_words = count_file_sentences_words(file)
        logging.info(f"{total_examples} examples and {total_words} unique words in corpus file {i}")
        model = build_model_vocabulary(file, model, update)
        model = train_model(file, total_examples, total_words, model)
        path = save_model(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, model)
        i+=1
    logging.info(f"Finished training. Total examples seen: {model.corpus_count}, Total words seen: {model.corpus_total_words}")
    logging.info("END FLOW ******************* Train BioWord Protein Corpus (Iteratively) *******************")
    return model

# #######################################
#      MAIN                             #
# #######################################

if __name__=="__main__":
    #travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'corpus_train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]

    parser = argparse.ArgumentParser(description='Perform BioWord Protein Corpus Training')
    # Corpus input arguments
    parser.add_argument('timestamp', help='Run timestamp')
    parser.add_argument('dataset_name', help='Input protein dataset name')
    parser.add_argument('filter', help='MR filter')
    parser.add_argument('partition_rule', help='MR partition rule')
    # Model training parameters
    parser.add_argument('--vector-size', type=int, default=VECTOR_SIZE, help='Model vector size')
    parser.add_argument('--max-cpu', type=int, default=cpu_count, help='Max number of CPUs to use in parallel training')
    
    args = parser.parse_args()

    timestamp = args.timestamp
    family_dataset_name = getattr(dataset_names, args.dataset_name)
    filter_name = getattr(filters, args.filter).name
    partition_rule_name = getattr(partition_rules, args.partition_rule)["name"]

    overall_start_time = time.time()
    logging.info("Starting corpus training...")

    train_corpus(input_data_root_path, family_dataset_name, timestamp, 
                 filter_name, partition_rule_name, args.vector_size, args.max_cpu)

    logging.info(f"Script finished in {time.time() - overall_start_time:.2f} seconds.")