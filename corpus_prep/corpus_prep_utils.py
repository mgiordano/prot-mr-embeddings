#from prefect import Task
#from prefect.context import FlowRunContext
#from prefect.utilities.asyncutils import sync_compatible
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from gensim.utils import tokenize
from gensim import utils

@dataclass
class MRFilter:

    by: str = field(metadata={"description": "The field or special logic by which to filter"}, default="none")
    name: str = field(metadata={"description": "The name of the filter"}, default="Unnamed filter")
    condition: str = field(metadata={"description": "The condition operator by which to filter"}, default="")
    value: str = field(metadata={"description": "The value by which to filter the condition"}, default="")

"""def in_prefect_flow_context():
    try:
        from prefect import context
        flow_run_ctx = context.get_run_context()
        return (
            flow_run_ctx.flow_run is not None
        )  # Check if in a flow run context
    except (ImportError, AttributeError, RuntimeError):  
        return False

class CustomTask(Task):
    @sync_compatible
    async def __call__(self, *args, **kwargs):
        #if FlowRunContext.get() is not None:
        if in_prefect_flow_context():
            # In a flow context, behave like a normal task
            return await super().__call__(*args, **kwargs)
        else:
            return await self.fn(*args, **kwargs)
        
# custom task decorator to switch prefect behaviour to normal
# when calling from outside flow
def task(**task_kwargs):
    def decorator(func):
        return CustomTask(fn=func, **task_kwargs)
    return decorator"""

# #######################################
#      I/O                              #
# #######################################

class RunFilesCorpus:
    def __init__(self, corpus_folder_path, run_file_prefix):
        self.path = corpus_folder_path
        self.prefix_filter = run_file_prefix

    def __iter__(self):
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            if filename.startswith(self.prefix_filter) and filename.endswith(".csv"):
                if os.path.isfile(file_path):  # Make sure it's a file
                    yield file_path
                    #with utils.open(file_path, 'r', encoding='utf-8') as fin:
                        #for line in fin:
                            #yield list(tokenize(line))

def print_df(dataframe, limit=10):
    print(dataframe.head(limit).to_markdown(index=False, numalign="left", stralign="left"), end="\n")
    print("-")

def parse_json(str):
    try:
        return json.loads(str)
    except json.JSONDecodeError:
        raise 

# Function to convert nested structures to JSON strings
def to_json(nested_data):
    return json.dumps(nested_data)

def get_date_from_formatted_ts(formatted_ts: str):
    return formatted_ts.split("_")[0]

def create_run_id(family_dataset_name: str, filter_name: str, partition_rule_name: str, separator="-"):
    now_timestamp = datetime.now()
    ts_formatted = now_timestamp.strftime("%Y%m%d_%H_%M_%S")
    run_id = separator.join([ts_formatted, family_dataset_name, 
              filter_name, partition_rule_name])
    return run_id

def save_local(input_data_root_path: str, family_dataset_name: str, run_id: str, dataset_df):
    date_formatted = get_date_from_formatted_ts(run_id.split("-")[0])
    parent_folder_path = os.path.join(input_data_root_path, family_dataset_name, date_formatted)
    os.makedirs(parent_folder_path, exist_ok=True)
    file_name = run_id+".csv"
    output_path = os.path.join(parent_folder_path, file_name)
    dataset_df.to_csv(output_path, index=False)
    return parent_folder_path

def get_file_path_by_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, step_name: str, filter_name: str, partition_rule_name: str):
    date = get_date_from_formatted_ts(timestamp)
    file_name = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name+"-"+step_name+".csv"
    return os.path.join(input_data_root_path, family_dataset_name, date, file_name)

def get_stage_run_table_name(family_dataset_name: str, timestamp: str, step_name: str, filter_name: str, partition_rule_name: str):
    return timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name+"-"+step_name

def get_model_path_by_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    date = get_date_from_formatted_ts(timestamp)
    file_name = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name+".model"
    return os.path.join(input_data_root_path, family_dataset_name, date, "models", file_name)

def count_lines(filename):
    with utils.open(filename, 'r', encoding='utf-8') as f:
        word_count = 0
        line_count = 0
        for line in f:
            line_count += 1
            word_count += len(list(tokenize(line)))
        return line_count, word_count
            

# #######################################
#      CONSTANTS                        #
# #######################################

class dataset_names():
    TEST_GROUP = "testGroupDataset"
    NANO_GROUP = "nanoGroupDataset"
    FAMILY = "familyDataset"

class data_step_names():
    S1_FILTERED_MR = "s1_filtered_mrs"
    S2_JOINED_MR = "s2_joined_mrs"
    S3_CORPUS = "s3_corpus"

class filters():
    MR_FILTER_NONE = MRFilter(by="1", condition="=", value="1", name="filter_none")

    MR_FILTER_KEEP_SIGNIFICANT = MRFilter(condition="include", by="significance", name="filter_keep_significant")

    MR_FILTER_DROP_SMR = MRFilter(condition="!=", by="type", value="SMR", name="filter_drop_smr")

    MR_FILTER_DROP_NE = MRFilter(condition="!=", by="type", value="NE", name="filter_drop_ne")

    MR_FILTER_DROP_NN = MRFilter(condition="!=", by="type", value="NN", name="filter_drop_nn")

    MR_FILTER_DROP_ALL = MRFilter(condition= "=", by="type", value="NONE", name="filter_drop_all")

    def create_filter_keep_len_plus(length: int):
        return MRFilter(condition= ">=", by="LENGTH(pattern)", value=length, name=f"filter_keep_length_{str(length)}plus")

class partition_rules():
    PARTITION_RULE_USE_ALL = {
        "name" : "partition_use_all"
    }