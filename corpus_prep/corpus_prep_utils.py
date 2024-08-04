from prefect import task as prefect_task
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import json
import os

@dataclass
class MRFilter:

    by: str = field(metadata={"description": "The field or special logic by which to filter"}, default="none")
    name: str = field(metadata={"description": "The name of the filter"}, default="Unnamed filter")
    condition: str = field(metadata={"description": "The condition operator by which to filter"}, default="")
    value: str = field(metadata={"description": "The value by which to filter the condition"}, default="")

def in_prefect_flow_context():
    try:
        from prefect import context
        flow_run_ctx = context.get_run_context()
        return (
            flow_run_ctx.flow_run is not None
        )  # Check if in a flow run context
    except (ImportError, AttributeError, RuntimeError):  
        return False
    
# custom task decorator to switch prefect behaviour to normal
# when calling from outside flow
def task(**task_kwargs): # Accept kwargs for @task
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if in_prefect_flow_context():
                return prefect_task(func, **task_kwargs)(*args, **kwargs)  # Apply with kwargs
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

# #######################################
#      I/O                              #
# #######################################

def parse_json(str):
    try:
        return json.loads(str)
    except json.JSONDecodeError:
        raise 

# Function to convert nested structures to JSON strings
def to_json(nested_data):
    return json.dumps(nested_data)

def save_dataset(input_data_root_path: str, family_dataset_name: str, dataset_df, file_name):
    output_df = dataset_df.copy(deep=False)
    now_timestamp = datetime.now()
    date_formatted = now_timestamp.strftime("%Y%m%d")
    ts_formatted = now_timestamp.strftime("%Y%m%d_%H_%M_%S")
    parent_folder_path = os.path.join(input_data_root_path, family_dataset_name, date_formatted)
    os.makedirs(parent_folder_path, exist_ok=True)
    write_path = os.path.join(parent_folder_path, ts_formatted+":"+file_name+".csv")

    try:
        # Apply the function to the column with nested structure
        output_df['affected_proteins'] = output_df['affected_proteins'].apply(to_json)
    except:
        pass

    output_df.to_csv(write_path, index=False)

def load_dataset(input_data_root_path: str, family_dataset_name: str, timestamp: str, step_name: str, filter_name: str, partition_rule_name: str):
    date = timestamp.split("_")[0]
    file_name = timestamp+":"+step_name+":"+family_dataset_name+":"+filter_name+":"+partition_rule_name+".csv"
    file_path = os.path.join(input_data_root_path, family_dataset_name, date, file_name)
    dataset_df = pd.read_csv(file_path, converters={'affected_proteins': parse_json})

    #if step_name == data_step_names.S1_FILTERED_MR:
        #dataset_df['affected_proteins'] = dataset_df['affected_proteins'].apply(parse_json)
        #dataset_df['affected_proteins'] = pd.json_normalize(dataset_df['affected_proteins'], max_level=0) 
    return dataset_df

# #######################################
#      CONSTANTS                        #
# #######################################

class dataset_names():
    TEST_GROUP = "testGroupDataset"
    NANO_GROUP = "nanoGroupDataset"
    FAMILY = "familyDataset"

class data_step_names():
    S1_FILTERED_MR = "s1_filtered_mrs"
    S3_CORPUS = "s3_corpus"

class filters():
    MR_FILTER_NONE = MRFilter(by="none", name="filter_none")

    MR_FILTER_KEEP_SIGNIFICANT = MRFilter(condition="include", by="significance", name="filter_keep_significant")

    MR_FILTER_DROP_SMR = MRFilter(condition="!=", by="type", value="SMR", name="filter_drop_smr")

    MR_FILTER_DROP_NE = MRFilter(condition="!=", by="type", value="NE", name="filter_drop_ne")

    MR_FILTER_DROP_NN = MRFilter(condition="!=", by="type", value="NN", name="filter_drop_nn")

    MR_FILTER_KEEP_LEN6PLUS = MRFilter(condition= ">=", by="length", value=6, name="filter_keep_length_6plus")

class partition_rules():
    PARTITION_RULE_USE_ALL = {
        "name" : "partition_use_all"
    }