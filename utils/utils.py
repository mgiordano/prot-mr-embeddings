#from prefect import Task
#from prefect.context import FlowRunContext
#from prefect.utilities.asyncutils import sync_compatible
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from gensim.utils import tokenize
from gensim import utils
from utils.database.storage_helper import StorageHelper
import re

@dataclass
class MRFilter:

    by: str = field(metadata={"description": "The field or special logic by which to filter"}, default="none")
    name: str = field(metadata={"description": "The name of the filter"}, default="Unnamed filter")
    condition: str = field(metadata={"description": "The condition operator by which to filter"}, default="")
    value: str = field(metadata={"description": "The value by which to filter the condition"}, default="")

    def get_filter_for_query(self):
        # escape string values with ''
        value = f"'{self.value}'" if (isinstance(self.value, str) and self.condition != 'BETWEEN') else self.value
        filter_string = f"{self.by} {self.condition} {value}"
        return filter_string

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

def extract_shard_number(filename):
    """
    Extract the numeric shard number from a filename.
    Handles various shard naming patterns like:
    - filename_000000000000.csv
    - filename_0.csv  
    - filename_1.csv
    - filename_filtered_patterns_000000000001.csv
    
    Returns the numeric value for sorting, or 0 if no number found.
    """
    # Look for patterns like _000000000000.csv, _0.csv, _1.csv at the end
    match = re.search(r'_(\d+)\.csv$', filename)
    if match:
        return int(match.group(1))
    
    # If no pattern found, return 0 (will be sorted first)
    return 0

def sort_shard_files(file_paths):
    """
    Sort shard files by their numeric suffix in ascending order.
    
    Args:
        file_paths: List of file paths to sort
        
    Returns:
        List of file paths sorted by shard number
    """
    return sorted(file_paths, key=lambda path: extract_shard_number(os.path.basename(path)))

class RunFilesCorpus:
    def __init__(self, corpus_folder_path, run_file_prefix):
        self.path = corpus_folder_path
        self.prefix_filter = run_file_prefix

    def __iter__(self):
        # Collect all matching files first
        matching_files = []
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            if filename.startswith(self.prefix_filter) and filename.endswith(".csv"):
                if os.path.isfile(file_path):  # Make sure it's a file
                    matching_files.append(file_path)
        
        # Sort files by shard number before yielding
        for file_path in sort_shard_files(matching_files):
            yield file_path

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

def get_model_path_by_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str, load_vectors=False):
    date = get_date_from_formatted_ts(timestamp)
    file_name = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name+".model"
    file_name += ".wv.vectors_ngrams.kv" if load_vectors else ""
    return os.path.join(input_data_root_path, family_dataset_name, date, "models", file_name)

def count_lines(filename):
    with utils.open(filename, 'r', encoding='utf-8') as f:
        word_count = 0
        line_count = 0
        for line in f:
            line_count += 1
            word_count += len(list(tokenize(line)))
        return line_count, word_count
    
def get_corpus_file_iterator_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str, is_for_train=True):
    date = get_date_from_formatted_ts(timestamp)
    corpus_type = "for_train" if is_for_train else "for_eval"
    file_name_prefix = timestamp+"-"+family_dataset_name+"-"+filter_name+"-"+partition_rule_name+"-"+data_step_names.S3_CORPUS+"_"+corpus_type+"_"
    parent_path = os.path.join(input_data_root_path, family_dataset_name, date)
    StorageHelper().download_files_matching_prefix("processed_proteins", file_name_prefix, family_dataset_name+"/corpus/"+corpus_type, parent_path)
    return RunFilesCorpus(parent_path, file_name_prefix)

def join_csv_shards(output_file_path: str, shard_files, header_line: str = None, skip_joined_files: bool = True):
    """
    Generic function to join CSV shard files into a single file.
    
    Args:
        output_file_path: Path where the joined file should be created
        shard_files: Iterable of file paths to join (can be file paths or file iterator)
        header_line: Optional header line to write first (if None, uses header from first file)
        skip_joined_files: Whether to skip files that already contain "joined" in the name
    
    Returns:
        Path to the joined file
    """
    if not os.path.exists(output_file_path):
        print(f"Creating joined file {output_file_path}...")
        
        # Convert iterator to list and filter/sort files
        file_list = []
        for file_path in shard_files:
            # Skip already joined files if requested
            if skip_joined_files and "joined" in os.path.basename(file_path):
                continue
            if os.path.isfile(file_path):
                file_list.append(file_path)
        
        # Sort files by shard number to ensure consistent ordering
        sorted_files = sort_shard_files(file_list)
        
        with open(output_file_path, "w", encoding='utf-8') as outfile:
            # Write custom header if provided
            if header_line:
                outfile.write(header_line)
                if not header_line.endswith('\n'):
                    outfile.write('\n')
            
            for i, file_path in enumerate(sorted_files):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    print(f"Joining shard file {file_path} (shard #{extract_shard_number(os.path.basename(file_path))})")
                    
                    # Skip header for all files except the first (unless custom header provided)
                    if i > 0 or header_line:
                        try:
                            next(infile)  # Skip header row
                        except StopIteration:
                            continue  # Empty file
                    
                    # Copy all lines from this shard
                    for line in infile:
                        outfile.write(line)
    
    return output_file_path

def create_or_load_joined_corpus_file(run_files_iterator):
    joined_file_name = run_files_iterator.prefix_filter + "joined.csv"
    joined_file_path = os.path.join(run_files_iterator.path, joined_file_name)
    return join_csv_shards(joined_file_path, run_files_iterator, skip_joined_files=True)

def save_vectors_to_tsv(vectors, filename_prefix, filename_suffix, parent_folder_path, chunk_size=None):
    """Save vectors to TSV file, with optional chunked processing for large datasets"""
    filename = filename_prefix + filename_suffix + ".tsv"
    out_path = os.path.join(parent_folder_path, filename)
    
    if isinstance(vectors, pd.DataFrame):
        # Use chunked processing if chunk_size is specified and dataset is large
        if chunk_size is not None and len(vectors) > chunk_size:
            import logging
            total_rows = len(vectors)
            logging.info(f"Saving {total_rows} vectors to {out_path} in chunks of {chunk_size}")
            
            # Save in chunks to avoid memory issues
            for i in range(0, total_rows, chunk_size):
                chunk_end = min(i + chunk_size, total_rows)
                chunk = vectors.iloc[i:chunk_end]
                
                # Write mode: 'w' for first chunk, 'a' for subsequent chunks
                mode = 'w' if i == 0 else 'a'
                
                chunk.to_csv(out_path, sep='\t', header=False, index=False, 
                            float_format='%.20f', mode=mode)
                
                # Log progress for large datasets
                if chunk_end % 100000 == 0 or chunk_end == total_rows:
                    logging.info(f"Saved {chunk_end}/{total_rows} vector rows ({(chunk_end/total_rows)*100:.1f}%)")
        else:
            # Original behavior for smaller datasets
            vectors.to_csv(out_path, sep='\t', header=False, index=False, float_format='%.20f')
    else:
        # Original numpy array handling
        np.savetxt(out_path, vectors, delimiter='\t', fmt='%.20f')
    
    return out_path

def get_control_sequences_file_path(input_data_root_path: str, family_dataset_name: str, timestamp: str):
    """Get the file path for control sequences dataset."""
    control_dataset_name = f"control_{family_dataset_name}"
    file_name = f"{control_dataset_name}_sequence_dataset.csv"
    return os.path.join(input_data_root_path, family_dataset_name, file_name)

def get_filtered_patterns_file_path_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    """Get or create the joined filtered patterns file from a run."""
    from utils.database.database_helper import DatabaseHelper
    
    # Initialize database helper
    db_helper = DatabaseHelper(family_dataset_name, input_data_root_path, "", dry_run=False)
    
    # Construct the filtered MRs table name
    filtered_mrs_table_name = get_stage_run_table_name(family_dataset_name, timestamp, data_step_names.S1_FILTERED_MR, filter_name, partition_rule_name)
    
    # Check if table exists
    if not db_helper.check_table_existence(db_helper.BQ_STAGE_DATASET_NAME, filtered_mrs_table_name):
        raise ValueError(f"Filtered MRs table {filtered_mrs_table_name} not found. Please run the corpus preparation pipeline first.")
    
    # Export to GCS and download
    date = get_date_from_formatted_ts(timestamp)
    gcs_root_path = os.path.join(family_dataset_name, date, "filtered_patterns")
    export_suffix = "_filtered_patterns_*"
    
    print(f"Exporting filtered patterns table {filtered_mrs_table_name} to GCS...")
    db_helper.export_table_to_gcs_as_csv(filtered_mrs_table_name, gcs_root_path, 
                                         export_columns=["pattern"], 
                                         file_name_suffix=export_suffix)
    
    # Download the exported files
    local_download_path = os.path.join(input_data_root_path, family_dataset_name, date)
    file_name_prefix = filtered_mrs_table_name + "_filtered_patterns_"
    
    print(f"Downloading filtered patterns files to {local_download_path}...")
    db_helper.storage_helper.download_files_matching_prefix(
        db_helper.storage_helper.bucket_name, 
        file_name_prefix, 
        gcs_root_path, 
        local_download_path
    )
    
    # Create joined CSV file from downloaded shards
    joined_file_name = f"{filtered_mrs_table_name}_filtered_patterns_joined.csv"
    joined_file_path = os.path.join(local_download_path, joined_file_name)
    
    # Get list of shard files to join
    shard_files = []
    for filename in os.listdir(local_download_path):
        if filename.startswith(file_name_prefix) and filename.endswith(".csv"):
            shard_files.append(os.path.join(local_download_path, filename))
    
    # Use the generic join function with custom header
    return join_csv_shards(joined_file_path, shard_files, header_line="pattern", skip_joined_files=True)

def save_control_results(input_data_root_path: str, family_dataset_name: str, timestamp: str, 
                        filter_name: str, partition_rule_name: str, results_df):
    """Save control results following the same naming convention as other pipeline outputs."""
    # Create run ID and file path following the same convention
    run_id = f"{timestamp}-{family_dataset_name}-{filter_name}-{partition_rule_name}"
    output_file_name = f"{run_id}-{data_step_names.S3_CORPUS}_control_for_eval.csv"
    
    date = get_date_from_formatted_ts(timestamp)
    output_dir = os.path.join(input_data_root_path, family_dataset_name, date)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_file_name)
    results_df.to_csv(output_path, index=False)
    
    print(f"Control results saved to {output_path}")
    return output_path

def get_control_corpus_file_path_from_run(input_data_root_path: str, family_dataset_name: str, timestamp: str, filter_name: str, partition_rule_name: str):
    """Get the file path for control corpus dataset generated by corpus_prep_control.py."""
    run_id = f"{timestamp}-{family_dataset_name}-{filter_name}-{partition_rule_name}"
    output_file_name = f"{run_id}-{data_step_names.S3_CORPUS}_control_for_eval.csv"
    
    date = get_date_from_formatted_ts(timestamp)
    output_path = os.path.join(input_data_root_path, family_dataset_name, date, output_file_name)
    
    return output_path

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
    def create_filter_keep_len_plus(length: int):
        return MRFilter(condition= ">=", by="LENGTH(pattern)", value=length, name=f"filter_keep_length_{str(length)}plus")
    
    def create_filter_keep_len_between(min_length: int, max_length: int):
        return MRFilter(condition= "BETWEEN", by="LENGTH(pattern)", 
                        value=f"{min_length} AND {max_length}", 
                        name=f"filter_keep_length_{str(min_length)}to{str(max_length)}")

    MR_FILTER_NONE = MRFilter(by="1", condition="=", value="1", name="filter_none")

    MR_FILTER_KEEP_SIGNIFICANT = MRFilter(condition="include", by="significance", name="filter_keep_significant")

    MR_FILTER_DROP_SMR = MRFilter(condition="!=", by="type", value="SMR", name="filter_drop_smr")

    MR_FILTER_DROP_NE = MRFilter(condition="!=", by="type", value="NE", name="filter_drop_ne")

    MR_FILTER_DROP_NN = MRFilter(condition="!=", by="type", value="NN", name="filter_drop_nn")

    MR_FILTER_DROP_ALL = MRFilter(condition= "=", by="type", value="NONE", name="filter_drop_all")

    MR_FILTER_KEEP_4_10 = create_filter_keep_len_between(4, 10)


class partition_rules():
    PARTITION_RULE_USE_ALL = {
        "name" : "partition_use_all"
    }

class bioword_rules():
    BIOWORD_RULE_PARTITION_COLUMN = "word_partition"
    BIOWORD_RULE_SEQ_COLUMN = "sequence"