import ahocorasick
import os
import time
from dotenv import dotenv_values
import logging
import argparse
from gensim.models.fasttext import FastText
from gensim.utils import tokenize
import pandas as pd
import numpy as np
from utils.utils import dataset_names, filters, partition_rules, bioword_rules
import utils.utils as corpus_prep_utils

VECTOR_SIZE = 100

# #######################################
#      PATTERN MATCHING                 #
# #######################################

def build_automaton(patterns_list):
    """Build and finalize an Aho-Corasick automaton from a list of MR patterns."""
    A = ahocorasick.Automaton(ahocorasick.STORE_ANY)
    for pattern_str in patterns_list:
        if pattern_str and isinstance(pattern_str, str):
            A.add_word(pattern_str, pattern_str)
        else:
            logging.warning(f"Skipping invalid pattern: {pattern_str}")
    A.make_automaton()
    return A


def apply_automaton_to_sequences(sequences_series, automaton):
    """
    Apply Aho-Corasick automaton to each sequence to produce a word_partition string.
    Matches are sorted by start position and joined with spaces — identical to the
    control-sequence approach in corpus_prep_control.py.
    Returns a list of strings (one per sequence; empty string when no match found).
    """
    output = []
    total = len(sequences_series)
    start = time.time()

    for i, sequence in enumerate(sequences_series):
        if (i + 1) % 10000 == 0 or (i + 1) == total:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else float("inf")
            logging.info(
                f"Pattern matching: {i+1}/{total} sequences "
                f"({(i+1)/total*100:.1f}%, {rate:.0f} seq/s)"
            )

        if not isinstance(sequence, str) or not sequence:
            output.append("")
            continue

        matches = []
        for end_idx, pattern in automaton.iter(sequence):
            start_idx = end_idx - len(pattern) + 1
            matches.append((start_idx, pattern))

        if not matches:
            output.append("")
            continue

        matches.sort(key=lambda x: x[0])
        output.append(" ".join(p for _, p in matches))

    return output


# #######################################
#      EMBEDDING                        #
# #######################################

def get_vector_for_bioword_partition(bioword_partition, sequence, model):
    """Embedding vector for a sequence from its bioword partition (same as model_embeddings.py)."""
    if not isinstance(bioword_partition, str):
        raise TypeError(f"bioword_partition must be str, got {type(bioword_partition).__name__}")

    tokens = list(tokenize(bioword_partition))
    if not tokens:
        tokens.append(sequence)

    return model.wv.get_mean_vector(tokens, pre_normalize=False, post_normalize=True)


def compute_sequence_vectors(corpus_df, model, bioword_rule_column="word_partition", chunk_size=10000):
    """Compute embedding vectors from bioword partitions (same logic as model_embeddings.py)."""
    logging.info("START TASK - compute_sequence_vectors (chunked)")

    bio_vectors = []
    total_rows = len(corpus_df)

    for i in range(0, total_rows, chunk_size):
        chunk_end = min(i + chunk_size, total_rows)
        chunk = corpus_df[bioword_rule_column].iloc[i:chunk_end]
        sequence_chunk = corpus_df["sequence"].iloc[i:chunk_end].astype(str)
        chunk_vectors = pd.concat([chunk.astype(str), sequence_chunk], axis=1).apply(
            lambda row: get_vector_for_bioword_partition(row.iloc[0], row.iloc[1], model), axis=1
        )
        bio_vectors.extend(chunk_vectors.tolist())

        if chunk_end % 50000 == 0 or chunk_end == total_rows:
            logging.info(
                f"Computed vectors: {chunk_end}/{total_rows} ({chunk_end/total_rows*100:.1f}%)"
            )

    corpus_df = corpus_df.copy()
    corpus_df["bio_vector"] = bio_vectors
    logging.info("END TASK - compute_sequence_vectors (chunked)")
    return corpus_df


def load_model(model_path, use_mmap=False):
    logging.info("START TASK - load_model")
    if use_mmap:
        logging.info("Loading model with mmap='r' — slower but uses less RAM")
        model = FastText.load(model_path, mmap="r")
    else:
        logging.info("Loading model into RAM — faster but uses more memory")
        model = FastText.load(model_path)
    logging.info("END TASK - load_model")
    return model


# #######################################
#      I/O HELPERS                      #
# #######################################

def load_input_corpus(input_data_root_path, input_dataset_name, input_timestamp,
                      input_filter_name, input_partition_rule_name):
    """Load the for_eval corpus from the input run."""
    logging.info("START TASK - load_input_corpus")
    corpus_file_iterator = corpus_prep_utils.get_corpus_file_iterator_from_run(
        input_data_root_path, input_dataset_name, input_timestamp,
        input_filter_name, input_partition_rule_name, is_for_train=False
    )
    corpus_path = corpus_prep_utils.create_or_load_joined_corpus_file(corpus_file_iterator)
    df = pd.read_csv(corpus_path, encoding="utf-8", keep_default_na=False)
    logging.info(f"Loaded {len(df)} sequences from {corpus_path}")
    logging.info("END TASK - load_input_corpus")
    return df


def create_and_save_metadata(corpus_df, filename_prefix, parent_folder_path):
    """Save metadata TSV (all columns except computed internal ones)."""
    logging.info("START TASK - save metadata.tsv")
    exclude_columns = {"word_partition", "bio_vector"}
    metadata_columns = [col for col in corpus_df.columns if col not in exclude_columns]
    metadata_df = corpus_df[metadata_columns].copy()
    out_path = os.path.join(parent_folder_path, filename_prefix + "-metadata.tsv")
    metadata_df.to_csv(out_path, sep="\t", index=False)
    logging.info(f"Metadata saved to {out_path}")
    logging.info("END TASK - save metadata.tsv")


# #######################################
#      MAIN FLOW                        #
# #######################################

def create_cross_embeddings(
    input_data_root_path,
    # --- input dataset (sequences to embed) ---
    input_dataset_name,
    input_timestamp,
    input_filter_name,
    input_partition_rule_name,
    # --- model run (patterns + trained model to use) ---
    model_dataset_name,
    model_timestamp,
    model_filter_name,
    model_partition_rule_name,
    # --- compact label for the model run ---
    model_tag,
    # --- options ---
    use_mmap=False,
    metadata_only=False,
):
    if metadata_only:
        logging.info("START FLOW *** Create Cross-Dataset Metadata Only ***")
    else:
        logging.info("START FLOW *** Create Cross-Dataset Embeddings ***")

    # Output mirrors the structure of model_embeddings.py: placed under the
    # *input* run's vector_output folder, with filename derived from the input
    # run id plus the compact cross-model tag.
    #
    #   <input_run_id>-cross_<model_tag>-metadata.tsv
    #   <input_run_id>-cross_<model_tag>-vectors_bio.tsv
    input_date = corpus_prep_utils.get_date_from_formatted_ts(input_timestamp)
    parent_folder_path = os.path.join(
        input_data_root_path, input_dataset_name, input_date, "vector_output"
    )
    os.makedirs(parent_folder_path, exist_ok=True)

    input_run_id = f"{input_timestamp}-{input_dataset_name}-{input_filter_name}-{input_partition_rule_name}"
    filename_prefix = f"{input_run_id}-cross_{model_tag}"

    # --- Load input corpus ---
    corpus_df = load_input_corpus(
        input_data_root_path,
        input_dataset_name,
        input_timestamp,
        input_filter_name,
        input_partition_rule_name,
    )

    # Optimize string columns
    for col in ("sequence_name", "sequence_family_name", "sequence_family_type"):
        if col in corpus_df.columns:
            corpus_df[col] = corpus_df[col].astype("category")
    logging.info(f"Input corpus: {len(corpus_df)} sequences")

    # --- Build bioword partitions using the MODEL run's filtered MR patterns ---
    logging.info("Loading filtered patterns from model run...")
    patterns_file_path = corpus_prep_utils.get_filtered_patterns_file_path_from_run(
        input_data_root_path, model_dataset_name, model_timestamp,
        model_filter_name, model_partition_rule_name
    )
    patterns_df = pd.read_csv(patterns_file_path, usecols=["pattern"], dtype={"pattern": str})
    patterns_list = patterns_df["pattern"].dropna().unique().tolist()
    logging.info(f"Loaded {len(patterns_list)} unique patterns")

    logging.info("Building Aho-Corasick automaton...")
    t0 = time.time()
    automaton = build_automaton(patterns_list)
    logging.info(f"Automaton built in {time.time() - t0:.2f}s")
    del patterns_list, patterns_df

    logging.info("Applying patterns to input sequences...")
    word_partitions = apply_automaton_to_sequences(corpus_df["sequence"], automaton)
    corpus_df = corpus_df.copy()
    corpus_df["word_partition"] = word_partitions
    del automaton
    logging.info("Pattern matching complete")

    # --- Save metadata (always) ---
    create_and_save_metadata(corpus_df, filename_prefix, parent_folder_path)

    if metadata_only:
        logging.info("END FLOW *** Create Cross-Dataset Metadata Only ***")
        return

    # --- Load FastText model from the model run ---
    model_path = corpus_prep_utils.get_model_path_by_run(
        input_data_root_path,
        model_dataset_name,
        model_timestamp,
        model_filter_name,
        model_partition_rule_name,
    )
    logging.info(f"Loading model from: {model_path}")
    model = load_model(model_path, use_mmap)

    # --- Compute vectors from bioword partition ---
    bioword_rule_column = bioword_rules.BIOWORD_RULE_PARTITION_COLUMN  # "word_partition"
    corpus_df = compute_sequence_vectors(corpus_df, model, bioword_rule_column)

    del model
    logging.info("Model freed from memory")

    if "word_partition" in corpus_df.columns:
        corpus_df.drop("word_partition", axis=1, inplace=True)
        logging.info("Dropped word_partition column")

    # --- Save vectors ---
    logging.info("START TASK - save vectors_bio.tsv")
    biovector_list = corpus_df["bio_vector"].tolist()
    logging.info(f"Creating vectors DataFrame: {len(biovector_list)} rows × {VECTOR_SIZE} dims")
    biovectors_df = pd.DataFrame(
        biovector_list,
        columns=[f"dim_{i}" for i in range(VECTOR_SIZE)],
        dtype=np.float32,
    )
    corpus_prep_utils.save_vectors_to_tsv(
        biovectors_df, filename_prefix, "-vectors_bio", parent_folder_path, chunk_size=50000
    )
    del biovectors_df, biovector_list
    logging.info("END TASK - save vectors_bio.tsv")
    logging.info("END FLOW *** Create Cross-Dataset Embeddings ***")


# #######################################
#      MAIN                             #
# #######################################

if __name__ == "__main__":
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    config = dotenv_values(dotenv_path)

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, "model_embeddings_cross.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Create protein embeddings for an input dataset using a model trained on a different dataset. "
            "The model run's filtered MR patterns are applied to the input sequences via Aho-Corasick "
            "to produce bioword partitions, which are then embedded with the model's FastText weights."
        )
    )

    # --- Input dataset (sequences to embed) ---
    inp = parser.add_argument_group("input dataset (sequences to embed)")
    inp.add_argument("input_timestamp",      help="Run timestamp of the input dataset")
    inp.add_argument("input_dataset_name",   help="Input dataset name (see dataset_names)")
    inp.add_argument("input_filter",         help="MR filter used for the input run")
    inp.add_argument("input_partition_rule", help="MR partition rule used for the input run")

    # --- Model run (patterns + trained model) ---
    mdl = parser.add_argument_group("model run (MR patterns + trained model to use)")
    mdl.add_argument("model_timestamp",      help="Run timestamp of the model to load")
    mdl.add_argument("model_dataset_name",   help="Dataset name the model was trained on (see dataset_names)")
    mdl.add_argument("model_filter",         help="MR filter used during model training")
    mdl.add_argument("model_partition_rule", help="MR partition rule used during model training")

    # --- Compact model label ---
    parser.add_argument(
        "model_tag",
        help=(
            "Short label identifying the model (e.g. 'bsc', 'family200'). "
            "Appended to output filenames as: <input_run_id>-cross_<model_tag>-*"
        )
    )

    parser.add_argument("--mmap",     action="store_true", help="Use memory mapping when loading the model")
    parser.add_argument("--metadata", action="store_true", help="Create metadata only (skip vector computation)")

    args = parser.parse_args()

    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]

    create_cross_embeddings(
        input_data_root_path=input_data_root_path,
        # input dataset
        input_dataset_name=getattr(dataset_names, args.input_dataset_name),
        input_timestamp=args.input_timestamp,
        input_filter_name=getattr(filters, args.input_filter).name,
        input_partition_rule_name=getattr(partition_rules, args.input_partition_rule)["name"],
        # model run
        model_dataset_name=getattr(dataset_names, args.model_dataset_name),
        model_timestamp=args.model_timestamp,
        model_filter_name=getattr(filters, args.model_filter).name,
        model_partition_rule_name=getattr(partition_rules, args.model_partition_rule)["name"],
        # compact label
        model_tag=args.model_tag,
        # options
        use_mmap=args.mmap,
        metadata_only=args.metadata,
    )
