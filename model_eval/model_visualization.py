import os
import sys
import argparse
import logging
import glob
import pandas as pd
import numpy as np
from dotenv import dotenv_values
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle

from utils.utils import dataset_names, filters, partition_rules
import utils.utils as utils

def parse_filename_parameters(filename):
    """
    Parse experiment parameters from filename
    """
    # Remove file extension and get the base filename
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Split parameters by '-'
    params = base_filename.split('-')
    
    # Extract timestamp and parameters
    timestamp = params[0]
    dataset = params[1]
    filter_type = params[2].replace('filter_', '')
    partition = params[3].replace('partition_', '')
    vectors_method = params[4].replace('vectors_', '')
    method = params[5]
    perplexity = params[6]
    learning_rate = params[7]
    max_iterations = params[8]
    random_state = params[9]
    
    return {
        'timestamp': timestamp,
        'dataset': dataset,
        'filter': filter_type,
        'partition': partition,
        'vectors_method': vectors_method,
        'method': method,
        'perplexity': perplexity,
        'learning_rate': learning_rate,
        'max_iterations': max_iterations,
        'random_state': random_state
    }

def create_plot_title(params):
    """
    Create a formatted title from parameters
    """
    return (f"t-SNE Visualization of {params['dataset']}\n"
            f"Method: {params['method']}, Perplexity: {params['perplexity']}, "
            f"Learning Rate: {params['learning_rate']}\n"
            f"Max Iterations: {params['max_iterations']}, "
            f"Filter: {params['filter']}, Partition: {params['partition']}")

def optimize_legend_columns(num_labels, fig_height):
    """
    Calculate optimal number of legend columns based on figure height
    """
    # Approximate height of each legend entry in inches
    entry_height = 0.25
    # Available height in inches (considering figure height and margins)
    available_height = fig_height * 0.9
    # Maximum entries per column
    max_entries_per_column = math.floor(available_height / entry_height)
    # Calculate optimal number of columns
    return math.ceil(num_labels / max_entries_per_column)

def create_large_scatter_plot(df, label_column, output_file=None, point_size=1, alpha=0.5, xlim=None, ylim=None):
    """
    Create a scatter plot optimized for large datasets with enhanced formatting
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Parse parameters from filename if provided
    title = f"2D Vector Visualization by {label_column}"
    if output_file:
        params = parse_filename_parameters(output_file)
        title = create_plot_title(params)

    # Get unique labels for coloring
    unique_labels = df[label_column].unique()
    
    # Special handling for family name plotting: use gray shades for control families
    if label_column == "sequence_family_name" and "sequence_family_type" in df.columns:
        # Identify which families are control families
        control_families = set()
        non_control_families = set()
        
        for label in unique_labels:
            family_data = df[df[label_column] == label]
            # Check if this family has any control sequences
            if (family_data["sequence_family_type"] == "control").any():
                control_families.add(label)
            else:
                non_control_families.add(label)
        
        # Create color mapping
        colors = {}
        
        # Assign colorful colors to non-control families
        if non_control_families:
            colorful_palette = sns.color_palette('husl', n_colors=len(non_control_families))
            for idx, family in enumerate(sorted(non_control_families)):
                colors[family] = colorful_palette[idx]
        
        # Assign gray shades to control families
        if control_families:
            # Create different shades of gray for control families
            gray_palette = sns.color_palette('gray', n_colors=len(control_families))
            for idx, family in enumerate(sorted(control_families)):
                colors[family] = gray_palette[idx]
        
        # Create scatter plots with assigned colors
        for label in unique_labels:
            mask = df[label_column] == label
            ax.scatter(df.loc[mask, 'reduced_vector_d1'],
                      df.loc[mask, 'reduced_vector_d2'],
                      c=[colors[label]], label=label,
                      alpha=alpha, s=point_size)
        
        # Create custom legend with assigned colors
        legend_elements = []
        for label in unique_labels:
            legend_elements.append(
                Rectangle((0, 0), 1, 1, fc=colors[label], 
                         label=label, alpha=1)  # Use full opacity for legend
            )
    
    else:
        # Standard coloring for other cases (family type, etc.)
        colors = sns.color_palette('husl', n_colors=len(unique_labels))
        
        # Create scatter plots
        for idx, label in enumerate(unique_labels):
            mask = df[label_column] == label
            ax.scatter(df.loc[mask, 'reduced_vector_d1'],
                      df.loc[mask, 'reduced_vector_d2'],
                      c=[colors[idx]], label=label,
                      alpha=alpha, s=point_size)
        
        # Create custom legend with larger markers
        legend_elements = []
        for idx, label in enumerate(unique_labels):
            legend_elements.append(
                Rectangle((0, 0), 1, 1, fc=colors[idx], 
                         label=label, alpha=1)  # Use full opacity for legend
            )
    
    # Calculate optimal number of columns
    n_cols = optimize_legend_columns(len(unique_labels), fig.get_figheight())
    
    # Add legend with optimized columns
    ax.legend(handles=legend_elements, 
             bbox_to_anchor=(1.05, 1),
             loc='upper left',
             borderaxespad=0.,
             ncol=n_cols)
    
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        logging.info(f"Plot saved to: {output_file}")
    
    plt.close()  # Close the figure to free memory

#@task(log_prints=True)
def load_and_merge_data(vector_file_path, metadata_file_path):
    """Load vector data and merge with metadata"""
    logging.info(f"START TASK - load_and_merge_data for {os.path.basename(vector_file_path)}")
    
    # Load vector data
    logging.info(f"Loading vector data from: {vector_file_path}")
    vectors_df = pd.read_csv(vector_file_path, sep='\t', dtype=np.float32)
    
    # Rename columns to match expected format
    vectors_df.columns = ['reduced_vector_d1', 'reduced_vector_d2']
    
    # Add index to match with metadata
    vectors_df.reset_index(inplace=True)
    vectors_df.rename(columns={'index': 'sequence_index'}, inplace=True)
    
    # Load metadata
    logging.info(f"Loading metadata from: {metadata_file_path}")
    metadata_df = pd.read_csv(metadata_file_path, sep='\t')
    metadata_df.reset_index(inplace=True)
    metadata_df.rename(columns={'index': 'sequence_index'}, inplace=True)
    
    # Merge dataframes
    merged_df = pd.merge(vectors_df, metadata_df, on='sequence_index', how='inner')
    
    logging.info(f"Merged dataset shape: {merged_df.shape}")
    logging.info("END TASK - load_and_merge_data")
    
    return merged_df

#@task(log_prints=True)
def create_plots_for_experiment(experiment_folder_path, metadata_file_path, run_id, output_folder_path, use_combined=False):
    """Create plots for all TSV files in the experiment folder"""
    logging.info("START TASK - create_plots_for_experiment")
    
    # Find all TSV files in experiment folder (excluding metadata files)
    tsv_pattern = os.path.join(experiment_folder_path, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    
    # Filter out any metadata files and focus on vector files
    vector_files = [f for f in tsv_files if 'vectors_tsne' in os.path.basename(f)]
    
    if not vector_files:
        logging.warning(f"No t-SNE vector files found in {experiment_folder_path}")
        return
    
    logging.info(f"Found {len(vector_files)} t-SNE vector files to process")
    
    # Create charts subfolder inside the experiment folder
    charts_folder = os.path.join(experiment_folder_path, "charts")
    os.makedirs(charts_folder, exist_ok=True)
    
    label_columns = ["sequence_family_name", "sequence_family_type"]
    
    for vector_file in vector_files:
        logging.info(f"Processing vector file: {os.path.basename(vector_file)}")
        
        try:
            # Load and merge data
            merged_df = load_and_merge_data(vector_file, metadata_file_path)
            
            # Create plots for each label column
            for label_column in label_columns:
                if label_column not in merged_df.columns:
                    logging.warning(f"Label column '{label_column}' not found in metadata. Skipping.")
                    continue
                
                # Generate output filename
                base_filename = os.path.splitext(os.path.basename(vector_file))[0]
                plot_filename = f"{base_filename}-{label_column}.png"
                plot_output_path = os.path.join(charts_folder, plot_filename)
                
                logging.info(f"Creating plot for {label_column}: {plot_filename}")
                
                # Create the plot
                create_large_scatter_plot(
                    merged_df, 
                    label_column, 
                    output_file=plot_output_path, 
                    point_size=0.5, 
                    alpha=1
                )
                
        except Exception as e:
            logging.error(f"Error processing {vector_file}: {str(e)}")
            continue
    
    logging.info("END TASK - create_plots_for_experiment")

#@flow(name="Create Visualization Plots", log_prints=True)
def create_visualization_plots(input_data_root_path, family_dataset_name, timestamp, filter_name, partition_rule_name, experiment_name, use_combined=False):
    """Main flow to create visualization plots for experiment results"""
    logging.info("START FLOW ******************* Create Visualization Plots *******************")
    
    # Create output structure
    date = utils.get_date_from_formatted_ts(timestamp)
    vector_output_folder_path = os.path.join(input_data_root_path, family_dataset_name, date, "vector_output")
    
    # Determine run ID and metadata file path
    run_id = timestamp + "-" + family_dataset_name + "-" + filter_name + "-" + partition_rule_name
    
    if use_combined:
        run_id_for_metadata = run_id + "-combined"
        metadata_filename = run_id_for_metadata + "-metadata.tsv"
        experiment_folder_name = experiment_name + "-combined"
    else:
        metadata_filename = run_id + "-metadata.tsv"
        experiment_folder_name = experiment_name
    
    metadata_file_path = os.path.join(vector_output_folder_path, metadata_filename)
    
    # Check if metadata file exists
    if not os.path.exists(metadata_file_path):
        if use_combined:
            raise FileNotFoundError(f"Combined metadata file not found: {metadata_file_path}. Please run model_combine_datasets.py first.")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
    
    # Locate experiment folder
    experiments_folder_path = os.path.join(vector_output_folder_path, "experiments")
    experiment_folder_path = os.path.join(experiments_folder_path, experiment_folder_name)
    
    if not os.path.exists(experiment_folder_path):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_folder_path}")
    
    logging.info(f"Processing experiment folder: {experiment_folder_path}")
    logging.info(f"Using metadata file: {metadata_file_path}")
    
    # Create plots for all vector files in the experiment
    create_plots_for_experiment(experiment_folder_path, metadata_file_path, run_id, vector_output_folder_path, use_combined)
    
    logging.info("END FLOW ******************* Create Visualization Plots *******************")

# #######################################
#      MAIN                             #
# #######################################

if __name__ == "__main__":
    # Travels up a level to find the .env
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'model_visualization.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    
    parser = argparse.ArgumentParser(description='Create visualization plots from experiment t-SNE results')
    parser.add_argument('timestamp',
                        help='Run timestamp')
    parser.add_argument('dataset_name', 
                        help='Input protein dataset name')
    parser.add_argument('filter',
                        help='MR Filter')
    parser.add_argument('partition_rule',
                        help='MR partition rule')
    parser.add_argument('experiment_name',
                        help='Experiment folder name (without .json extension)')
    parser.add_argument('--control', 
                        action='store_true',
                        help='Use combined dataset (original + control) for visualization')
    
    # Parse arguments
    args = parser.parse_args()

    # Input parameters
    input_data_root_path = config["INPUT_DATA_ROOT_PATH"]
    
    # Set run data to work on
    family_dataset_name = getattr(dataset_names, args.dataset_name)
    timestamp = args.timestamp
    filter_name = getattr(filters, args.filter).name
    partition_rule_name = getattr(partition_rules, args.partition_rule)["name"]
    experiment_name = args.experiment_name
    
    # Create visualization plots
    create_visualization_plots(
        input_data_root_path, 
        family_dataset_name, 
        timestamp, 
        filter_name, 
        partition_rule_name, 
        experiment_name, 
        args.control
    ) 