# Import required libraries
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime

# Import your existing visualization functions
sys.path.append('model_viz')
from model_viz_enhanced import (
    load_and_merge_data, 
    generate_consistent_color_palette,
    setup_thesis_style,
    format_axes,
    save_figure,
    parse_filename_parameters,
    create_experiment_params_text
)

# Import additional modules for path construction
from dotenv import dotenv_values
sys.path.append('utils')
import utils.utils as utils
from utils.utils import dataset_names, filters, partition_rules

# Setup environment - same as original script
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(''), '.env'))
config = dotenv_values(dotenv_path)

# Get INPUT_DATA_ROOT_PATH from environment
INPUT_DATA_ROOT_PATH = config["INPUT_DATA_ROOT_PATH"]

print("=== CONFIGURATION ===")
print(f"Using INPUT_DATA_ROOT_PATH: {INPUT_DATA_ROOT_PATH}")
print("Please set your experiment parameters below:")

# Create parameter selection widgets
timestamp_input = widgets.Text(
    value='',
    placeholder='e.g., 20241125_102030',
    description='Timestamp:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

dataset_selector = widgets.Dropdown(
    options=[('Family Dataset', 'FAMILY'), ('Test Group', 'TEST_GROUP'), ('Nano Group', 'NANO_GROUP')],
    value='FAMILY',
    description='Dataset:',
    style={'description_width': 'initial'}
)

filter_options = [
    ('None', 'MR_FILTER_NONE'),
    ('Keep Significant', 'MR_FILTER_KEEP_SIGNIFICANT'),
    ('Drop SMR', 'MR_FILTER_DROP_SMR'),
    ('Drop NE', 'MR_FILTER_DROP_NE'),
    ('Drop NN', 'MR_FILTER_DROP_NN'),
    ('Keep Length 4-10', 'MR_FILTER_KEEP_4_10'),
    ('Keep Length 5-7', 'MR_FILTER_KEEP_5_7'),
    ('Keep Length 8', 'MR_FILTER_KEEP_8')
]

filter_selector = widgets.Dropdown(
    options=filter_options,
    value='MR_FILTER_NONE',
    description='MR Filter:',
    style={'description_width': 'initial'}
)

partition_selector = widgets.Dropdown(
    options=[('Use All', 'PARTITION_RULE_USE_ALL')],
    value='PARTITION_RULE_USE_ALL',
    description='Partition:',
    style={'description_width': 'initial'}
)

experiment_input = widgets.Text(
    value='',
    placeholder='e.g., tsne_experiment',
    description='Experiment:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

use_combined_checkbox = widgets.Checkbox(
    value=False,
    description='Use Combined Dataset (with control)',
    style={'description_width': 'initial'}
)

# Export suffix input
export_suffix_input = widgets.Text(
    value='interactive',
    placeholder='e.g., interactive, filtered, custom',
    description='Export Suffix:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# Path construction functions - same logic as original script
def construct_paths(timestamp, dataset_name, filter_name, partition_rule_name, experiment_name, use_combined):
    """Construct all necessary paths using the same logic as the original script."""
    
    # Get actual values from the constants
    family_dataset_name = getattr(dataset_names, dataset_name)
    filter_obj = getattr(filters, filter_name)
    filter_name_str = filter_obj.name
    partition_rule_obj = getattr(partition_rules, partition_rule_name)
    partition_rule_name_str = partition_rule_obj["name"]
    
    # Create output structure
    date = utils.get_date_from_formatted_ts(timestamp)
    vector_output_folder_path = os.path.join(INPUT_DATA_ROOT_PATH, family_dataset_name, date, "vector_output")
    
    # Determine run ID and metadata file path
    run_id = timestamp + "-" + family_dataset_name + "-" + filter_name_str + "-" + partition_rule_name_str
    
    if use_combined:
        run_id_for_metadata = run_id + "-combined"
        metadata_filename = run_id_for_metadata + "-metadata.tsv"
        experiment_folder_name = experiment_name + "-combined"
    else:
        metadata_filename = run_id + "-metadata.tsv"
        experiment_folder_name = experiment_name
    
    metadata_file_path = os.path.join(vector_output_folder_path, metadata_filename)
    
    # Locate experiment folder
    experiments_folder_path = os.path.join(vector_output_folder_path, "experiments")
    experiment_folder_path = os.path.join(experiments_folder_path, experiment_folder_name)
    
    # Charts folder (where exports will go)
    charts_folder_path = os.path.join(experiment_folder_path, "charts")
    
    return {
        'metadata_file_path': metadata_file_path,
        'experiment_folder_path': experiment_folder_path,
        'charts_folder_path': charts_folder_path,
        'run_id': run_id
    }

def find_vector_files(experiment_folder):
    """Find all vector files in the experiment folder."""
    if not os.path.exists(experiment_folder):
        return []
        
    tsv_pattern = os.path.join(experiment_folder, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    
    # Filter for vector files
    vector_files = [f for f in tsv_files if any(method in os.path.basename(f) 
                   for method in ['vectors_tsne', 'vectors_umap'])]
    
    return sorted(vector_files)

def validate_and_load_paths():
    """Validate parameters and load available files."""
    
    # Check if required parameters are filled
    if not timestamp_input.value.strip():
        return {"error": "Please enter a timestamp"}
    
    if not experiment_input.value.strip():
        return {"error": "Please enter an experiment name"}
    
    try:
        # Construct paths
        paths = construct_paths(
            timestamp_input.value.strip(),
            dataset_selector.value,
            filter_selector.value,
            partition_selector.value,
            experiment_input.value.strip(),
            use_combined_checkbox.value
        )
        
        # Check if paths exist
        if not os.path.exists(paths['metadata_file_path']):
            return {"error": f"Metadata file not found: {paths['metadata_file_path']}"}
        
        if not os.path.exists(paths['experiment_folder_path']):
            return {"error": f"Experiment folder not found: {paths['experiment_folder_path']}"}
        
        # Find vector files
        vector_files = find_vector_files(paths['experiment_folder_path'])
        if not vector_files:
            return {"error": f"No vector files found in: {paths['experiment_folder_path']}"}
        
        # Create charts folder if it doesn't exist
        os.makedirs(paths['charts_folder_path'], exist_ok=True)
        
        return {
            "success": True,
            "paths": paths,
            "vector_files": vector_files
        }
        
    except Exception as e:
        return {"error": f"Error constructing paths: {str(e)}"}

print("Path construction functions ready!")

# Interactive plotting function
def create_interactive_scatter_plot(df, label_column, selected_families,
                                  point_size=0.1, alpha=0.5, figsize=(12, 8), background='white',
                                  fixed_limits=None, vertical_line_x=None, plot_order='alpha_asc'):
    """Create scatter plot with only selected families visible, using fixed axis limits for comparison.
    Optionally draw a vertical separation line at x = vertical_line_x when provided.
    Uses stored original labels for consistent color generation across different filters.

    plot_order options:
    - 'alpha_asc': Alphabetical ascending (A-Z)
    - 'alpha_desc': Alphabetical descending (Z-A)
    - 'size_asc': Class size ascending (smallest first, on top)
    - 'size_desc': Class size descending (largest first, on bottom)
    """
    global current_data

    # Setup thesis style
    setup_thesis_style()
    
    # Filter data to only selected families
    if selected_families:
        filtered_df = df[df[label_column].isin(selected_families)].copy()
    else:
        filtered_df = df.copy()
    
    if filtered_df.empty:
        print("No data to display with current selection.")
        return None, None
    
    # Create figure with background color
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    
    # Set background colors
    if background == 'black':
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        # Set text colors for dark theme
        text_color = 'white'
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['top'].set_color(text_color)
        ax.spines['right'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
    elif background == 'gray':
        # Thesis style light gray background from viz_style.py
        fig.patch.set_facecolor('white')  # Keep figure background white
        ax.set_facecolor('#fafafa')  # Light gray plot area
        text_color = '#333333'  # Dark gray text for good contrast
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['top'].set_color(text_color)
        ax.spines['right'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        # Add subtle grid like in thesis style
        ax.grid(True, alpha=0.3, linewidth=0.5, color='white')
    else:  # white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        text_color = 'black'
    
    # Get all unique labels for consistent coloring from stored original labels
    if label_column in original_labels:
        all_unique_labels = original_labels[label_column]
        # print(f"DEBUG: Using stored original labels for {label_column}: {len(all_unique_labels)} labels")
    else:
        # Fallback to current data if original labels not available
        all_unique_labels = sorted(df[label_column].unique())
        # print(f"DEBUG: Fallback - using current df labels for {label_column}: {len(all_unique_labels)} labels")

    # Determine control families using original dataset for consistency
    control_families = None
    if label_column == "sequence_family_name" and "sequence_family_type" in current_data.columns:
        control_families = set()
        for label in all_unique_labels:
            # Use original dataset to determine control families for consistency
            if label in current_data[label_column].values:
                family_data = current_data[current_data[label_column] == label]
                if (family_data["sequence_family_type"] == "control").any():
                    control_families.add(label)

        # print(f"DEBUG: Control families for {label_column}: {sorted(control_families) if control_families else 'None'}")
    
    # Generate consistent color palette for ALL families (not just selected ones)
    colors = generate_consistent_color_palette(all_unique_labels, control_families)
    
    # Create scatter plots for selected families only
    visible_families = list(df[label_column].unique())

    # Sort families based on selected plotting order
    if plot_order == 'alpha_asc':
        # Alphabetical ascending (A-Z)
        visible_families = sorted(visible_families)
    elif plot_order == 'alpha_desc':
        # Alphabetical descending (Z-A)
        visible_families = sorted(visible_families, reverse=True)
    elif plot_order == 'size_asc':
        # Class size ascending (smallest first, will be on top)
        class_sizes = {}
        for label in visible_families:
            mask = df[label_column] == label
            class_sizes[label] = mask.sum()
        visible_families = sorted(visible_families, key=lambda x: class_sizes[x])
    elif plot_order == 'size_desc':
        # Class size descending (largest first, will be on bottom)
        class_sizes = {}
        for label in visible_families:
            mask = df[label_column] == label
            class_sizes[label] = mask.sum()
        visible_families = sorted(visible_families, key=lambda x: class_sizes[x], reverse=True)
    else:
        # Default to alphabetical ascending
        visible_families = sorted(visible_families)

    # Plot families in the determined order
    for label in visible_families:
        mask = filtered_df[label_column] == label
        ax.scatter(filtered_df.loc[mask, 'reduced_vector_d1'],
                  filtered_df.loc[mask, 'reduced_vector_d2'],
                  c=[colors[label]], label=label,
                  alpha=alpha, s=point_size)
    
    # Optional vertical separation line
    if vertical_line_x is not None:
        if background == 'black':
            line_color = 'gold'
            line_alpha = 0.8
        else:
            line_color = '#d62728'  # subtle red on light/gray backgrounds
            line_alpha = 0.6
        ax.axvline(x=vertical_line_x, linestyle='--', linewidth=1.5, color=line_color, alpha=line_alpha, zorder=3)
    
    # Create legend for visible families in alphabetical order (consistent colors)
    if visible_families:
        legend_elements = []
        # Always sort legend alphabetically for consistent color assignment
        legend_families = sorted(visible_families)
        for label in legend_families:
            legend_elements.append(
                Rectangle((0, 0), 1, 1, fc=colors[label],
                         label=label, alpha=1)
            )
        
        # Calculate optimal number of columns
        n_cols = min(2, 1 if len(visible_families) < 24 else len(visible_families))
        
        # Create legend title with plotting order info
        legend_title = get_legend_title(label_column)

        # Add plotting order information to legend title
        if plot_order == 'alpha_asc':
            #legend_title += " (A-Z)"
        elif plot_order == 'alpha_desc':
            #legend_title += " (Z-A)"
        elif plot_order == 'size_asc':
            #legend_title += " (small→large)"
        elif plot_order == 'size_desc':
            #legend_title += " (large→small)"

        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(1.05, 1),
                 loc='upper left',
                 borderaxespad=0.,
                 ncol=n_cols,
                 title=legend_title)
    
    # Format axes
    format_axes(ax, xlabel="Dim 1", ylabel="Dim 2")
    
    # Add selection info with background-appropriate colors
    total_families = len(all_unique_labels)
    selected_count = len(selected_families) if selected_families else total_families
    
    # Set info box colors based on background
    if background == 'black':
        info_bg_color = 'black'
        info_text_color = 'white'
        info_edge_color = 'white'
    elif background == 'gray':
        info_bg_color = 'white'  # White info box on gray background
        info_text_color = '#333333'  # Dark gray text
        info_edge_color = '#333333'  # Dark gray border
    else:  # white
        info_bg_color = 'white'
        info_text_color = 'black'
        info_edge_color = 'gray'
    
    #ax.text(0.02, 0.98, f"Showing {selected_count}/{total_families} families",
    #        transform=ax.transAxes,
    #        fontsize=10,
    #        color=info_text_color,
    #        verticalalignment='top',
    #        bbox=dict(boxstyle='round,pad=0.3', 
    #                 facecolor=info_bg_color, 
    #                 edgecolor=info_edge_color, 
    #                 alpha=0.8))
    
    # Apply fixed axis limits for consistent comparison
    if fixed_limits:
        ax.set_xlim(fixed_limits['xlim'])
        ax.set_ylim(fixed_limits['ylim'])
    else:
        # If no fixed limits provided, ensure symmetric scaling
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        max_range = max(x_max - x_min, y_max - y_min) / 2
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
    
    plt.tight_layout()
    return fig, ax

# Function to get legend title based on label column
def get_legend_title(label_column):
    """Get the appropriate legend title based on the label column."""
    if label_column == 'sequence_family_name':
        return "Familias de proteínas"
    elif label_column == 'sequence_family_type':
        return "Familias de proteínas"
    elif label_column == 'partition_type':
        return "Tipo de representación"
    else:
        return label_column.replace('_', ' ').title()

print("Interactive plotting function ready!")

# Global variables to store current state
current_data = None
current_families = []
current_file = None
current_fig = None
current_paths = None
fixed_axis_limits = None  # Store the fixed axis limits based on full dataset

# Data filter state
data_filter_column = None
data_filter_values = []
data_filter_checkboxes = []

# Store original labels for consistent coloring across all columns
original_labels = {}  # Will store {column_name: [label1, label2, ...]}

# File selection widget (will be populated after loading experiment)
file_selector = widgets.Dropdown(
    options=[('🔄 Run Step 1 first to load available TSV files', '')],
    description='📄 Select TSV:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='800px')
)

# Data filter column selector (filters the dataset before plotting)
data_filter_column_selector = widgets.Dropdown(
    options=[('No Filter', ''), ('Family Name', 'sequence_family_name'), ('Family Type', 'sequence_family_type'), ('Partition Type', 'partition_type')],
    value='',
    description='Filter data by:',
    style={'description_width': 'initial'}
)

# Data filter values container (will hold checkboxes for selected column values)
data_filter_values_container = widgets.VBox([])
data_filter_values_container.layout.display = 'none'  # Hidden until a column is selected

# Label column selector (used for coloring/legend AND additional filtering)
label_selector = widgets.Dropdown(
    options=[('Family Name', 'sequence_family_name'), ('Family Type', 'sequence_family_type'), ('Partition Type', 'partition_type')],
    value='sequence_family_name',
    description='Label/color by:',
    style={'description_width': 'initial'}
)

# Plot parameters
point_size_slider = widgets.FloatSlider(
    value=0.1,
    min=0.01,
    max=2.0,
    step=0.01,
    description='Point size:',
    style={'description_width': 'initial'}
)

alpha_slider = widgets.FloatSlider(
    value=0.5,
    min=0.1,
    max=1.0,
    step=0.05,
    description='Transparency:',
    style={'description_width': 'initial'}
)

background_selector = widgets.Dropdown(
    options=[
        ('⚪ White Background', 'white'),
        ('🔘 Light Gray (Thesis Style)', 'gray'),
        ('⚫ Black Background', 'black')
    ],
    value='gray',  # Default to thesis style
    description='Background:',
    style={'description_width': 'initial'}
)

# Plotting order selector for controlling layer ordering
plot_order_selector = widgets.Dropdown(
    options=[
        ('Alphabetical (A-Z)', 'alpha_asc'),
        ('Alphabetical (Z-A)', 'alpha_desc'),
        ('Class Size (Small-Large)', 'size_asc'),
        ('Class Size (Large-Small)', 'size_desc')
    ],
    value='alpha_asc',  # Default to alphabetical ascending
    description='Plot Order:',
    tooltip='Controls the order in which classes are plotted (affects layering)',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px')
)

# Vertical separation line X coordinate (optional)
vline_x_text = widgets.Text(
    value='',
    placeholder='e.g., 0.0',
    description='Vertical line X:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px')
)

# Control buttons
validate_button = widgets.Button(description="Validate & Load Files", button_style='primary')
load_data_button = widgets.Button(description="Load Selected File", button_style='success')

# Data filter controls
data_filter_select_all_toggle = widgets.ToggleButton(
    value=True,
    description='Select All / Deselect All',
    button_style='info',
    layout=widgets.Layout(width='200px')
)
data_filter_apply_button = widgets.Button(description="🔧 Apply Data Filter", button_style='success')

# Data filter controls container (will hold select all/deselect all and apply buttons)
data_filter_controls_container = widgets.HBox([data_filter_select_all_toggle, data_filter_apply_button])
data_filter_controls_container.layout.display = 'none'  # Hidden until data is loaded

# Family filtering controls
select_all_toggle = widgets.ToggleButton(
    value=True,
    description='Select All / Deselect All',
    button_style='info',
    layout=widgets.Layout(width='200px')
)
apply_filters_button = widgets.Button(description="🎨 Apply Filters & Redraw", button_style='success')
export_button = widgets.Button(description="💾 Export Current View", button_style='success')

# Output areas
info_output = widgets.Output()
plot_output = widgets.Output()

# =====================
# STEP 6: Bounding box count tool (widgets)
# =====================
bbox_heading = widgets.HTML("<b>🧮 STEP 6: Bounding box count tool</b>")

bbox_family_selector = widgets.Dropdown(
    options=[],
    description='Family:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# Corner coordinate inputs (chart coordinates)
tl_x = widgets.FloatText(description='TL x', layout=widgets.Layout(width='150px'))
tl_y = widgets.FloatText(description='TL y', layout=widgets.Layout(width='150px'))
tr_x = widgets.FloatText(description='TR x', layout=widgets.Layout(width='150px'))
tr_y = widgets.FloatText(description='TR y', layout=widgets.Layout(width='150px'))
br_x = widgets.FloatText(description='BR x', layout=widgets.Layout(width='150px'))
br_y = widgets.FloatText(description='BR y', layout=widgets.Layout(width='150px'))
bl_x = widgets.FloatText(description='BL x', layout=widgets.Layout(width='150px'))
bl_y = widgets.FloatText(description='BL y', layout=widgets.Layout(width='150px'))

fill_from_view_button = widgets.Button(description="Use current view corners")
count_bbox_button = widgets.Button(description="Count points in box", button_style='info')

bbox_output = widgets.Output()

bbox_tool_container = widgets.VBox([
    bbox_heading,
    widgets.HBox([bbox_family_selector]),
    widgets.HTML("Define the 4 corner coordinates of the rectangle (chart coordinates):"),
    widgets.HBox([tl_x, tl_y, tr_x, tr_y]),
    widgets.HBox([br_x, br_y, bl_x, bl_y]),
    widgets.HBox([fill_from_view_button, count_bbox_button]),
    bbox_output
])

# Hidden until a dataset is loaded
bbox_tool_container.layout.display = 'none'

print("Widgets created!")

# Function to calculate fixed axis limits based on full dataset
def calculate_fixed_limits(df):
    """Calculate fixed axis limits based on the full dataset for consistent scaling."""
    x_min, x_max = df['reduced_vector_d1'].min(), df['reduced_vector_d1'].max()
    y_min, y_max = df['reduced_vector_d2'].min(), df['reduced_vector_d2'].max()
    
    # Add some padding (5% on each side)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = x_range * 0.05
    y_padding = y_range * 0.05
    
    # Ensure symmetric scaling for equal aspect ratio
    max_range = max(x_range, y_range) / 2
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    return {
        'xlim': (x_mid - max_range - x_padding, x_mid + max_range + x_padding),
        'ylim': (y_mid - max_range - y_padding, y_mid + max_range + y_padding)
    }

# Function to load specific vector file data (full dataset without filtering)
def load_vector_file_data(file_path, metadata_file_path):
    """Load vector file data and return available columns for filtering."""
    global current_data, current_file, fixed_axis_limits, original_labels

    try:
        # Load and merge data
        current_data = load_and_merge_data(file_path, metadata_file_path)
        current_file = file_path

        # Calculate fixed axis limits based on the FULL dataset
        fixed_axis_limits = calculate_fixed_limits(current_data)

        # Store original labels for consistent coloring
        original_labels = {}
        label_columns = ['sequence_family_name', 'sequence_family_type', 'partition_type']
        for col in label_columns:
            if col in current_data.columns:
                original_labels[col] = sorted(current_data[col].unique())
                # print(f"DEBUG: Stored {len(original_labels[col])} original labels for {col}")

        # Return available columns for filtering
        available_columns = []
        filter_columns = ['sequence_family_name', 'sequence_family_type', 'partition_type']
        for col in filter_columns:
            if col in current_data.columns:
                available_columns.append(col)

        return available_columns

    except Exception as e:
        print(f"Error loading data: {e}")
        current_data = None
        current_file = None
        fixed_axis_limits = None
        original_labels = {}
        return []

# Function to apply data filtering to the dataset
def apply_data_filter(data):
    """Apply data filter to the dataset based on selected column and values."""
    if data_filter_column is None or not data_filter_column:
        return data

    selected_values = get_selected_data_filter_values()
    if not selected_values:
        return data

    # Filter the data
    filtered_data = data[data[data_filter_column].astype(str).isin(selected_values)].copy()
    return filtered_data

# Function to apply label filtering to the dataset
def apply_label_filter(data, label_column):
    """Apply label filter to the dataset and return filtered families."""
    if label_column not in data.columns:
        return data, []

    # Get unique families from filtered data
    families = sorted(data[label_column].unique())

    # Apply family selection if checkboxes exist
    selected_families = get_selected_families(family_checkboxes) if family_checkboxes else families

    if selected_families:
        filtered_data = data[data[label_column].isin(selected_families)].copy()
    else:
        filtered_data = data.copy()

    return filtered_data, families

# Function to get filtered data for plotting
def get_filtered_data_for_plotting(label_column):
    """Get the fully filtered data for plotting (data filter → label filter)."""
    if current_data is None:
        return None, []

    # Apply data filter first
    data_filtered = apply_data_filter(current_data)

    # Apply label filter second
    plot_data, all_families = apply_label_filter(data_filtered, label_column)

    return plot_data, all_families

# Function to create family checkboxes
def create_family_checkboxes(families):
    """Create checkbox widgets for family selection."""
    checkboxes = []

    for family in families:
        checkbox = widgets.Checkbox(
            value=True,  # Start with all selected
            description=family,
            layout=widgets.Layout(width='200px')
        )
        checkboxes.append(checkbox)

    return checkboxes

# Function to get selected families from checkboxes
def get_selected_families(checkboxes):
    """Get list of selected families from checkboxes."""
    return [cb.description for cb in checkboxes if cb.value]

# Function to create data filter value checkboxes
def create_data_filter_checkboxes(values):
    """Create checkbox widgets for data filter values."""
    global data_filter_checkboxes
    checkboxes = []

    for value in values:
        checkbox = widgets.Checkbox(
            value=True,  # Start with all selected
            description=str(value),
            layout=widgets.Layout(width='200px')
        )
        checkboxes.append(checkbox)

    data_filter_checkboxes = checkboxes
    return checkboxes

# Function to get selected data filter values from checkboxes
def get_selected_data_filter_values():
    """Get list of selected data filter values from checkboxes."""
    return [cb.description for cb in data_filter_checkboxes if cb.value]

# Function to update data filter checkboxes when column changes
def initialize_data_filter_controls():
    """Initialize data filter controls to default state."""
    global data_filter_column, data_filter_values, original_labels
    data_filter_column_selector.value = ''
    data_filter_values_container.children = []
    data_filter_values_container.layout.display = 'none'
    data_filter_controls_container.layout.display = 'none'
    data_filter_column = None
    data_filter_values = []
    original_labels = {}

def update_data_filter_checkboxes():
    """Update the data filter checkboxes based on selected column."""
    global data_filter_column, data_filter_values

    column = data_filter_column_selector.value

    if not column or current_data is None:
        data_filter_values_container.children = []
        data_filter_values_container.layout.display = 'none'
        data_filter_controls_container.layout.display = 'none'
        data_filter_column = None
        data_filter_values = []
        return

    # Get unique values for the selected column
    if column in current_data.columns:
        values = sorted(current_data[column].unique())
        checkboxes = create_data_filter_checkboxes(values)

        # Display checkboxes in columns
        n_cols = 3
        items_per_col = len(checkboxes) // n_cols + (1 if len(checkboxes) % n_cols else 0)
        checkbox_columns = []

        for i in range(0, len(checkboxes), items_per_col):
            column_boxes = checkboxes[i:i + items_per_col]
            checkbox_columns.append(widgets.VBox(column_boxes))

        checkbox_grid = widgets.HBox(checkbox_columns)
        data_filter_values_container.children = [checkbox_grid]
        data_filter_values_container.layout.display = ''
        data_filter_controls_container.layout.display = ''

        data_filter_column = column
        data_filter_values = values
    else:
        data_filter_values_container.children = []
        data_filter_values_container.layout.display = 'none'
        data_filter_column = None
        data_filter_values = []

def update_family_checkboxes_from_data_filter():
    """Update family checkboxes when data filter changes."""
    global family_checkboxes

    if current_data is None or not label_selector.value:
        return

    # Get filtered data based on current data filter
    filtered_data, _ = get_filtered_data_for_plotting(label_selector.value)

    if filtered_data is None or filtered_data.empty:
        family_checkboxes = []
        return

    # Get available families from filtered data
    if label_selector.value in filtered_data.columns:
        available_families = sorted(filtered_data[label_selector.value].unique())

        # Create new checkboxes for available families
        family_checkboxes = create_family_checkboxes(available_families)

        # Display checkboxes in columns
        n_cols = 3
        items_per_col = len(family_checkboxes) // n_cols + (1 if len(family_checkboxes) % n_cols else 0)
        checkbox_columns = []

        for i in range(0, len(family_checkboxes), items_per_col):
            column_boxes = family_checkboxes[i:i + items_per_col]
            checkbox_columns.append(widgets.VBox(column_boxes))

        checkbox_grid = widgets.HBox(checkbox_columns)

        # Update the info output to show the family selection
        with info_output:
            clear_output(wait=True)
            print(f"✅ Updated family selection based on data filter")
            print(f"Available families for '{label_selector.value}': {', '.join(available_families[:5])}{'...' if len(available_families) > 5 else ''}")
            display(checkbox_grid)
            print("\n🎛️ Filter Controls:")
            display(widgets.HBox([select_all_toggle, apply_filters_button]))
    else:
        family_checkboxes = []

# =====================
# STEP 6 helpers
# =====================

def _fill_bbox_from_current_view():
    """Fill the 4 corner boxes from current figure axis limits."""
    global current_fig
    if current_fig is None:
        return
    ax = current_fig.axes[0] if current_fig.axes else None
    if ax is None:
        return
    (xmin, xmax) = ax.get_xlim()
    (ymin, ymax) = ax.get_ylim()
    # top-left (xmin, ymax), top-right (xmax, ymax), bottom-right (xmax, ymin), bottom-left (xmin, ymin)
    tl_x.value, tl_y.value = float(xmin), float(ymax)
    tr_x.value, tr_y.value = float(xmax), float(ymax)
    br_x.value, br_y.value = float(xmax), float(ymin)
    bl_x.value, bl_y.value = float(xmin), float(ymin)


def _count_points_in_bbox():
    """Count points for selected family inside axis-aligned rectangle defined by corners."""
    if current_data is None:
        return 0
    if bbox_family_selector.value is None or bbox_family_selector.value == '':
        return 0

    # Get filtered data for current selections
    plot_data, _ = get_filtered_data_for_plotting(label_selector.value)
    if plot_data is None or plot_data.empty:
        return 0

    # compute axis-aligned bounds from given corners
    xs = [tl_x.value, tr_x.value, br_x.value, bl_x.value]
    ys = [tl_y.value, tr_y.value, br_y.value, bl_y.value]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # filter by family and bounds
    fam_col = label_selector.value
    df_fam = plot_data[plot_data[fam_col] == bbox_family_selector.value]
    if df_fam.empty:
        return 0
    in_x = (df_fam['reduced_vector_d1'] >= xmin) & (df_fam['reduced_vector_d1'] <= xmax)
    in_y = (df_fam['reduced_vector_d2'] >= ymin) & (df_fam['reduced_vector_d2'] <= ymax)
    return int((in_x & in_y).sum())

print("Helper functions ready!")

# Main interaction logic
family_checkboxes = []

def on_validate_button_clicked(b):
    """Handle validate button click - check paths and load available files."""
    global current_paths
    
    with info_output:
        clear_output()
        print("Validating parameters and loading files...")
        
        result = validate_and_load_paths()
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Success - store paths and update file selector
        current_paths = result["paths"]
        vector_files = result["vector_files"]
        
        # Update file selector with available files - create descriptive labels
        file_options = []
        for f in vector_files:
            basename = os.path.basename(f)
            # Parse parameters to create more descriptive labels
            try:
                params = parse_filename_parameters(f)
                if params['type'] == 'tsne':
                    label = f"📊 t-SNE (perp={params['perplexity']}, lr={params['learning_rate']}) - {basename}"
                elif params['type'] == 'umap':
                    label = f"🗺️ UMAP (neighbors={params['n_neighbors']}, dist={params['min_dist']}) - {basename}"
                else:
                    label = f"📄 {basename}"
            except:
                label = f"📄 {basename}"
            
            file_options.append((label, f))
        
        file_selector.options = file_options
        file_selector.value = vector_files[0] if vector_files else ''
        
        print(f"✅ Success! Found {len(vector_files)} vector TSV files:")
        
        # Categorize files by type for better display
        tsne_files = [f for f in vector_files if 'tsne' in os.path.basename(f)]
        umap_files = [f for f in vector_files if 'umap' in os.path.basename(f)]
        
        if tsne_files:
            print(f"\n   📊 t-SNE files ({len(tsne_files)}):")
            for i, file in enumerate(tsne_files[:3]):
                print(f"     • {os.path.basename(file)}")
            if len(tsne_files) > 3:
                print(f"     ... and {len(tsne_files)-3} more t-SNE files")
        
        if umap_files:
            print(f"\n   🗺️ UMAP files ({len(umap_files)}):")
            for i, file in enumerate(umap_files[:3]):
                print(f"     • {os.path.basename(file)}")
            if len(umap_files) > 3:
                print(f"     ... and {len(umap_files)-3} more UMAP files")
            
        print(f"\n📁 Paths configured:")
        print(f"  Experiment folder: {current_paths['experiment_folder_path']}")
        print(f"  Metadata file: {current_paths['metadata_file_path']}")
        print(f"  Export folder: {current_paths['charts_folder_path']}")
        print(f"\n➡️ Step 2: Select a specific TSV file from the dropdown above and click 'Load Selected File'")

def on_load_data_button_clicked(b):
    """Handle load data button click - load specific vector file."""
    global family_checkboxes

    with info_output:
        clear_output()

        if current_paths is None:
            print("❌ Please validate parameters first!")
            return

        if not file_selector.value:
            print("❌ Please select a vector file!")
            return

        print("Loading vector file data...")

        # Load full dataset (no filtering yet)
        available_columns = load_vector_file_data(
            file_selector.value,
            current_paths['metadata_file_path']
        )

        if available_columns:
            print(f"✅ Loaded dataset with {len(current_data)} points")
            print(f"Available filter columns: {', '.join(available_columns)}")

            # Initialize data filter checkboxes for the current data
            update_data_filter_checkboxes()

            # Show data filter controls
            data_filter_controls_container.layout.display = ''

            # Get families for the current label column (after data filtering)
            plot_data, families = get_filtered_data_for_plotting(label_selector.value)

            if families:
                print(f"✅ Found {len(families)} families for label column '{label_selector.value}': {', '.join(families[:5])}{'...' if len(families) > 5 else ''}")

                # Create new family checkboxes
                family_checkboxes = create_family_checkboxes(families)

                # Display checkboxes in columns
                n_cols = 3
                items_per_col = len(family_checkboxes) // n_cols + (1 if len(family_checkboxes) % n_cols else 0)
                checkbox_columns = []

                for i in range(0, len(family_checkboxes), items_per_col):
                    column_boxes = family_checkboxes[i:i + items_per_col]
                    checkbox_columns.append(widgets.VBox(column_boxes))

                checkbox_grid = widgets.HBox(checkbox_columns)

                # DON'T connect checkbox changes to plot update - user will click Apply instead

                print("\n🔘 Family Selection (click to toggle, then click 'Apply Filters'):")
                display(checkbox_grid)

                print("\n🎛️ Filter Controls:")
                display(widgets.HBox([select_all_toggle, apply_filters_button]))

                # Populate STEP 6 family dropdown and show tool
                bbox_family_selector.options = families
                bbox_family_selector.value = families[0] if families else None
                bbox_tool_container.layout.display = ''

                # Initial plot with all families selected
                update_plot()
            else:
                print(f"❌ No families found for label column '{label_selector.value}'. Check data format.")
        else:
            print("❌ Failed to load data. Check file paths and data format.")

def update_plot():
    """Update the plot based on current selections."""
    global current_fig

    if current_data is None:
        return

    with plot_output:
        clear_output(wait=True)

        # Get filtered data for plotting (applies both data filter and label filter)
        plot_data, all_families = get_filtered_data_for_plotting(label_selector.value)

        if plot_data is None or plot_data.empty:
            print("No data to display with current filter selections.")
            return

        # Get selected families from the label filter
        selected_families = get_selected_families(family_checkboxes) if family_checkboxes else all_families

        # Parse vertical line X value (optional)
        vline_value = None
        try:
            text_val = vline_x_text.value.strip()
            if text_val != '':
                vline_value = float(text_val)
        except Exception:
            vline_value = None

        # Create plot with fixed axis limits for consistent comparison
        current_fig, ax = create_interactive_scatter_plot(
            plot_data,
            label_selector.value,
            selected_families,
            point_size=point_size_slider.value,
            alpha=alpha_slider.value,
            background=background_selector.value,
            fixed_limits=fixed_axis_limits,
            vertical_line_x=vline_value,
            plot_order=plot_order_selector.value
        )

        if current_fig:
            plt.show()

def on_select_all_toggle_changed(change):
    """Handle select all/deselect all toggle."""
    if family_checkboxes:
        new_value = change['new']
        for checkbox in family_checkboxes:
            checkbox.value = new_value
        
        # Update button text to reflect current state
        if new_value:
            select_all_toggle.description = 'Deselect All'
        else:
            select_all_toggle.description = 'Select All'

def on_apply_filters_clicked(b):
    """Apply current filter selections and redraw plot."""
    update_plot()

def on_data_filter_select_all_toggle_changed(change):
    """Handle data filter select all/deselect all toggle."""
    if data_filter_checkboxes:
        new_value = change['new']
        for checkbox in data_filter_checkboxes:
            checkbox.value = new_value

        # Update button text to reflect current state
        if new_value:
            data_filter_select_all_toggle.description = 'Deselect All'
        else:
            data_filter_select_all_toggle.description = 'Select All'

def on_data_filter_apply_clicked(b):
    """Apply data filter selections and update family checkboxes."""
    # Get the current column selection directly from the widget
    current_column = data_filter_column_selector.value

    if not current_column:
        print("No data filter column selected")
        return

    # Update the global variable with the current selection
    global data_filter_column
    data_filter_column = current_column

    print(f"Data filter apply clicked - filtering by column: {current_column}")
    print(f"Selected values: {get_selected_data_filter_values()}")

    # When data filter changes, we need to update the family checkboxes
    # because the available families might have changed
    update_family_checkboxes_from_data_filter()
    update_plot()

def on_export_clicked(b):
    """Export current view to the charts folder."""
    if current_fig is None:
        with info_output:
            print("❌ No plot to export. Please load data first.")
        return
    
    if current_paths is None:
        with info_output:
            print("❌ No export path configured. Please validate parameters first.")
        return
    
    # Generate filename with custom suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(current_file))[0] if current_file else "plot"
    selected_families = get_selected_families(family_checkboxes)
    n_selected = len(selected_families)
    suffix = export_suffix_input.value.strip() or "interactive"
    
    filename = f"{base_name}_{suffix}_{n_selected}families_{timestamp}.png"
    filepath = os.path.join(current_paths['charts_folder_path'], filename)
    
    # Save figure
    try:
        save_figure(current_fig, filepath)
        with info_output:
            print(f"✅ Exported: {filename}")
            print(f"📁 Location: {filepath}")
    except Exception as e:
        with info_output:
            print(f"❌ Export failed: {e}")

# STEP 6 events

def on_fill_from_view_clicked(b):
    _fill_bbox_from_current_view()


def on_count_bbox_clicked(b):
    with bbox_output:
        clear_output()
        count = _count_points_in_bbox()
        fam = bbox_family_selector.value
        print(f"Found {count} points for family '{fam}' inside the defined box.")

# Connect button events
validate_button.on_click(on_validate_button_clicked)
load_data_button.on_click(on_load_data_button_clicked)
data_filter_select_all_toggle.observe(on_data_filter_select_all_toggle_changed, names='value')
data_filter_apply_button.on_click(on_data_filter_apply_clicked)
select_all_toggle.observe(on_select_all_toggle_changed, names='value')
apply_filters_button.on_click(on_apply_filters_clicked)
export_button.on_click(on_export_clicked)
fill_from_view_button.on_click(on_fill_from_view_clicked)
count_bbox_button.on_click(on_count_bbox_clicked)

# Connect parameter changes to plot update (these are lightweight so keep auto-update)
point_size_slider.observe(lambda change: update_plot() if current_data is not None else None, names='value')
alpha_slider.observe(lambda change: update_plot() if current_data is not None else None, names='value')
background_selector.observe(lambda change: update_plot() if current_data is not None else None, names='value')
vline_x_text.observe(lambda change: update_plot() if current_data is not None else None, names='value')
plot_order_selector.observe(lambda change: update_plot() if current_data is not None else None, names='value')
label_selector.observe(lambda change: on_load_data_button_clicked(None) if current_paths else None, names='value')

# Connect data filter changes
data_filter_column_selector.observe(lambda change: update_data_filter_checkboxes(), names='value')

# Initialize data filter controls
initialize_data_filter_controls()

print("Event handlers connected!")

# Display the main interface
print("=== INTERACTIVE PROTEIN FAMILY VISUALIZATION EXPLORER ===")
print("(Parameter widgets are shown above in the configuration cell)")


print("\\nParameter Selection:")

# Display the parameter widgets immediately
display(widgets.VBox([
    widgets.HTML("<b>Experiment Parameters:</b>"),
    widgets.HBox([timestamp_input, dataset_selector]),
    widgets.HBox([filter_selector, partition_selector]),
    widgets.HBox([experiment_input, use_combined_checkbox]),
    widgets.HTML("<br><b>Export Settings:</b>"),
    export_suffix_input
]))

print("\n📂 STEP 1: Validate experiment parameters and discover available TSV files")
display(validate_button)

print("\n📄 STEP 2: Choose which TSV file to visualize and filter")
print("   (This dropdown will be populated after Step 1 with all available .tsv vector files)")
display(file_selector)

print("\n🎛️ STEP 3: Load the selected TSV and set visualization parameters")
display(load_data_button)
display(widgets.HBox([point_size_slider, alpha_slider, background_selector, vline_x_text]))

print("\n🔧 STEP 4: Data filtering (optional - filters dataset before plotting)")
print("   (This section will be populated after loading data)")
display(widgets.VBox([
    widgets.HTML("<b>Data Filter:</b>"),
    data_filter_column_selector,
    data_filter_values_container,
    widgets.HTML("<b>Data Filter Controls:</b>"),
    data_filter_controls_container
]))

print("\n🏷️ STEP 5: Label column and family filtering")
print("   (Select which column to use for coloring and legend)")
display(label_selector)
display(widgets.HBox([plot_order_selector, widgets.HTML("<small>Controls plotting order (affects layer visibility)</small>")]))
display(export_button)

print("\n📊 STATUS & FAMILY SELECTION:")
display(info_output)

print("\n📈 VISUALIZATION:")
display(plot_output)

print("\n🧮 STEP 6: Bounding box count tool (appears after loading data)")
display(bbox_tool_container)
