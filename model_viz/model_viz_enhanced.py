"""
Enhanced Model Visualization Script
==================================

This module provides enhanced visualization for t-SNE and UMAP experiment results
with improved styling, consistent coloring, and flexible layout options.

Features:
- Integrated thesis styling for consistent, professional plots
- Alphabetical ordering of protein families for consistent coloring
- Single and grid visualization modes
- Separate experiment parameter legend box
- Support for both family_name and family_type chart types
"""

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
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
import math
import colorcet as cc


from utils.utils import dataset_names, filters, partition_rules
import utils.utils as utils

# ===============================
# GLOBAL COLOR PALETTE
# ===============================

# Custom color palette with specific hex colors
CUSTOM_COLOR_PALETTE = [
    '#ea0202',
    '#ffa087',
    '#a98585',  
    '#d43c00',
    '#ec4300',
    '#db6630',
    '#cf6300',
    '#864d18',
    '#904500',
    '#60543e',
    "#201d16",
    '#ff9500',
    '#ff9a14',
    '#fbba00',
    '#ffff0e',
    '#a89a03',
    '#d4f75b', 
    '#839f77',
    '#79e209',
    '#01ac00',
    '#01ff01',
    '#009524',
    '#009d37',
    '#46856a',
    '#3aeba5',
    '#1e8e78',
    '#02e2c7',
    '#009ba0',
    '#006869',
    '#00618c', 
    '#4ccdff',
    '#52a9f0',
    '#4faaf1',
    '#0200fc',
    '#9587ff',
    '#542d9a',
    '#7e00c5',
    '#c82bd2', 
    '#ff00ff', 
    '#fe30c3',
    '#ff00bb',
    '#b5569a', 
    '#96258d',  
    '#d4002d',
    '#bf3362',
    '#f80079'
    
]

# Convert hex colors to RGB tuples for matplotlib
def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple with values between 0-1."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

# Convert all hex colors to RGB tuples
CUSTOM_COLOR_PALETTE_RGB = [hex_to_rgb(color) for color in CUSTOM_COLOR_PALETTE]

# ===============================
# INTEGRATED THESIS STYLING
# ===============================

# ===============================
# INTEGRATED THESIS STYLING
# ===============================

def setup_thesis_style():
    """
    Configure matplotlib for consistent styling that matches the original script.
    This function should be called at the beginning of all plotting scripts.
    """
    # Use minimal styling to match original - no seaborn style
    
    # Configure matplotlib parameters for consistency
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (12, 8),  # Match original default
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        
        # Font settings - match original
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
        
        # Layout
        'axes.titlepad': 20,
        'axes.labelpad': 10,
    })

def get_thesis_colors():
    """
    Return a consistent color palette for thesis charts - modern scientific style.
    
    Returns:
        dict: Dictionary with color schemes for different chart types
    """
    return {
        'primary': '#3498db',      # Modern blue
        'secondary': '#e74c3c',    # Modern red
        'tertiary': '#2ecc71',     # Modern green
        'quaternary': '#f39c12',   # Modern orange
        'success': '#27ae60',      # Success green
        'warning': '#f39c12',      # Warning orange
        'info': '#3498db',         # Info blue
        'neutral': '#95a5a6',      # Modern gray
        
        # Modern color palettes for different chart types
        'categorical': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#34495e', '#e67e22', '#95a5a6'],
        'sequential': 'viridis',
        'diverging': 'RdBu_r',
        'qualitative': 'Set2',
        'single_color': '#3498db',  # For single-series plots
        
        # Specific modern colors for scientific plots
        'uniform_bar': '#fdeaea',   # Light pink for uniform baseline
        'natural_line': '#e74c3c'   # Vibrant red for natural data
    }

def save_figure(fig, filename, format='png', **kwargs):
    """
    Save figure with consistent settings for thesis.
    
    Args:
        fig: matplotlib figure object
        filename (str): output filename
        format (str): file format ('png', 'pdf', 'svg')
        **kwargs: additional arguments for savefig
    """
    default_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    default_kwargs.update(kwargs)
    
    fig.savefig(filename, format=format, **default_kwargs)
    print(f"Figure saved: {filename}")

def format_axes(ax, title=None, xlabel=None, ylabel=None, 
                title_fontweight='bold', label_fontweight='normal'):
    """
    Apply consistent formatting to axes.
    
    Args:
        ax: matplotlib axes object
        title (str): plot title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title_fontweight (str): font weight for title
        label_fontweight (str): font weight for axis labels
    """
    if title:
        ax.set_title(title, fontweight=title_fontweight, pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight=label_fontweight)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight=label_fontweight)

# ===============================
# PARAMETER PARSING
# ===============================

def parse_filename_parameters(filename):
    """
    Parse experiment parameters from filename.
    Supports both t-SNE and UMAP formats.
    """
    # Remove file extension and get the base filename
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Split parameters by '-'
    params = base_filename.split('-')
    

    
    # Common parameters
    timestamp = params[0]
    dataset = params[1]
    filter_type = params[2].replace('filter_', '')
    partition = params[3].replace('partition_', '')
    vectors_method = params[4].replace('vectors_', '')
    
    if 'tsne' in vectors_method:
        # t-SNE format: method-perplexity-learning_rate-max_iterations-random_state
        # Handle variable parameter lengths safely
        method = params[5] if len(params) > 5 else 'unknown'
        perplexity = params[6] if len(params) > 6 else 'unknown'
        learning_rate = params[7] if len(params) > 7 else 'unknown'
        max_iterations = params[8] if len(params) > 8 else 'unknown'
        random_state = params[9] if len(params) > 9 else 'unknown'
        
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
            'random_state': random_state,
            'type': 'tsne'
        }
    
    elif 'umap' in vectors_method:
        # UMAP format: metric-n_neighbors-learning_rate-n_epochs-min_dist-spread-random_state
        # Handle variable parameter lengths safely
        metric = params[5] if len(params) > 5 else 'unknown'
        n_neighbors = params[6] if len(params) > 6 else 'unknown'
        learning_rate = params[7] if len(params) > 7 else 'unknown'
        n_epochs = params[8] if len(params) > 8 else 'unknown'
        min_dist = params[9] if len(params) > 9 else 'unknown'
        spread = params[10] if len(params) > 10 else 'unknown'
        random_state = params[11] if len(params) > 11 else 'unknown'
        
        return {
            'timestamp': timestamp,
            'dataset': dataset,
            'filter': filter_type,
            'partition': partition,
            'vectors_method': vectors_method,
            'metric': metric,
            'n_neighbors': n_neighbors,
            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'min_dist': min_dist,
            'spread': spread,
            'random_state': random_state,
            'type': 'umap'
        }
    
    else:
        # Default/unknown format
        return {
            'timestamp': timestamp,
            'dataset': dataset,
            'filter': filter_type,
            'partition': partition,
            'vectors_method': vectors_method,
            'type': 'unknown'
        }

def create_experiment_params_text(params):
    """
    Create formatted text for experiment parameters.
    
    Args:
        params (dict): Parsed parameters from filename
        
    Returns:
        str: Formatted parameter text
    """
    if params['type'] == 'tsne':
        return (f"t-SNE Parameters:\n"
                f"Method: {params['method']}\n"
                f"Perplexity: {params['perplexity']}\n"
                f"Learning Rate: {params['learning_rate']}\n"
                f"Max Iter: {params['max_iterations']}\n"
                f"Filter: {params['filter']}\n"
                f"Partition: {params['partition']}")
    
    elif params['type'] == 'umap':
        return (f"UMAP Parameters:\n"
                f"Metric: {params['metric']}\n"
                f"n_neighbors: {params['n_neighbors']}\n"
                f"Learning Rate: {params['learning_rate']}\n"
                f"n_epochs: {params['n_epochs']}\n"
                f"min_dist: {params['min_dist']}\n"
                f"Filter: {params['filter']}\n"
                f"Partition: {params['partition']}")
    
    else:
        return (f"Experiment Parameters:\n"
                f"Filter: {params['filter']}\n"
                f"Partition: {params['partition']}")

# ===============================
# LABEL MAPPING FUNCTIONS
# ===============================

def map_family_type_labels(df, label_column):
    """
    Map family type labels to Spanish equivalents for better visualization.
    
    This function specifically handles the mapping for sequence_family_type labels:
    - "fibrilar" -> "fibrosa"
    - "repetitive" -> "repetitiva"
    
    Args:
        df (DataFrame): Input dataframe containing the label column
        label_column (str): Name of the column to apply mapping to
        
    Returns:
        DataFrame: Dataframe with mapped labels
    """
    # Only apply mapping for family type columns
    if label_column != "sequence_family_type":
        return df
    
    # Create a copy to avoid modifying the original dataframe
    df_mapped = df.copy()
    
    # Define the mapping dictionary
    type_mapping = {
        "fibrilar": "fibrosa",
        "repetitive": "repetitiva"
    }
    
    # Apply the mapping
    df_mapped[label_column] = df_mapped[label_column].map(type_mapping).fillna(df_mapped[label_column])
    
    logging.info(f"Applied label mapping for {label_column}: {type_mapping}")
    
    return df_mapped

# ===============================
# COLOR AND LEGEND MANAGEMENT
# ===============================

def sort_colors_by_hue(colors):
    """
    Sort a list of RGB colors by their hue value with improved natural progression.
    
    This enhanced version creates a more visually pleasing color progression by:
    1. Using HSL color space instead of HSV for better perceptual ordering
    2. Handling special cases like very dark/light colors
    3. Ensuring smooth transitions between similar hues
    
    Args:
        colors (list): List of RGB color tuples
        
    Returns:
        list: Same colors sorted in a natural, visually pleasing hue order
    """
    import colorsys
    import numpy as np
    
    # Convert RGB to HSL and store original indices
    hsl_colors = []
    for i, rgb in enumerate(colors):
        h, l, s = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
        
        # Calculate color intensity and saturation metrics
        intensity = l
        saturation = s
        
        # Store as (hue, lightness, saturation, intensity, original_index)
        hsl_colors.append((h, l, s, intensity, i))
    
    # First separate grayscale (very low saturation) from colorful colors
    gray_threshold = 0.15  # Threshold for considering a color as "gray"
    grays = [c for c in hsl_colors if c[2] < gray_threshold]
    colors_hsl = [c for c in hsl_colors if c[2] >= gray_threshold]
    
    # Sort colorful colors by hue
    colors_hsl.sort(key=lambda x: x[0])
    
    # Sort grays by lightness (dark to light)
    grays.sort(key=lambda x: x[1])
    
    # Combine: first colorful colors in hue order, then grays from dark to light
    sorted_indices = [i for _, _, _, _, i in colors_hsl + grays]
    
    # Return RGB colors in the new order
    return [colors[i] for i in sorted_indices]

def generate_improved_color_palette(n_colors):
    """
    Generate an improved color palette with better perceptual spacing and ordering.
    
    This creates a palette that:
    1. Has good perceptual separation between colors
    2. Follows a natural hue progression
    3. Maintains good contrast and readability
    
    Args:
        n_colors (int): Number of colors needed
        
    Returns:
        list: List of RGB color tuples
    """
    import colorsys
    import numpy as np
    
    if n_colors <= 1:
        return [(0.2, 0.4, 0.8)]  # Default blue
    
    # Start with glasbey_light for good distinctness
    base_palette = sns.color_palette(cc.glasbey_light, n_colors=n_colors)
    
    # Sort by hue for natural progression
    sorted_palette = sort_colors_by_hue(base_palette)
        
    return sorted_palette
    

def generate_consistent_color_palette(labels, control_families=None):
    """
    Generate a consistent color palette for labels with alphabetical ordering.
    Colors are sorted by hue for easier legend navigation.
    
    Args:
        labels (list): List of unique labels
        control_families (set, optional): Set of control family names
        
    Returns:
        dict: Mapping of labels to colors
    """
    # Sort labels alphabetically to ensure consistency
    sorted_labels = sorted(labels)
    
    # Separate control and non-control families if control_families is provided
    if control_families:
        control_labels = [label for label in sorted_labels if label in control_families]
        non_control_labels = [label for label in sorted_labels if label not in control_families]
        
        colors = {}
        
        # Assign colorful colors to non-control families
        if non_control_labels:
            # Use improved color palette for non-control families
            colorful_palette = generate_improved_color_palette(len(non_control_labels))
            for idx, family in enumerate(non_control_labels):  # Already sorted
                colors[family] = colorful_palette[idx]
        
        # Assign muted colors to control families for better distinction
        if control_labels:
            # Use a set of muted, distinguishable colors instead of pure grayscale
            # This provides better visual separation while maintaining the "control" aesthetic
            muted_colors = [
                '#8c8c8c',  # Medium gray
                '#7d94b5',  # Muted blue-gray
                '#9caf88',  # Muted green-gray
                '#b5999c',  # Muted rose-gray
                '#c4a573',  # Muted gold-gray
                '#8fb3c7',  # Muted blue
                '#a8c09a',  # Muted green
                '#c19fa8',  # Muted pink
                '#d4b896',  # Muted tan
                '#9fb3c7',  # Another muted blue shade
            ]

            # For control families, use grayscale with good separation
            #lightness_values = np.linspace(0.3, 0.7, len(control_labels))
            #gray_palette = [(l, l, l) for l in lightness_values]
            # If we have more control families than colors, cycle through the palette
            for idx, family in enumerate(control_labels):  # Already sorted
                colors[family] = muted_colors[idx % len(muted_colors)]
                #colors[family] = gray_palette[idx]

        return colors
    
    else:
        # Standard coloring for all labels
        color_palette = generate_improved_color_palette(len(sorted_labels))
        
        colors = {}
        for idx, label in enumerate(sorted_labels):
            colors[label] = color_palette[idx]
        
        return colors

def optimize_legend_columns(num_labels, fig_height):
    """
    Calculate optimal number of legend columns based on figure height.
    
    Args:
        num_labels (int): Number of legend entries
        fig_height (float): Figure height in inches
        
    Returns:
        int: Optimal number of columns
    """
    # Approximate height of each legend entry in inches
    entry_height = 0.25
    # Available height in inches (considering figure height and margins)
    available_height = fig_height * 0.9
    # Maximum entries per column
    max_entries_per_column = math.floor(available_height / entry_height)
    # Calculate optimal number of columns
    return math.ceil(num_labels / max_entries_per_column)

# ===============================
# CORE PLOTTING FUNCTIONS
# ===============================

def create_enhanced_scatter_plot(df, label_column, output_file=None, point_size=0.1, alpha=0.5, 
                                xlim=None, ylim=None, show_params=False, legend_title=None):
    """
    Create an enhanced scatter plot with consistent styling and improved legends.
    
    Args:
        df (DataFrame): Data containing reduced vectors and labels
        label_column (str): Column name to use for coloring
        output_file (str, optional): Path to save the plot
        point_size (float): Size of scatter points
        alpha (float): Transparency of points
        xlim (tuple, optional): X-axis limits
        ylim (tuple, optional): Y-axis limits
        show_params (bool): Whether to show experiment parameters
        
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Setup thesis style
    setup_thesis_style()
    
    # Apply label mapping for family type columns
    df = map_family_type_labels(df, label_column)
    
    # Create figure with appropriate size - use square aspect ratio to prevent distortion
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Set equal aspect ratio to prevent distortion, but allow flexibility for layout
    ax.set_aspect('equal')
    
    # Get unique labels and sort alphabetically
    unique_labels = sorted(df[label_column].unique())
    
    # Determine if we have control families for special handling
    control_families = None
    if label_column == "sequence_family_name" and "sequence_family_type" in df.columns:
        control_families = set()
        # Apply mapping to family type column for control detection
        df_mapped_types = map_family_type_labels(df, "sequence_family_type")
        for label in unique_labels:
            family_data = df[df[label_column] == label]
            if not family_data.empty and (df_mapped_types[df_mapped_types[label_column] == label]["sequence_family_type"] == "control").any():
                control_families.add(label)
    
    # Generate consistent color palette
    colors = generate_consistent_color_palette(unique_labels, control_families)
    
    # Create scatter plots with alphabetically consistent colors
    for label in unique_labels:
        mask = df[label_column] == label
        ax.scatter(df.loc[mask, 'reduced_vector_d1'],
                  df.loc[mask, 'reduced_vector_d2'],
                  c=[colors[label]], label=label,
                  alpha=alpha, s=point_size)
    
    # Create custom legend with alphabetically ordered entries
    legend_elements = []
    for label in unique_labels:
        legend_elements.append(
            Rectangle((0, 0), 1, 1, fc=colors[label], 
                     label=label, alpha=1)  # Use full opacity for legend
        )
    
    # Calculate optimal number of columns for main legend
    n_cols = optimize_legend_columns(len(unique_labels), fig.get_figheight())
    
    # Add main legend with optimized columns
    main_legend = ax.legend(handles=legend_elements, 
                           bbox_to_anchor=(1.05, 1),
                           loc='upper left',
                           borderaxespad=0.,
                           ncol=n_cols,
                           title=legend_title)
    
    # Add experiment parameters legend if requested and output_file is provided
    if show_params and output_file:
        params = parse_filename_parameters(output_file)
        params_text = create_experiment_params_text(params)
        
        # Create parameter text box in the lower right corner of the plot area
        ax.text(0.98, 0.02, params_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor='white', 
                         edgecolor='gray', 
                         alpha=0.9,
                         linewidth=1))
    
    # Format axes with thesis styling
    format_axes(ax, 
                xlabel="Dim 1", 
                ylabel="Dim 2")
    
    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        # Ensure axis limits are symmetric for consistent scaling
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        max_range = max(x_max - x_min, y_max - y_min) / 2
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig, ax

def create_single_overlap_heatmap(vector_file, metadata_file_path, label_column,
                                 output_file=None, grid_resolution=50):
    """
    Create a single overlap heatmap for one experiment.
    
    Args:
        vector_file (str): Path to vector file
        metadata_file_path (str): Path to metadata file
        label_column (str): Column name to use for family identification
        output_file (str, optional): Path to save the plot
        grid_resolution (int): Number of grid cells per dimension (default 50 for 50x50 grid)
        
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Setup thesis style
    setup_thesis_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_file_path, sep='\t')
    metadata_df.reset_index(inplace=True)
    metadata_df.rename(columns={'index': 'sequence_index'}, inplace=True)
    
    try:
        # Load vector data
        vectors_df = pd.read_csv(vector_file, sep='\t', dtype=np.float32)
        vectors_df.columns = ['reduced_vector_d1', 'reduced_vector_d2']
        vectors_df.reset_index(inplace=True)
        vectors_df.rename(columns={'index': 'sequence_index'}, inplace=True)
        
        # Merge with metadata
        merged_df = pd.merge(vectors_df, metadata_df, on='sequence_index', how='inner')
        
        # Apply label mapping for family type columns
        merged_df = map_family_type_labels(merged_df, label_column)
        
        if label_column not in merged_df.columns:
            ax.text(0.5, 0.5, f'No {label_column} data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return fig, ax
        
        # Calculate data bounds
        x_min, x_max = merged_df['reduced_vector_d1'].min(), merged_df['reduced_vector_d1'].max()
        y_min, y_max = merged_df['reduced_vector_d2'].min(), merged_df['reduced_vector_d2'].max()
        
        # Make the grid square by using the maximum range
        max_range = max(x_max - x_min, y_max - y_min)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        x_min = x_center - max_range / 2
        x_max = x_center + max_range / 2
        y_min = y_center - max_range / 2
        y_max = y_center + max_range / 2
        
        # Create grid edges
        x_edges = np.linspace(x_min, x_max, grid_resolution + 1)
        y_edges = np.linspace(y_min, y_max, grid_resolution + 1)
        
        # Calculate overlap heatmap
        overlap_grid = np.zeros((grid_resolution, grid_resolution))
        
        for grid_row in range(grid_resolution):
            for grid_col in range(grid_resolution):
                x_min_cell = x_edges[grid_col]
                x_max_cell = x_edges[grid_col + 1]
                y_min_cell = y_edges[grid_row]
                y_max_cell = y_edges[grid_row + 1]
                
                in_cell_mask = (
                    (merged_df['reduced_vector_d1'] >= x_min_cell) &
                    (merged_df['reduced_vector_d1'] < x_max_cell) &
                    (merged_df['reduced_vector_d2'] >= y_min_cell) &
                    (merged_df['reduced_vector_d2'] < y_max_cell)
                )
                
                cell_data = merged_df[in_cell_mask]
                
                if len(cell_data) == 0:
                    overlap_grid[grid_row, grid_col] = np.nan  # Empty cells as NaN
                else:
                    unique_families = cell_data[label_column].nunique()
                    total_points = len(cell_data)
                    
                    if unique_families <= 1:
                        overlap_grid[grid_row, grid_col] = 0  # Perfect separation
                    else:
                        family_counts = cell_data[label_column].value_counts()
                        family_proportions = family_counts / total_points
                        entropy = -np.sum(family_proportions * np.log2(family_proportions + 1e-10))
                        overlap_score = entropy * unique_families * np.log(total_points + 1)
                        overlap_grid[grid_row, grid_col] = overlap_score
        
        # Create heatmap
        im = ax.imshow(overlap_grid, extent=[x_min, x_max, y_min, y_max],
                      origin='lower', cmap='viridis', aspect='equal', interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Densidad de superposición', rotation=270, labelpad=15)
        
        # Parse parameters for title
        params = parse_filename_parameters(vector_file)
        if params['type'] == 'tsne':
            title = f"Overlap Analysis - Perplexity = {params['perplexity']}"
        elif params['type'] == 'umap':
            title = f"Overlap Analysis - n_neighbors: {params['n_neighbors']}"
        else:
            title = f"Overlap Analysis - {os.path.basename(vector_file)}"
        
        ax.set_title(title, fontweight='normal')
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        
    except Exception as e:
        logging.error(f"Error processing {vector_file}: {str(e)}")
        ax.text(0.5, 0.5, 'Error loading data', transform=ax.transAxes, 
               ha='center', va='center', fontsize=12)
        ax.set_title(f"Error: {os.path.basename(vector_file)}")
    
    return fig, ax

def create_overlap_heatmap(vector_files, metadata_file_path, label_column,
                          output_file=None, grid_resolution=50):
    """
    Create an overlap heatmap analysis showing family overlap in different experiments.
    
    Args:
        vector_files (list): List of vector file paths
        metadata_file_path (str): Path to metadata file
        label_column (str): Column name to use for family identification
        output_file (str, optional): Path to save the plot
        grid_resolution (int): Number of grid cells per dimension (default 50 for 50x50 grid)
        
    Returns:
        tuple: (fig, axes) matplotlib objects
    """
    # Setup thesis style
    setup_thesis_style()
    
    # Determine grid dimensions
    n_files = len(vector_files)
    n_cols = min(3, n_files)  # Maximum 3 columns
    n_rows = math.ceil(n_files / n_cols)
    
    # Create figure with appropriate size
    fig_width = 5 * n_cols
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                           constrained_layout=True)
    
    # Handle different subplot array cases
    if n_files == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        pass  # axes is already correct as 1D array
    elif n_cols == 1 and n_rows > 1:
        pass  # axes is already correct as 1D array
    
    # Sort vector files by perplexity parameter for t-SNE or n_neighbors for UMAP
    def get_sort_key(filename):
        try:
            params = parse_filename_parameters(filename)
            if params['type'] == 'tsne':
                return int(params['perplexity'])
            elif params['type'] == 'umap':
                return int(params['n_neighbors'])
            else:
                return 0
        except:
            return 0
    
    vector_files = sorted(vector_files, key=get_sort_key)
    
    # Load metadata once
    metadata_df = pd.read_csv(metadata_file_path, sep='\t')
    metadata_df.reset_index(inplace=True)
    metadata_df.rename(columns={'index': 'sequence_index'}, inplace=True)
    
    # Calculate global data bounds for consistent grid across all plots
    all_data_points = []
    overlap_grids = []  # Store all grids for global max calculation
    for vector_file in vector_files:
        try:
            vectors_df = pd.read_csv(vector_file, sep='\t', dtype=np.float32)
            vectors_df.columns = ['reduced_vector_d1', 'reduced_vector_d2']
            vectors_df.reset_index(inplace=True)
            vectors_df.rename(columns={'index': 'sequence_index'}, inplace=True)
            
            merged_df = pd.merge(vectors_df, metadata_df, on='sequence_index', how='inner')
            
            # Apply label mapping for family type columns
            merged_df = map_family_type_labels(merged_df, label_column)
            
            if label_column in merged_df.columns:
                all_data_points.extend(merged_df[['reduced_vector_d1', 'reduced_vector_d2']].values)
        except Exception as e:
            logging.error(f"Error loading {vector_file}: {str(e)}")
    
    if not all_data_points:
        logging.error("No valid data points found")
        return fig, axes
    
    all_data_points = np.array(all_data_points)
    global_x_min, global_x_max = np.min(all_data_points[:, 0]), np.max(all_data_points[:, 0])
    global_y_min, global_y_max = np.min(all_data_points[:, 1]), np.max(all_data_points[:, 1])
    
    # Make the grid square by using the maximum range
    max_range = max(global_x_max - global_x_min, global_y_max - global_y_min)
    x_center = (global_x_min + global_x_max) / 2
    y_center = (global_y_min + global_y_max) / 2
    
    global_x_min = x_center - max_range / 2
    global_x_max = x_center + max_range / 2
    global_y_min = y_center - max_range / 2
    global_y_max = y_center + max_range / 2
    
    # Create grid edges
    x_edges = np.linspace(global_x_min, global_x_max, grid_resolution + 1)
    y_edges = np.linspace(global_y_min, global_y_max, grid_resolution + 1)
    
    # First pass: calculate all overlap grids and find global max
    for vector_file in vector_files:
        try:
            vectors_df = pd.read_csv(vector_file, sep='\t', dtype=np.float32)
            vectors_df.columns = ['reduced_vector_d1', 'reduced_vector_d2']
            vectors_df.reset_index(inplace=True)
            vectors_df.rename(columns={'index': 'sequence_index'}, inplace=True)
            merged_df = pd.merge(vectors_df, metadata_df, on='sequence_index', how='inner')
            
            # Apply label mapping for family type columns
            merged_df = map_family_type_labels(merged_df, label_column)
            
            overlap_grid = np.zeros((grid_resolution, grid_resolution))
            if label_column in merged_df.columns:
                for grid_row in range(grid_resolution):
                    for grid_col in range(grid_resolution):
                        x_min_cell = x_edges[grid_col]
                        x_max_cell = x_edges[grid_col + 1]
                        y_min_cell = y_edges[grid_row]
                        y_max_cell = y_edges[grid_row + 1]
                        in_cell_mask = (
                            (merged_df['reduced_vector_d1'] >= x_min_cell) &
                            (merged_df['reduced_vector_d1'] < x_max_cell) &
                            (merged_df['reduced_vector_d2'] >= y_min_cell) &
                            (merged_df['reduced_vector_d2'] < y_max_cell)
                        )
                        cell_data = merged_df[in_cell_mask]
                        if len(cell_data) == 0:
                            overlap_grid[grid_row, grid_col] = np.nan
                        else:
                            unique_families = cell_data[label_column].nunique()
                            total_points = len(cell_data)
                            if unique_families <= 1:
                                overlap_grid[grid_row, grid_col] = 0
                            else:
                                family_counts = cell_data[label_column].value_counts()
                                family_proportions = family_counts / total_points
                                entropy = -np.sum(family_proportions * np.log2(family_proportions + 1e-10))
                                overlap_score = entropy * unique_families * np.log(total_points + 1)
                                overlap_grid[grid_row, grid_col] = overlap_score
            overlap_grids.append(overlap_grid)
        except Exception as e:
            logging.error(f"Error loading {vector_file}: {str(e)}")
            overlap_grids.append(np.zeros((grid_resolution, grid_resolution)))
    
    global_max = np.nanmax([np.nanmax(grid) for grid in overlap_grids]) if overlap_grids else 1.0
    
    # Second pass: plot all heatmaps with shared color scale
    for i, (vector_file, overlap_grid) in enumerate(zip(vector_files, overlap_grids)):
        row = i // n_cols
        col = i % n_cols
        if n_files == 1:
            ax = axes[0] if isinstance(axes, list) else axes
        elif n_rows == 1 and n_cols > 1:
            ax = axes[col]
        elif n_cols == 1 and n_rows > 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        im = ax.imshow(overlap_grid, extent=[global_x_min, global_x_max, global_y_min, global_y_max],
                      origin='lower', cmap='viridis', aspect='equal', interpolation='bilinear', vmin=0, vmax=global_max)
        params = parse_filename_parameters(vector_file)
        if params['type'] == 'tsne':
            subtitle = f"Perplexity = {params['perplexity']}"
        elif params['type'] == 'umap':
            subtitle = f"n_neighbors: {params['n_neighbors']}"
        else:
            subtitle = os.path.basename(vector_file)
        ax.set_title(subtitle, fontweight='normal')
        if row == n_rows - 1:
            ax.set_xlabel("Dim 1")
        if col == 0:
            ax.set_ylabel("Dim 2")
    
    # Hide empty subplots
    total_subplots = n_rows * n_cols
    for i in range(n_files, total_subplots):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1 and n_cols > 1:
            axes[col].set_visible(False)
        elif n_cols == 1 and n_rows > 1:
            axes[row].set_visible(False)
        elif n_rows > 1 and n_cols > 1:
            axes[row, col].set_visible(False)
    
    # Add a single colorbar at the bottom for grid mode
    if n_files > 1:
        cbar = fig.colorbar(im, ax=axes.ravel(), orientation='horizontal', fraction=0.05, pad=0.08)
        cbar.set_label('Densidad de superposición', fontsize=12)
    else:
        cbar = plt.colorbar(im, ax=axes[0] if isinstance(axes, list) else axes, shrink=0.8)
        cbar.set_label('Densidad de superposición', rotation=270, labelpad=15)
    
    return fig, axes

def create_grid_visualization(vector_files, metadata_file_path, label_column, 
                             output_file=None, point_size=0.1, alpha=0.2, legend_title=None):
    """
    Create a grid visualization showing multiple experiments in a single figure.
    
    Args:
        vector_files (list): List of vector file paths
        metadata_file_path (str): Path to metadata file
        label_column (str): Column name to use for coloring
        output_file (str, optional): Path to save the plot
        point_size (float): Size of scatter points
        alpha (float): Transparency of points
        
    Returns:
        tuple: (fig, axes) matplotlib objects
    """
    # Setup thesis style
    setup_thesis_style()
    
    # Determine grid dimensions
    n_files = len(vector_files)
    n_cols = min(3, n_files)  # Maximum 3 columns
    n_rows = math.ceil(n_files / n_cols)
    

    
    # Create figure with appropriate size - more compact to reduce white space
    fig_width = 5 * n_cols  # Reduced from 6 to 5 for less white space
    fig_height = 5 * n_rows  # Reduced from 5 to 4 for less white space
    # Add extra height for the legend at the bottom - estimate based on file count
    # We'll adjust based on actual label count later
    legend_height = 1.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height + legend_height), 
                           constrained_layout=False)  # Use constrained_layout for better spacing
    
    # Handle different subplot array cases
    if n_files == 1:
        # Single subplot - axes is a single Axes object
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        # Single row, multiple columns - axes is 1D array
        pass  # axes is already correct as 1D array
    elif n_cols == 1 and n_rows > 1:
        # Single column, multiple rows - axes is 1D array
        pass  # axes is already correct as 1D array
    # For multiple rows and columns, axes is already a 2D array
    
    # Sort vector files by perplexity parameter for t-SNE or n_neighbors for UMAP
    def get_sort_key(filename):
        try:
            params = parse_filename_parameters(filename)
            if params['type'] == 'tsne':
                return int(params['perplexity'])
            elif params['type'] == 'umap':
                return int(params['n_neighbors'])
            else:
                return 0
        except:
            return 0
    
    vector_files = sorted(vector_files, key=get_sort_key)
    
    # Load metadata once
    metadata_df = pd.read_csv(metadata_file_path, sep='\t')
    metadata_df.reset_index(inplace=True)
    metadata_df.rename(columns={'index': 'sequence_index'}, inplace=True)
    
    # Get all unique labels across all experiments for consistent coloring
    all_labels = set()
    all_data = []
    
    for vector_file in vector_files:
        try:
            # Load vector data
            vectors_df = pd.read_csv(vector_file, sep='\t', dtype=np.float32)
            vectors_df.columns = ['reduced_vector_d1', 'reduced_vector_d2']
            vectors_df.reset_index(inplace=True)
            vectors_df.rename(columns={'index': 'sequence_index'}, inplace=True)
            
            # Merge with metadata
            merged_df = pd.merge(vectors_df, metadata_df, on='sequence_index', how='inner')
            
            # Apply label mapping for family type columns
            merged_df = map_family_type_labels(merged_df, label_column)
            
            all_data.append(merged_df)
            
            # Collect unique labels
            if label_column in merged_df.columns:
                all_labels.update(merged_df[label_column].unique())
                
        except Exception as e:
            logging.error(f"Error loading {vector_file}: {str(e)}")
            all_data.append(None)
    
    # Sort labels alphabetically for consistent coloring
    unique_labels = sorted(all_labels)
    
    # Determine if we have control families
    control_families = None
    if label_column == "sequence_family_name" and len(all_data) > 0 and all_data[0] is not None:
        if "sequence_family_type" in all_data[0].columns:
            control_families = set()
            for df in all_data:
                if df is not None:
                    # Apply mapping to family type column for control detection
                    df_mapped_types = map_family_type_labels(df, "sequence_family_type")
                    for label in unique_labels:
                        family_data = df[df[label_column] == label]
                        if not family_data.empty and (df_mapped_types[df_mapped_types[label_column] == label]["sequence_family_type"] == "control").any():
                            control_families.add(label)
    
    # Generate consistent color palette for all subplots
    colors = generate_consistent_color_palette(unique_labels, control_families)
    
    # Create subplots
    for i, (vector_file, merged_df) in enumerate(zip(vector_files, all_data)):
        row = i // n_cols
        col = i % n_cols
        
        # Handle different axes array shapes correctly
        if n_files == 1:
            # Single subplot
            ax = axes[0] if isinstance(axes, list) else axes
        elif n_rows == 1 and n_cols > 1:
            # Single row, multiple columns - axes is 1D array
            ax = axes[col]
        elif n_cols == 1 and n_rows > 1:
            # Single column, multiple rows - axes is 1D array  
            ax = axes[row]
        else:
            # Multiple rows and columns - axes is 2D array
            ax = axes[row, col]
        
        if merged_df is None:
            ax.text(0.5, 0.5, 'Error loading data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"Error: {os.path.basename(vector_file)}")
            continue
        
        # Set equal aspect ratio to match single mode
        ax.set_aspect('equal')
        
        # Create scatter plot with consistent colors
        for label in unique_labels:
            if label in merged_df[label_column].values:
                mask = merged_df[label_column] == label
                ax.scatter(merged_df.loc[mask, 'reduced_vector_d1'],
                          merged_df.loc[mask, 'reduced_vector_d2'],
                          c=[colors[label]], label=label,
                          alpha=alpha, s=point_size)
        
        # Parse parameters for subtitle - lean format
        params = parse_filename_parameters(vector_file)
        
        if params['type'] == 'tsne':
            subtitle = f"Perplexity = {params['perplexity']}"
        elif params['type'] == 'umap':
            subtitle = f"n_neighbors: {params['n_neighbors']}"
        else:
            subtitle = os.path.basename(vector_file)
        
        # Format subplot with normal font weight
        ax.set_title(subtitle, fontweight='normal')
        if row == n_rows - 1:
            ax.set_xlabel("Dim 1")
        if col == 0:
            ax.set_ylabel("Dim 2")
            
        # Ensure axis limits are symmetric for consistent scaling
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        max_range = max(x_max - x_min, y_max - y_min) / 2
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        
        # Remove individual legends for subplots
        ax.legend().set_visible(False)
    
    # Hide empty subplots
    total_subplots = n_rows * n_cols
    for i in range(n_files, total_subplots):
        row = i // n_cols
        col = i % n_cols
        
        # Handle different axes array shapes for hiding subplots
        if n_rows == 1 and n_cols > 1:
            axes[col].set_visible(False)
        elif n_cols == 1 and n_rows > 1:
            axes[row].set_visible(False)
        elif n_rows > 1 and n_cols > 1:
            axes[row, col].set_visible(False)
        # For single subplot case, no empty subplots to hide
    
    # Create shared legend
    legend_elements = []
    for label in unique_labels:
        legend_elements.append(
            Rectangle((0, 0), 1, 1, fc=colors[label], 
                     label=label, alpha=1)
        )
    
    # Add shared legend at the bottom with more columns for horizontal layout
    # Calculate columns for bottom legend (more columns, shorter)
    max_cols_per_row = 10  # Increase number of columns to make legend more compact
    n_cols = min(max_cols_per_row, len(unique_labels))
    
    # Position legend outside the figure area, below the plots
    # Use custom legend title if provided, otherwise default
    if legend_title is None:
        legend_title = label_column.replace('_', ' ').title()
    
    fig.legend(handles=legend_elements, 
              loc='upper center',
              bbox_to_anchor=(0.5, 0.15),  # Position just below the figure, adjusted for better centering
              bbox_transform=fig.transFigure,
              title=legend_title,
              ncol=n_cols,
              columnspacing=0.8,  # Reduce space between columns
              handletextpad=0.4,  # Reduce space between handle and text
              frameon=True,       # Add frame for better visibility
              borderaxespad=0.5)
    
    # Remove overall title as requested
    
    # With constrained_layout=True, we don't need tight_layout()
    # Just adjust bottom margin for legend
    label_count = len(unique_labels)
    # Estimate legend height based on number of rows it will take
    legend_rows = math.ceil(label_count / max_cols_per_row)
    # Reserve space for legend: base margin + space for legend rows
    bottom_margin = 0.05 + (legend_rows * 0.03)  # More precise calculation
    plt.subplots_adjust(bottom=bottom_margin, hspace=0.3, wspace=0.3)  # Adjusted spacing between subplots
    
    return fig, axes

# ===============================
# DATA LOADING
# ===============================

def load_and_merge_data(vector_file_path, metadata_file_path):
    """
    Load vector data and merge with metadata.
    
    Args:
        vector_file_path (str): Path to vector TSV file
        metadata_file_path (str): Path to metadata TSV file
        
    Returns:
        DataFrame: Merged dataframe with vectors and metadata
    """
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
    metadata_df = pd.read_csv(metadata_file_path, sep='\t', encoding='utf-8', keep_default_na=False) 
    metadata_df.reset_index(inplace=True)
    metadata_df.rename(columns={'index': 'sequence_index'}, inplace=True)
    
    # Merge dataframes
    merged_df = pd.merge(vectors_df, metadata_df, on='sequence_index', how='inner')
    
    logging.info(f"Merged dataset shape: {merged_df.shape}")
    logging.info("END TASK - load_and_merge_data")
    
    return merged_df

# ===============================
# MAIN PROCESSING FUNCTIONS
# ===============================

def create_plots_for_experiment(experiment_folder_path, metadata_file_path, run_id, 
                               output_folder_path, mode='single', chart_type='name', 
                               use_combined=False, grid_resolution=50):
    """
    Create plots for all TSV files in the experiment folder.
    
    Args:
        experiment_folder_path (str): Path to experiment folder
        metadata_file_path (str): Path to metadata file
        run_id (str): Run identifier
        output_folder_path (str): Output folder path
        mode (str): 'single' or 'grid' visualization mode
        chart_type (str): 'name' or 'type' for family labeling
        use_combined (bool): Whether using combined dataset
    """
    logging.info("START TASK - create_plots_for_experiment")
    
    # Find all TSV files in experiment folder (excluding metadata files)
    tsv_pattern = os.path.join(experiment_folder_path, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    
    # Filter for vector files (both t-SNE and UMAP)
    vector_files = [f for f in tsv_files if any(method in os.path.basename(f) 
                   for method in ['vectors_tsne', 'vectors_umap'])]
    
    if not vector_files:
        logging.warning(f"No vector files found in {experiment_folder_path}")
        return
    
    logging.info(f"Found {len(vector_files)} vector files to process")
    
    # Create charts subfolder inside the experiment folder
    charts_folder = os.path.join(experiment_folder_path, "charts")
    os.makedirs(charts_folder, exist_ok=True)
    
    # Determine label columns based on chart_type
    if chart_type == 'name':
        label_columns = ["sequence_family_name"]
    elif chart_type == 'type':
        label_columns = ["sequence_family_type"]
    elif chart_type == 'overlap':
        label_columns = ["sequence_family_name"]  # Use family names for overlap analysis
    else:
        label_columns = ["sequence_family_name", "sequence_family_type"]
    
    # Process based on mode and chart type
    if chart_type == 'overlap':
        if mode == 'single':
            # Create individual overlap heatmaps for each vector file
            for vector_file in vector_files:
                logging.info(f"Processing overlap heatmap for: {os.path.basename(vector_file)}")
                
                for label_column in label_columns:
                    try:
                        # Generate output filename
                        base_filename = os.path.splitext(os.path.basename(vector_file))[0]
                        overlap_filename = f"{base_filename}-overlap-{label_column}.png"
                        overlap_output_path = os.path.join(charts_folder, overlap_filename)
                        
                        logging.info(f"Creating single overlap heatmap for {label_column}: {overlap_filename}")
                        
                        # Create single overlap heatmap
                        fig, ax = create_single_overlap_heatmap(
                            vector_file,
                            metadata_file_path,
                            label_column,
                            output_file=overlap_output_path,
                            grid_resolution=grid_resolution
                        )
                        
                        # Save the overlap plot
                        save_figure(fig, overlap_output_path)
                        plt.close(fig)
                        
                        logging.info(f"Single overlap heatmap saved: {overlap_filename}")
                        
                    except Exception as e:
                        logging.error(f"Error creating single overlap heatmap for {vector_file}: {str(e)}")
                        continue
        
        elif mode == 'grid':
            # Create grid overlap heatmap for all vector files
            for label_column in label_columns:
                logging.info(f"Creating grid overlap heatmap for {label_column}")
                
                # Generate output filename for overlap
                overlap_filename = f"overlap-grid-{label_column}.png"
                overlap_output_path = os.path.join(charts_folder, overlap_filename)
                
                try:
                    # Create overlap heatmap
                    fig, axes = create_overlap_heatmap(
                        vector_files,
                        metadata_file_path,
                        label_column,
                        output_file=overlap_output_path,
                        grid_resolution=grid_resolution  # Use passed parameter
                    )
                    
                    # Save the overlap plot
                    save_figure(fig, overlap_output_path)
                    plt.close(fig)
                    
                    logging.info(f"Grid overlap heatmap saved: {overlap_filename}")
                    
                except Exception as e:
                    logging.error(f"Error creating grid overlap heatmap for {label_column}: {str(e)}")
                    continue
    
    elif mode == 'single':
        # Create individual plots for each vector file and label column
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
                    
                    # Create the enhanced plot
                    fig, ax = create_enhanced_scatter_plot(
                        merged_df, 
                        label_column, 
                        output_file=plot_output_path, 
                        point_size=0.1, 
                        alpha=0.2,
                        legend_title="Familias de proteínas"
                    )
                    
                    # Save the plot
                    save_figure(fig, plot_output_path)
                    plt.close(fig)
                    
            except Exception as e:
                logging.error(f"Error processing {vector_file}: {str(e)}")
                continue
    
    elif mode == 'grid':
        # Create grid visualizations for each label column
        for label_column in label_columns:
            logging.info(f"Creating grid visualization for {label_column}")
            
            # Generate output filename for grid
            grid_filename = f"grid-{chart_type}-{label_column}.png"
            grid_output_path = os.path.join(charts_folder, grid_filename)
            
            try:
                # Create grid visualization
                fig, axes = create_grid_visualization(
                    vector_files,
                    metadata_file_path,
                    label_column,
                    output_file=grid_output_path,
                    point_size=0.05,
                    alpha=0.1,
                    legend_title="Familias de proteínas"  # Customize this as needed
                )
                
                # Save the grid plot
                save_figure(fig, grid_output_path)
                plt.close(fig)
                
                logging.info(f"Grid plot saved: {grid_filename}")
                
            except Exception as e:
                logging.error(f"Error creating grid visualization for {label_column}: {str(e)}")
                continue
    
    logging.info("END TASK - create_plots_for_experiment")

def create_visualization_plots(input_data_root_path, family_dataset_name, timestamp, 
                             filter_name, partition_rule_name, experiment_name, 
                             mode='single', chart_type='name', use_combined=False, grid_resolution=50):
    """
    Main function to create visualization plots for experiment results.
    
    Args:
        input_data_root_path (str): Root path for input data
        family_dataset_name (str): Dataset name
        timestamp (str): Timestamp identifier
        filter_name (str): Filter name
        partition_rule_name (str): Partition rule name
        experiment_name (str): Experiment name
        mode (str): 'single' or 'grid' visualization mode
        chart_type (str): 'name' or 'type' for family labeling
        use_combined (bool): Whether to use combined dataset
    """
    logging.info("START FLOW ******************* Create Enhanced Visualization Plots *******************")
    
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
            raise FileNotFoundError(f"Combined metadata file not found: {metadata_file_path}. "
                                  f"Please run model_combine_datasets.py first.")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
    
    # Locate experiment folder
    experiments_folder_path = os.path.join(vector_output_folder_path, "experiments")
    experiment_folder_path = os.path.join(experiments_folder_path, experiment_folder_name)
    
    if not os.path.exists(experiment_folder_path):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_folder_path}")
    
    logging.info(f"Processing experiment folder: {experiment_folder_path}")
    logging.info(f"Using metadata file: {metadata_file_path}")
    logging.info(f"Mode: {mode}, Chart type: {chart_type}")
    
    # Create plots for all vector files in the experiment
    create_plots_for_experiment(
        experiment_folder_path, 
        metadata_file_path, 
        run_id, 
        vector_output_folder_path, 
        mode=mode,
        chart_type=chart_type,
        use_combined=use_combined,
        grid_resolution=grid_resolution
    )
    
    logging.info("END FLOW ******************* Create Enhanced Visualization Plots *******************")

# ===============================
# MAIN ENTRY POINT
# ===============================

if __name__ == "__main__":
    # Setup environment
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    config = dotenv_values(dotenv_path)

    # Configure logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'model_viz_enhanced.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Create enhanced visualization plots from experiment results'
    )
    parser.add_argument('timestamp', help='Run timestamp')
    parser.add_argument('dataset_name', help='Input protein dataset name')
    parser.add_argument('filter', help='MR Filter')
    parser.add_argument('partition_rule', help='MR partition rule')
    parser.add_argument('experiment_name', help='Experiment folder name (without .json extension)')
    parser.add_argument('--control', action='store_true',
                       help='Use combined dataset (original + control) for visualization')
    parser.add_argument('--mode', choices=['single', 'grid'], default='single',
                       help='Visualization mode: single (individual plots) or grid (combined grid)')
    parser.add_argument('--chart-type', choices=['name', 'type', 'overlap'], default='name',
                       help='Chart type: name (family_name), type (family_type), or overlap (heatmap analysis)')
    parser.add_argument('--grid-resolution', type=int, default=50,
                       help='Grid resolution for overlap analysis (default: 50)')
    
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
    
    # Create enhanced visualization plots
    create_visualization_plots(
        input_data_root_path, 
        family_dataset_name, 
        timestamp, 
        filter_name, 
        partition_rule_name, 
        experiment_name, 
        mode=args.mode,
        chart_type=args.chart_type,
        use_combined=args.control,
        grid_resolution=args.grid_resolution
    ) 