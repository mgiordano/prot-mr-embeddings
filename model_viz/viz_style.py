"""
Thesis Chart Style Configuration
================================

This module provides consistent styling for all thesis charts.
Import and use setup_thesis_style() in all plotting scripts to maintain consistency.

Usage:
    from thesis_style import setup_thesis_style
    setup_thesis_style()
"""

import matplotlib.pyplot as plt
import seaborn as sns

def setup_thesis_style():
    """
    Configure matplotlib and seaborn for consistent, scientific paper styling.
    This function should be called at the beginning of all plotting scripts.
    """
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Configure matplotlib parameters for scientific papers
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        
        # Font settings - clean and modern like scientific papers, larger for thesis readability
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
        
        # Grid and spines - clean and minimal
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        
        # Colors and aesthetics - professional grayscale with accent colors
        'axes.edgecolor': '#333333',
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'axes.facecolor': '#fafafa',
        
        # Layout
        'figure.autolayout': True,
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
        'edgecolor': 'none',
        'pad_inches': 0.1
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
    
    # Ensure consistent background
    ax.set_facecolor('#fafafa')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 