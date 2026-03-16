"""
Interactive Protein Embedding Explorer
======================================
Standalone Dash + Plotly web app for exploring protein embeddings.

Usage:
    cd prot-mr-embeddings
    python -m model_viz.explorer_app
    Open http://localhost:8050

Features:
    - WebGL scatter (scattergl) handles 1M+ points
    - Three rendering modes: distinct colors, highlight, density heatmap
    - Searchable dropdown for high-cardinality columns
    - CATH hierarchical drill-down tab
    - Hover tooltips with protein metadata
    - PNG export
"""

import os
import sys
import glob
from pathlib import Path

# Ensure project root is on path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate
import colorcet as cc
import datashader as ds
from dotenv import load_dotenv

import utils.utils as corpus_utils
from utils.utils import dataset_names, filters, partition_rules

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(_root, ".env"), override=True)
INPUT_DATA_ROOT_PATH = os.environ.get("INPUT_DATA_ROOT_PATH", "")

CARD_THRESHOLD = 50          # ≤ this → distinct colors; above → highlight
CATH_HIERARCHY = ["cath_class", "cath_architecture", "superfamily", "funfam_id"]
INTERNAL_COLS = {
    "reduced_vector_d1", "reduced_vector_d2", "sequence_index",
    "sequence", "word_partition",
}
# Columns excluded from "Color by" dropdown (superset of INTERNAL_COLS)
_COLORBY_EXCLUDE = INTERNAL_COLS | {"sequence_name"}

# Zoom-adaptive decimation settings
_GRID_BINS = 500             # bins per axis for spatial indexing
_MAX_POINTS_ZOOMED_OUT = 300_000   # cap when fully zoomed out
_MAX_POINTS_PARTIAL    = 500_000   # cap when partially zoomed in

# 30 hand-picked distinct colours (low-cardinality palette)
_PAL30 = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ffe119", "#e6beff",
    "#1abc9c", "#2ecc71", "#e74c3c", "#3498db", "#9b59b6",
    "#f39c12", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
]

# ── Server-side state (single-user local tool) ───────────────────────────────
_S = dict(df=None, meta_cols=[], card={}, paths=None,
          grid_x_edges=None, grid_y_edges=None, grid_bin_x=None, grid_bin_y=None,
          trace_index_map=None, last_fig=None, last_fig_args=None,
          color_map={}, hier_filt_df=None)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_available_datasets():
    if not INPUT_DATA_ROOT_PATH or not os.path.exists(INPUT_DATA_ROOT_PATH):
        return []
    # Datasets are directories in the root path
    return sorted([d for d in os.listdir(INPUT_DATA_ROOT_PATH)
                   if os.path.isdir(os.path.join(INPUT_DATA_ROOT_PATH, d))])


def get_available_dates(dataset):
    if not dataset:
        return []
    ds_path = os.path.join(INPUT_DATA_ROOT_PATH, dataset)
    if not os.path.exists(ds_path):
        return []
    # Dates are usually YYYYMMDD folders inside the dataset
    return sorted([d for d in os.listdir(ds_path)
                   if os.path.isdir(os.path.join(ds_path, d)) and d.isdigit()], reverse=True)


def get_available_runs(dataset, date):
    if not dataset or not date:
        return []
    vo_path = os.path.join(INPUT_DATA_ROOT_PATH, dataset, date, "vector_output")
    if not os.path.exists(vo_path):
        return []
    runs = set()
    # Find all metadata files to identify runs
    for f in glob.glob(os.path.join(vo_path, "*-metadata.tsv")):
        bn = os.path.basename(f)
        # remove "-metadata.tsv" or "-filtered-metadata.tsv"
        run_id = bn.replace("-metadata.tsv", "").replace("-filtered", "").replace("-combined", "")
        runs.add(run_id)
    return sorted(list(runs), reverse=True)


def get_available_experiments(dataset, date):
    if not dataset or not date:
        return []
    exp_path = os.path.join(INPUT_DATA_ROOT_PATH, dataset, date, "vector_output", "experiments")
    if not os.path.exists(exp_path):
        return []
    return sorted([d for d in os.listdir(exp_path)
                   if os.path.isdir(os.path.join(exp_path, d))], reverse=True)



def find_vector_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in glob.glob(os.path.join(folder, "*.tsv"))
        if any(t in os.path.basename(f) for t in ("vectors_tsne", "vectors_umap", "vectors_densmap", "vectors_pca"))
    )


def load_data(vec_path, meta_path):
    vdf = pd.read_csv(vec_path, sep="\t", dtype=np.float32)
    # If more than 2 columns, take only the first two (for 2D visualization)
    if vdf.shape[1] > 2:
        vdf = vdf.iloc[:, :2]
    vdf.columns = ["reduced_vector_d1", "reduced_vector_d2"]
    vdf = vdf.reset_index().rename(columns={"index": "sequence_index"})
    mdf = pd.read_csv(meta_path, sep="\t", encoding="utf-8", keep_default_na=False)
    mdf = mdf.reset_index().rename(columns={"index": "sequence_index"})
    merged = pd.merge(vdf, mdf, on="sequence_index", how="inner")
    # Cast known categorical columns that pandas may parse as numeric
    # (e.g. cath_class=1 → int64, cath_architecture=1.1 → float64)
    for c in CATH_HIERARCHY:
        if c in merged.columns and merged[c].dtype != object:
            merged[c] = merged[c].astype(str)
    # Convert string cols to category dtype → 5-10× memory reduction, faster groupby
    for c in merged.select_dtypes(include="object").columns:
        merged[c] = merged[c].fillna("N/A").astype("category")
    return merged


def _build_spatial_index(df):
    """Precompute spatial grid bins for fast zoom-level decimation."""
    x = df["reduced_vector_d1"].values
    y = df["reduced_vector_d2"].values
    x_edges = np.linspace(x.min(), x.max() + 1e-9, _GRID_BINS + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-9, _GRID_BINS + 1)
    bin_x = np.digitize(x, x_edges) - 1  # 0-indexed
    bin_y = np.digitize(y, y_edges) - 1
    return x_edges, y_edges, bin_x.astype(np.int16), bin_y.astype(np.int16)


def _decimate(df, max_pts, x_range=None, y_range=None):
    """Return a uniform random decimated subset of *df* for the current view."""
    if x_range is not None and y_range is not None:
        mask = (
            (df["reduced_vector_d1"] >= x_range[0]) &
            (df["reduced_vector_d1"] <= x_range[1]) &
            (df["reduced_vector_d2"] >= y_range[0]) &
            (df["reduced_vector_d2"] <= y_range[1])
        )
        visible = df[mask]
    else:
        visible = df

    if len(visible) <= max_pts:
        return visible

    # Uniform random subsample
    rng = np.random.RandomState(42)
    idx = np.arange(len(visible))
    rng.shuffle(idx)
    if len(idx) > max_pts:
        idx = idx[:max_pts]
    return visible.iloc[idx]


def detect_columns(df):
    cols, card = [], {}
    for c in df.columns:
        if c in _COLORBY_EXCLUDE:
            continue
        if df[c].dtype == object or df[c].dtype.name == "category":
            n = df[c].nunique()
            cols.append(c)
            card[c] = n
        elif pd.api.types.is_numeric_dtype(df[c]):
            # Include low-cardinality numeric cols (they are likely categorical)
            n = df[c].nunique()
            if n <= CARD_THRESHOLD:
                df[c] = df[c].astype(str).astype("category")
                cols.append(c)
                card[c] = n
    return cols, card


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _colors_for(labels):
    """Map sorted labels → hex colour strings."""
    gb = cc.glasbey_light
    return {lab: (_PAL30[i] if i < len(_PAL30) else gb[i % len(gb)])
            for i, lab in enumerate(sorted(labels))}

def _global_colors_for_col(df, col):
    """Returns a deterministic colour map for all active values of a column in the global dataset."""
    if _S.get("global_colors") is None:
        _S["global_colors"] = {}
        
    if col in _S["global_colors"]:
        return _S["global_colors"][col]

    if col not in df.columns:
        return {}
    
    if df[col].dtype.name == "category":
        cats = sorted(list(df[col].cat.categories))
    else:
        cats = sorted(df[col].dropna().unique())
        
    _S["global_colors"][col] = _colors_for(cats)
    return _S["global_colors"][col]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_distinct(df, col, selected, pt, alpha, bg):
    """Scatter with distinct color per category (low cardinality)."""
    if df[col].dtype.name == 'category':
        all_cats = sorted([c for c in df[col].cat.categories if c in df[col].unique()])
    else:
        all_cats = sorted(df[col].unique())
    sel = set(selected) if selected else set(all_cats)
    colors = _S["color_map"]  # precomputed stable color map
    fig = go.Figure()
    trace_index_map = []

    # Filter to visible categories and compute true counts
    vis_df = df[df[col].isin(sel)]
    true_counts = vis_df[col].value_counts()
    
    # Decimate visible points for rendering performance
    plot_df = _decimate(vis_df, _MAX_POINTS_ZOOMED_OUT)

    for cat, group in plot_df.groupby(col, observed=True, sort=True):
        if cat not in sel or group.empty:
            continue
        c = colors.get(cat, "#888888")
        count_str = f"{true_counts[cat]:,}"
        fig.add_trace(go.Scattergl(
            x=group["reduced_vector_d1"].values,
            y=group["reduced_vector_d2"].values,
            mode="markers",
            marker=dict(size=pt, color=c, opacity=alpha, line_width=0),
            name=f"{cat} ({count_str})",
            hoverinfo="none",
        ))
        # Map (traceIndex, pointIndex) → original df row index
        trace_index_map.append(group.index.values)
    _S["trace_index_map"] = trace_index_map
    _style(fig, bg, f"{col} — {len(sel)}/{len(all_cats)} categories")
    return fig


# Maximum points for the gray backdrop (visual difference is negligible)
_BACKDROP_CAP = 100_000


def build_highlight(df, col, highlighted, pt, alpha, bg):
    """Gray backdrop + coloured highlights (high cardinality)."""
    fig = go.Figure()
    # Downsample backdrop to cap — sending 1M gray points is the #1 bottleneck
    if len(df) > _BACKDROP_CAP:
        backdrop = df.sample(n=_BACKDROP_CAP, random_state=42)
    else:
        backdrop = df
    fig.add_trace(go.Scattergl(
        x=backdrop["reduced_vector_d1"].values,
        y=backdrop["reduced_vector_d2"].values,
        mode="markers",
        marker=dict(size=max(pt * 0.7, 0.5), color="#d0d0d0", opacity=0.12,
                    line_width=0),
        name="all", hoverinfo="skip", showlegend=False,
    ))
    trace_index_map = [backdrop.index.values]  # trace 0 = backdrop
    if highlighted:
        colors = _S["color_map"]  # precomputed stable color map
        hl_set = set(highlighted)
        hl_mask = df[col].isin(hl_set)
        hl_df_full = df[hl_mask]
        true_counts = hl_df_full[col].value_counts()
        
        # Cap highlighted points if they are anomalously huge, to avoid freezing
        hl_df = _decimate(hl_df_full, _MAX_POINTS_ZOOMED_OUT)
        
        for cat, group in hl_df.groupby(col, observed=True, sort=True):
            if group.empty:
                continue
            c = colors.get(cat, "#888888")
            count_str = f"{true_counts[cat]:,}"
            fig.add_trace(go.Scattergl(
                x=group["reduced_vector_d1"].values,
                y=group["reduced_vector_d2"].values,
                mode="markers",
                marker=dict(size=pt, color=c, opacity=alpha,
                            line_width=0),
                name=f"{cat} ({count_str})",
                hoverinfo="none",
            ))
            trace_index_map.append(group.index.values)
    _S["trace_index_map"] = trace_index_map
    n_tot = df[col].nunique()
    n_hl = len(highlighted) if highlighted else 0
    _style(fig, bg, f"{col} — highlight {n_hl} of {n_tot:,}")
    return fig


def build_density(df, col, cat_filter, bg):
    """Datashader density heatmap."""
    pdf = df[df[col] == cat_filter] if (cat_filter and col) else df
    title = (f"Density: {col}={cat_filter} ({len(pdf):,} pts)"
             if cat_filter else f"Density: all {len(pdf):,} pts")
    if pdf.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font_size=20)
        return fig
    xr = (pdf["reduced_vector_d1"].min(), pdf["reduced_vector_d1"].max())
    yr = (pdf["reduced_vector_d2"].min(), pdf["reduced_vector_d2"].max())
    xp, yp = (xr[1] - xr[0]) * .05, (yr[1] - yr[0]) * .05
    xr, yr = (xr[0] - xp, xr[1] + xp), (yr[0] - yp, yr[1] + yp)
    cvs = ds.Canvas(plot_width=300, plot_height=300, x_range=xr, y_range=yr)
    agg = cvs.points(pdf, "reduced_vector_d1", "reduced_vector_d2")
    z = agg.values.astype(float)
    z[z == 0] = np.nan
    xs = np.linspace(xr[0], xr[1], z.shape[1])
    ys = np.linspace(yr[0], yr[1], z.shape[0])
    cmap = "Hot" if bg == "black" else "Blues"
    fig = go.Figure(go.Heatmap(
        z=np.log1p(z), x=xs, y=ys, colorscale=cmap, showscale=True,
        hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>log count: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=700, margin=dict(l=50, r=20, t=50, b=50))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _apply_bg(fig, bg)
    return fig


def build_hierarchy(df, path_json, pt, alpha, bg,
                    viz_col=None, viz_mode=None, viz_cats=None, viz_highlighted=None):
    """CATH hierarchy drill-down with sidebar-controlled visualization.

    When *viz_col* is provided the filtered subset is rendered using the
    sidebar's colour-by column, mode, and category selections instead of
    the default CATH-level column.
    """
    available = [c for c in CATH_HIERARCHY if c in df.columns]
    if not available:
        fig = go.Figure()
        fig.add_annotation(text="No CATH hierarchy columns in data", showarrow=False, font_size=16)
        return fig, [], available, 0

    path_list = path_json if path_json else []
    filt = df
    for c, v in path_list:
        filt = filt[filt[c] == v]
    if filt.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data at this level", showarrow=False, font_size=16)
        return fig, [], available, len(path_list)

    # Store filtered subset for sidebar callbacks
    _S["hier_filt_df"] = filt

    lvl = min(len(path_list), len(available) - 1)
    hier_col = available[lvl]
    hier_cats = sorted(filt[hier_col].unique())

    # Decide which column to colour by: sidebar choice or default CATH level
    render_col = viz_col if (viz_col and viz_col in filt.columns) else hier_col
    card = filt[render_col].nunique()

    # Compute stable color map for the render column (using global dataset for consistency)
    prev_cmap = _S["color_map"]
    _S["color_map"] = _global_colors_for_col(df, render_col)

    # Choose rendering mode
    mode = viz_mode or ("distinct" if card <= CARD_THRESHOLD else "highlight")

    if mode == "density":
        # For density we pass the first highlighted cat as filter, or None
        density_cat = viz_cats if isinstance(viz_cats, str) else None
        fig = build_density(filt, render_col, density_cat, bg)
    elif mode == "distinct":
        fig = build_distinct(filt, render_col, viz_cats or None, pt, alpha, bg)
    else:  # highlight
        if viz_highlighted:
            hl = viz_highlighted
        else:
            hl = filt[render_col].value_counts().head(10).index.tolist()
        fig = build_highlight(filt, render_col, hl, pt, alpha, bg)

    _S["color_map"] = prev_cmap  # restore scatter tab color map

    # Build breadcrumb title
    bc = "All"
    for c, v in path_list:
        bc += f" → {c}: {v}"
    bc += f" → [{hier_col} ({filt[hier_col].nunique():,})]"
    if render_col != hier_col:
        bc += f"  ·  color: {render_col} ({card:,})"
    fig.update_layout(title=bc)
    return fig, hier_cats, available, lvl


# ── Styling helpers ───────────────────────────────────────────────────────────

def _style(fig, bg, title):
    _apply_bg(fig, bg)
    fig.update_layout(
        title=dict(text=title, font_size=14),
        xaxis=dict(title="Dim 1", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Dim 2"),
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    font_size=10, itemsizing="constant"),
        hovermode="closest",
        height=700,
    )


def _apply_bg(fig, bg):
    if bg == "black":
        fig.update_layout(paper_bgcolor="#111", plot_bgcolor="#111", font_color="white",
                          xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
    elif bg == "gray":
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="#fafafa", font_color="#333",
                          xaxis=dict(gridcolor="#e0e0e0"), yaxis=dict(gridcolor="#e0e0e0"))
    else:
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")


# ══════════════════════════════════════════════════════════════════════════════
#  DASH LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

_ds_opts = [{"label": d, "value": d} for d in get_available_datasets()]

# CSS tokens
S_SIDEBAR = {"width": "330px", "minWidth": "330px", "padding": "16px",
             "borderRight": "1px solid #ddd", "overflowY": "auto",
             "fontFamily": "Arial, sans-serif", "fontSize": "13px",
             "backgroundColor": "#fafafa", "height": "100vh"}
S_MAIN    = {"flex": "1", "padding": "12px 16px", "overflowY": "auto",
             "fontFamily": "Arial, sans-serif", "height": "100vh"}
S_CARD    = {"marginBottom": "14px", "padding": "12px",
             "border": "1px solid #e0e0e0", "borderRadius": "6px",
             "backgroundColor": "white"}
S_LBL     = {"fontWeight": "bold", "fontSize": "11px", "color": "#555",
             "marginBottom": "4px", "display": "block"}
S_HDR     = {**S_LBL, "fontSize": "12px", "color": "#4363d8", "marginBottom": "8px"}
S_BTN     = {"width": "100%", "padding": "8px", "cursor": "pointer",
             "border": "none", "borderRadius": "4px", "fontWeight": "bold",
             "color": "white", "backgroundColor": "#4363d8", "marginTop": "8px"}
S_BTN_G   = {**S_BTN, "backgroundColor": "#27ae60"}
S_INPUT   = {"width": "100%", "padding": "6px", "marginBottom": "8px",
             "border": "1px solid #ccc", "borderRadius": "4px"}

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Protein Embedding Explorer"

sidebar = html.Div(style=S_SIDEBAR, children=[
    html.H3("🧬 Explorer", style={"margin": "0 0 16px 0", "color": "#333"}),

    # UI State Persistence
    dcc.Store(id="store-ui-state", storage_type="local", data={}),

    # ── Experiment ──
    html.Div(style=S_CARD, children=[
        html.Span("EXPERIMENT", style=S_HDR),
        
        html.Label("Dataset", style=S_LBL),
        dcc.Dropdown(id="sel-dataset", options=_ds_opts, placeholder="Select dataset…"),
        
        html.Label("Date", style={**S_LBL, "marginTop": "8px"}),
        dcc.Dropdown(id="sel-date", options=[], placeholder="Select date…"),

        html.Label("Run (Timestamp - Filter - Partition)", style={**S_LBL, "marginTop": "8px"}),
        dcc.Dropdown(id="sel-run", options=[], placeholder="Select run…"),

        html.Label("Experiment", style={**S_LBL, "marginTop": "8px"}),
        dcc.Dropdown(id="sel-experiment", options=[], placeholder="Select experiment…"),
    ]),

    # ── File ──
    html.Div(style=S_CARD, children=[
        html.Span("FILE", style=S_HDR),
        dcc.Dropdown(id="sel-file", options=[], placeholder="Select file…"),
        html.Label("Sample %", style={**S_LBL, "marginTop": "8px"}),
        dcc.Slider(id="sld-sample", min=10, max=100, step=5, value=100,
                   marks={10: "10%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Button("Load Data", id="btn-load-data", style=S_BTN_G),
    ]),

    # ── Viz controls (hidden until data loaded) ──
    html.Div(id="viz-panel", style={**S_CARD, "display": "none"}, children=[
        html.Span("VISUALIZATION", style=S_HDR),

        html.Label("Color by", style=S_LBL),
        dcc.Dropdown(id="sel-col", options=[], placeholder="…"),

        html.Label("Mode", style={**S_LBL, "marginTop": "8px"}),
        dcc.RadioItems(id="sel-mode",
                       options=[{"label": "Distinct Colors", "value": "distinct"},
                                {"label": "Highlight", "value": "highlight"},
                                {"label": "Density Heatmap", "value": "density"}],
                       value="distinct", inline=True,
                       style={"fontSize": "12px"}),

        # -- distinct-mode controls --
        html.Div(id="ctrl-distinct", children=[
            html.Label("Categories", style={**S_LBL, "marginTop": "8px"}),
            dcc.Dropdown(id="sel-cats", multi=True, searchable=True, placeholder="All shown"),
        ]),

        # -- highlight-mode controls --
        html.Div(id="ctrl-highlight", style={"display": "none"}, children=[
            html.Label("Highlight categories (search)", style={**S_LBL, "marginTop": "8px"}),
            dcc.Dropdown(id="sel-highlight", multi=True, searchable=True,
                         placeholder="Type to search…"),
            html.Label("Top N", style={**S_LBL, "marginTop": "8px"}),
            dcc.Slider(id="sld-topn", min=0, max=30, step=1, value=10,
                       marks={0: "0", 10: "10", 20: "20", 30: "30"}),
        ]),

        # -- density-mode controls --
        html.Div(id="ctrl-density", style={"display": "none"}, children=[
            html.Label("Filter to category (optional)", style={**S_LBL, "marginTop": "8px"}),
            dcc.Dropdown(id="sel-density-cat", searchable=True, placeholder="All points"),
        ]),

        html.Hr(style={"margin": "12px 0"}),
        html.Label("Point size", style=S_LBL),
        dcc.Slider(id="sld-pt", min=0.1, max=5, step=0.1, value=1.5,
                   marks={0.1: "0.1", 1: "1", 2: "2", 3: "3", 5: "5"}),
        html.Label("Opacity", style={**S_LBL, "marginTop": "8px"}),
        dcc.Slider(id="sld-alpha", min=0.05, max=1, step=0.05, value=0.5,
                   marks={0.1: "0.1", 0.5: "0.5", 1: "1"}),
        html.Label("Background", style={**S_LBL, "marginTop": "8px"}),
        dcc.RadioItems(id="sel-bg",
                       options=[{"label": "White", "value": "white"},
                                {"label": "Gray", "value": "gray"},
                                {"label": "Black", "value": "black"}],
                       value="gray", inline=True, style={"fontSize": "12px"}),

        html.Button("🎨 Render Plot", id="btn-render",
                    style={**S_BTN, "backgroundColor": "#e67e22", "marginTop": "12px"}),
        html.Button("📸 Export PNG", id="btn-export",
                    style={**S_BTN, "backgroundColor": "#9b59b6", "marginTop": "6px"}),
    ]),

    # ── Status ──
    html.Div(id="status-box", style={**S_CARD, "fontSize": "12px", "color": "#666"},
             children="Ready."),
])

main_area = html.Div(style=S_MAIN, children=[
    dcc.Tabs(id="tabs", value="scatter", children=[
        dcc.Tab(label="Scatter", value="scatter"),
        dcc.Tab(label="CATH Hierarchy", value="hierarchy"),
    ]),
    # -- Scatter tab content --
    html.Div(id="tab-scatter", children=[
        dcc.Loading(type="circle", children=[
            dcc.Graph(id="main-plot", config={"scrollZoom": True},
                      style={"height": "720px"}),
        ]),
        # Click-to-inspect info panel (replaces hover tooltips for performance)
        html.Div(id="click-info", style={
            "padding": "12px", "margin": "8px 0", "fontSize": "13px",
            "border": "1px solid #e0e0e0", "borderRadius": "6px",
            "backgroundColor": "#f9f9f9", "fontFamily": "monospace",
            "maxHeight": "200px", "overflowY": "auto",
            "color": "#555",
        }, children="💡 Click on a point to inspect its metadata."),
    ]),
    # -- Hierarchy tab content --
    html.Div(id="tab-hierarchy", style={"display": "none"}, children=[
        html.Div(style={"padding": "8px 0"}, children=[
            html.Span("Drill into: ", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="sel-hier-cat", style={"display": "inline-block", "width": "250px",
                                                    "verticalAlign": "middle"}, searchable=True),
            html.Button("Drill ▶", id="btn-drill",
                        style={"marginLeft": "8px", "padding": "6px 16px", "cursor": "pointer",
                               "backgroundColor": "#4363d8", "color": "white", "border": "none",
                               "borderRadius": "4px"}),
            html.Button("◀ Back", id="btn-back",
                        style={"marginLeft": "8px", "padding": "6px 16px", "cursor": "pointer",
                               "backgroundColor": "#e74c3c", "color": "white", "border": "none",
                               "borderRadius": "4px"}),
            html.Button("⟲ Reset", id="btn-reset-hier",
                        style={"marginLeft": "8px", "padding": "6px 16px", "cursor": "pointer",
                               "backgroundColor": "#95a5a6", "color": "white", "border": "none",
                               "borderRadius": "4px"}),
        ]),
        dcc.Loading(type="circle", children=[
            dcc.Graph(id="hier-plot", config={"scrollZoom": True},
                      style={"height": "680px"}),
        ]),
        # Click-to-inspect info panel for hierarchy view
        html.Div(id="hier-click-info", style={
            "padding": "12px", "margin": "8px 0", "fontSize": "13px",
            "border": "1px solid #e0e0e0", "borderRadius": "6px",
            "backgroundColor": "#f9f9f9", "fontFamily": "monospace",
            "maxHeight": "200px", "overflowY": "auto",
            "color": "#555",
        }, children="💡 Click on a point to inspect its metadata."),
    ]),
    # hidden stores
    dcc.Store(id="store-hier-path", data=[]),
    dcc.Store(id="store-data-loaded", data=False),
    dcc.Store(id="store-hier-render-tick", data=0),  # bumped by Render Plot btn
])

app.layout = html.Div(style={"display": "flex", "height": "100vh", "overflow": "hidden"},
                       children=[sidebar, main_area])


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Tab visibility & state reset ─────────────────────────────────────────────
@app.callback(
    Output("tab-scatter", "style"),
    Output("tab-hierarchy", "style"),
    Output("sel-col", "value", allow_duplicate=True),
    Output("main-plot", "figure", allow_duplicate=True),
    Output("hier-plot", "figure", allow_duplicate=True),
    Input("tabs", "value"),
    State("store-data-loaded", "data"),
    prevent_initial_call=True,
)
def toggle_tabs(tab, loaded):
    show = {"display": "block"}
    hide = {"display": "none"}
    
    # When switching tabs, clear the colour column selector to force a re-calc 
    # of the default column and options for the newly active tab's dataset subset.
    # We also pass an empty figure to clear out any stale plot from the previous view.
    empty_fig = go.Figure()
    
    # We only trigger this reset if data is actually loaded, otherwise it might 
    # conflict during the initial layout render.
    reset_col = None if loaded else no_update
    fig_scatter = empty_fig if loaded and tab != "scatter" else no_update
    fig_hier = empty_fig if loaded and tab != "hierarchy" else no_update

    return (
        show if tab == "scatter" else hide,
        show if tab == "hierarchy" else hide,
        reset_col,
        fig_scatter,
        fig_hier
    )


# ── Cascading UI Callbacks ────────────────────────────────────────────────────

@app.callback(
    Output("sel-date", "options"),
    Output("sel-date", "value"),
    Input("sel-dataset", "value"),
)
def update_dates(dataset):
    dates = get_available_dates(dataset)
    opts = [{"label": d, "value": d} for d in dates]
    return opts, dates[0] if dates else None


@app.callback(
    Output("sel-run", "options"),
    Output("sel-run", "value"),
    Output("sel-experiment", "options"),
    Output("sel-experiment", "value"),
    Input("sel-date", "value"),
    State("sel-dataset", "value"),
)
def update_runs_and_exps(date, dataset):
    runs = get_available_runs(dataset, date)
    exps = get_available_experiments(dataset, date)
    r_opts = [{"label": r, "value": r} for r in runs]
    e_opts = [{"label": e, "value": e} for e in exps]
    return (r_opts, runs[0] if runs else None,
            e_opts, exps[0] if exps else None)


@app.callback(
    Output("sel-file", "options"),
    Output("sel-file", "value"),
    Output("status-box", "children", allow_duplicate=True),
    Input("sel-run", "value"),
    Input("sel-experiment", "value"),
    State("sel-date", "value"),
    State("sel-dataset", "value"),
    prevent_initial_call=True,
)
def update_files(run, exp, date, dataset):
    if not all([dataset, date, run, exp]):
        return [], None, "⏳ Select dataset, date, run, and experiment."
    
    exp_folder = os.path.join(INPUT_DATA_ROOT_PATH, dataset, date, "vector_output", "experiments", exp)
    files = find_vector_files(exp_folder)
    if not files:
        return [], None, f"❌ No vector files in experiment '{exp}'"
    
    # Store the exact paths needed by load_data in _S
    # Determine the correct metadata file (handling -filtered suffix)
    vof = os.path.join(INPUT_DATA_ROOT_PATH, dataset, date, "vector_output")
    mf = f"{run}-metadata.tsv"
    if exp.endswith("-filtered"):
        mf = f"{run}-filtered-metadata.tsv"
    
    _S["paths"] = dict(
        metadata=os.path.join(vof, mf),
        exp_folder=exp_folder
    )
    
    opts = [{"label": os.path.basename(f), "value": f} for f in files]
    return opts, files[0], f"✅ Select file and click Load Data."


# ── UI State Persistence ──────────────────────────────────────────────────────

@app.callback(
    Output("store-ui-state", "data"),
    Input("sel-dataset", "value"),
    Input("sel-date", "value"),
    Input("sel-run", "value"),
    Input("sel-experiment", "value"),
    Input("sel-file", "value"),
    State("store-ui-state", "data"),
    prevent_initial_call=True,
)
def save_ui_state(ds, dt, run, exp, file, store):
    # Only save if we have values (avoids overwriting on initial clear)
    store = store or {}
    if ctx.triggered_id == "sel-dataset" and ds: store["dataset"] = ds
    if ctx.triggered_id == "sel-date" and dt: store["date"] = dt
    if ctx.triggered_id == "sel-run" and run: store["run"] = run
    if ctx.triggered_id == "sel-experiment" and exp: store["experiment"] = exp
    if ctx.triggered_id == "sel-file" and file: store["file"] = file
    return store


@app.callback(
    Output("sel-dataset", "value"),
    Input("store-ui-state", "modified_timestamp"),
    State("store-ui-state", "data"),
    State("sel-dataset", "value"),
)
def load_ui_state(ts, store, current_ds):
    if ts is None or store is None:
        raise PreventUpdate
    ds = store.get("dataset")
    if ds and ds != current_ds:
        return ds
    raise PreventUpdate


# ── Hide viz-panel immediately when Load Data is clicked ─────────────────────
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        return {"display": "none"};
    }
    """,
    Output("viz-panel", "style", allow_duplicate=True),
    Input("btn-load-data", "n_clicks"),
    prevent_initial_call=True,
)


# ── Load data ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("viz-panel", "style"),
    Output("sel-col", "options"),
    Output("sel-col", "value"),
    Output("store-data-loaded", "data"),
    Output("status-box", "children", allow_duplicate=True),
    Input("btn-load-data", "n_clicks"),
    State("sel-file", "value"),
    State("sld-sample", "value"),
    prevent_initial_call=True,
)
def load_file(n, fpath, sample_pct):
    if not fpath or not _S["paths"]:
        raise PreventUpdate
    try:
        full_df = load_data(fpath, _S["paths"]["metadata"])
        n_full = len(full_df)
        # Apply uniform random sampling if < 100%
        if sample_pct < 100:
            frac = sample_pct / 100.0
            _S["df"] = full_df.sample(frac=frac, random_state=42).reset_index(drop=True)
        else:
            _S["df"] = full_df
        del full_df  # free memory
        _S["meta_cols"], _S["card"] = detect_columns(_S["df"])
        _S["trace_index_map"] = None
        _S["last_fig"] = None
        _S["last_fig_args"] = None
        _S["global_colors"] = {}
    except Exception as e:
        return {"display": "none"}, [], None, False, f"❌ Load error: {e}"

    opts = [{"label": f"{c}  ({_S['card'][c]:,})", "value": c} for c in _S["meta_cols"]]
    default = opts[0]["value"] if opts else None
    n_rows = len(_S["df"])
    sample_msg = f" (sampled {sample_pct}% from {n_full:,})" if sample_pct < 100 else ""
    return ({**S_CARD}, opts, default, True,
            f"✅ Loaded {n_rows:,} points{sample_msg} · {len(_S['meta_cols'])} metadata cols.")


# ── Auto-set rendering mode when colour column changes ────────────────────────
@app.callback(
    Output("sel-mode", "value"),
    Output("sel-cats", "options"),
    Output("sel-cats", "value"),
    Output("sel-highlight", "options"),
    Output("sel-highlight", "value"),
    Output("sel-density-cat", "options"),
    Output("sel-density-cat", "value"),
    Output("sld-topn", "max"),
    Input("sel-col", "value"),
    State("sld-topn", "value"),
    State("store-data-loaded", "data"),
    State("tabs", "value"),
    prevent_initial_call=True,
)
def on_col_change(col, top_n_val, loaded, active_tab):
    if not loaded or not col or _S["df"] is None:
        raise PreventUpdate
    # Use the filtered subset when on the hierarchy tab
    source_df = (_S["hier_filt_df"]
                 if active_tab == "hierarchy" and _S.get("hier_filt_df") is not None
                 else _S["df"])
    if col not in source_df.columns:
        raise PreventUpdate
    # Single value_counts() call — O(n) instead of O(n*k)
    vc = source_df[col].value_counts()
    vc = vc[vc > 0]  # drop 0-count categories
    card = len(vc)
    cats = sorted(vc.index)
    # Precompute stable color map using global df so colors are consistent across filters
    _S["color_map"] = _global_colors_for_col(_S["df"], col)
    cat_opts = [{"label": f"{c}  ({vc[c]:,})", "value": c} for c in cats]
    mode = "distinct" if card <= CARD_THRESHOLD else "highlight"
    top_n = top_n_val if top_n_val is not None else 10
    # For distinct: pre-select all; for highlight: pre-select top N
    if mode == "distinct":
        return (mode, cat_opts, cats, cat_opts, [], cat_opts, None, min(card, 50))
    else:
        top_n_list = vc.head(top_n).index.tolist()
        return (mode, cat_opts, [], cat_opts, top_n_list, cat_opts, None, min(card, 50))


# ── Show/hide mode-specific controls ─────────────────────────────────────────
@app.callback(
    Output("ctrl-distinct", "style"),
    Output("ctrl-highlight", "style"),
    Output("ctrl-density", "style"),
    Input("sel-mode", "value"),
)
def toggle_mode_controls(mode):
    show, hide = {}, {"display": "none"}
    if mode == "distinct":
        return show, hide, hide
    elif mode == "highlight":
        return hide, show, hide
    else:
        return hide, hide, show


@app.callback(
    Output("sel-highlight", "value", allow_duplicate=True),
    Input("sld-topn", "value"),
    State("sel-col", "value"),
    State("sel-mode", "value"),
    State("tabs", "value"),
    prevent_initial_call=True,
)
def topn_fill(n, col, mode, active_tab):
    if mode != "highlight" or not col or _S["df"] is None:
        raise PreventUpdate
    if n == 0:
        return []
    source_df = (_S["hier_filt_df"]
                 if active_tab == "hierarchy" and _S.get("hier_filt_df") is not None
                 else _S["df"])
    if col not in source_df.columns:
        raise PreventUpdate
    return source_df[col].value_counts().head(n).index.tolist()


# ── Main scatter plot (button-triggered) ─────────────────────────────────────
@app.callback(
    Output("main-plot", "figure"),
    Output("store-hier-render-tick", "data"),
    Input("btn-render", "n_clicks"),
    State("sel-mode", "value"),
    State("sel-cats", "value"),
    State("sel-highlight", "value"),
    State("sel-density-cat", "value"),
    State("sld-pt", "value"),
    State("sld-alpha", "value"),
    State("sel-bg", "value"),
    State("sel-col", "value"),
    State("store-data-loaded", "data"),
    State("main-plot", "relayoutData"),
    State("tabs", "value"),
    State("store-hier-render-tick", "data"),
    prevent_initial_call=True,
)
def update_main_plot(_n, mode, cats, highlighted, density_cat, pt, alpha, bg, col, loaded, relayout, active_tab, tick):
    if not loaded or _S["df"] is None or not col:
        return go.Figure(), no_update

    # When on hierarchy tab, bump the render tick to trigger hierarchy re-render
    # instead of updating the scatter plot.
    if active_tab == "hierarchy":
        return no_update, (tick or 0) + 1

    if mode == "distinct":
        fig = build_distinct(_S["df"], col, cats or None, pt, alpha, bg)
    elif mode == "highlight":
        fig = build_highlight(_S["df"], col, highlighted or [], pt, alpha, bg)
    else:
        fig = build_density(_S["df"], col, density_cat, bg)

    # Preserve previous zoom/pan so re-rendering with a new category keeps
    # the same view region (useful for comparing regions across labels).
    if relayout and "xaxis.range[0]" in relayout and "yaxis.range[0]" in relayout:
        fig.update_xaxes(range=[relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]])
        fig.update_yaxes(range=[relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]])

    return fig, no_update


# ── Click-to-inspect metadata ────────────────────────────────────────────────
@app.callback(
    Output("click-info", "children"),
    Input("main-plot", "clickData"),
    State("store-data-loaded", "data"),
    prevent_initial_call=True,
)
def on_point_click(click_data, loaded):
    """Display full metadata for the clicked point."""
    if not loaded or _S["df"] is None or not click_data:
        raise PreventUpdate
    pt = click_data["points"][0]
    trace_idx = pt.get("curveNumber", 0)
    point_idx = pt.get("pointIndex", 0)

    # Resolve back to original df row
    idx_map = _S.get("trace_index_map")
    if idx_map is not None and trace_idx < len(idx_map):
        orig_idx = idx_map[trace_idx][point_idx]
        row = _S["df"].iloc[orig_idx] if orig_idx < len(_S["df"]) else None
    else:
        # Fallback: find closest point by coordinates
        x_click, y_click = pt.get("x"), pt.get("y")
        if x_click is None or y_click is None:
            raise PreventUpdate
        dist = ((_S["df"]["reduced_vector_d1"] - x_click) ** 2 +
                (_S["df"]["reduced_vector_d2"] - y_click) ** 2)
        orig_idx = dist.idxmin()
        row = _S["df"].iloc[orig_idx]

    if row is None:
        raise PreventUpdate

    # Build info display with colored pill badges for ALL categorical columns
    # (not just the currently selected color-by column).
    # Exclude only coordinate/internal cols; keep sequence_name visible.
    meta_cols = [c for c in _S["df"].columns if c not in INTERNAL_COLS]
    info_items = []
    for c in meta_cols:
        val = str(row[c])
        # Compute deterministic color for this column's value
        color = None
        if c not in _COLORBY_EXCLUDE and _S["df"][c].dtype.name == "category":
            col_colors = _global_colors_for_col(_S["df"], c)
            color = col_colors.get(val)
        if color:
            badge = html.Span(val, style={
                "backgroundColor": color,
                "color": "white",
                "padding": "2px 8px",
                "borderRadius": "10px",
                "fontSize": "12px",
                "fontWeight": "600",
                "textShadow": "0 1px 1px rgba(0,0,0,0.3)",
                "display": "inline-block",
            })
        else:
            badge = html.Span(val)
        info_items.append(html.Div([
            html.Span(f"{c}: ", style={"fontWeight": "bold", "color": "#555",
                                        "marginRight": "4px", "fontSize": "12px"}),
            badge,
        ], style={"marginBottom": "3px"}))
    return html.Div([
        html.Div("📍 Point Metadata", style={"fontWeight": "bold", "marginBottom": "6px",
                                              "color": "#4363d8", "fontFamily": "Arial"}),
    ] + info_items)


# ── Click-to-inspect metadata for hierarchy plot ─────────────────────────────
@app.callback(
    Output("hier-click-info", "children"),
    Input("hier-plot", "clickData"),
    State("store-data-loaded", "data"),
    State("store-hier-path", "data"),
    prevent_initial_call=True,
)
def on_hier_point_click(click_data, loaded, path_json):
    """Display full metadata for the clicked point in the hierarchy view."""
    if not loaded or _S["df"] is None or not click_data:
        raise PreventUpdate
    pt = click_data["points"][0]
    trace_idx = pt.get("curveNumber", 0)
    point_idx = pt.get("pointIndex", 0)

    # Resolve back to original df row
    idx_map = _S.get("trace_index_map")
    if idx_map is not None and trace_idx < len(idx_map):
        orig_idx = idx_map[trace_idx][point_idx]
        row = _S["df"].iloc[orig_idx] if orig_idx < len(_S["df"]) else None
    else:
        # Fallback: find closest point by coordinates
        x_click, y_click = pt.get("x"), pt.get("y")
        if x_click is None or y_click is None:
            raise PreventUpdate
        # Apply the same hierarchy filter to search within the correct subset
        filt = _S["df"]
        path_list = path_json if path_json else []
        for c, v in path_list:
            filt = filt[filt[c] == v]
        dist = ((filt["reduced_vector_d1"] - x_click) ** 2 +
                (filt["reduced_vector_d2"] - y_click) ** 2)
        orig_idx = dist.idxmin()
        row = _S["df"].iloc[orig_idx]

    if row is None:
        raise PreventUpdate

    # Build info display with colored pill badges
    meta_cols = [c for c in _S["df"].columns if c not in INTERNAL_COLS]
    info_items = []
    for c in meta_cols:
        val = str(row[c])
        color = None
        if c not in _COLORBY_EXCLUDE and _S["df"][c].dtype.name == "category":
            col_colors = _global_colors_for_col(_S["df"], c)
            color = col_colors.get(val)
        if color:
            badge = html.Span(val, style={
                "backgroundColor": color,
                "color": "white",
                "padding": "2px 8px",
                "borderRadius": "10px",
                "fontSize": "12px",
                "fontWeight": "600",
                "textShadow": "0 1px 1px rgba(0,0,0,0.3)",
                "display": "inline-block",
            })
        else:
            badge = html.Span(val)
        info_items.append(html.Div([
            html.Span(f"{c}: ", style={"fontWeight": "bold", "color": "#555",
                                        "marginRight": "4px", "fontSize": "12px"}),
            badge,
        ], style={"marginBottom": "3px"}))
    return html.Div([
        html.Div("📍 Point Metadata", style={"fontWeight": "bold", "marginBottom": "6px",
                                              "color": "#4363d8", "fontFamily": "Arial"}),
    ] + info_items)


# ── Hierarchy drill / back / reset ────────────────────────────────────────────
@app.callback(
    Output("store-hier-path", "data"),
    Input("btn-drill", "n_clicks"),
    Input("btn-back", "n_clicks"),
    Input("btn-reset-hier", "n_clicks"),
    State("sel-hier-cat", "value"),
    State("store-hier-path", "data"),
    State("store-data-loaded", "data"),
    prevent_initial_call=True,
)
def hier_navigate(drill_n, back_n, reset_n, cat, path, loaded):
    if not loaded or _S["df"] is None:
        raise PreventUpdate
    trigger = ctx.triggered_id
    if trigger == "btn-drill" and cat:
        available = [c for c in CATH_HIERARCHY if c in _S["df"].columns]
        lvl = min(len(path), len(available) - 1)
        col = available[lvl]
        return path + [[col, cat]]
    elif trigger == "btn-back" and path:
        return path[:-1]
    elif trigger == "btn-reset-hier":
        return []
    raise PreventUpdate


@app.callback(
    Output("hier-plot", "figure"),
    Output("sel-hier-cat", "options"),
    Output("sel-hier-cat", "value"),
    Output("sel-col", "options", allow_duplicate=True),
    Output("sel-col", "value", allow_duplicate=True),
    Output("sel-mode", "value", allow_duplicate=True),
    Output("sel-cats", "options", allow_duplicate=True),
    Output("sel-cats", "value", allow_duplicate=True),
    Output("sel-highlight", "options", allow_duplicate=True),
    Output("sel-highlight", "value", allow_duplicate=True),
    Output("sel-density-cat", "options", allow_duplicate=True),
    Output("sel-density-cat", "value", allow_duplicate=True),
    Output("sld-topn", "max", allow_duplicate=True),
    Input("store-hier-path", "data"),
    Input("tabs", "value"),
    Input("store-hier-render-tick", "data"),
    State("sld-topn", "value"),
    State("sld-pt", "value"),
    State("sld-alpha", "value"),
    State("sel-bg", "value"),
    State("sel-col", "value"),
    State("sel-mode", "value"),
    State("sel-cats", "value"),
    State("sel-highlight", "value"),
    State("sel-density-cat", "value"),
    State("store-data-loaded", "data"),
    prevent_initial_call=True,
)
def update_hierarchy(path, tab, render_tick, top_n_val, pt, alpha, bg,
                     viz_col, viz_mode, viz_cats, viz_highlighted, viz_density_cat,
                     loaded):
    if tab != "hierarchy" or not loaded or _S["df"] is None:
        raise PreventUpdate

    trigger = ctx.triggered_id
    is_render = (trigger == "store-hier-render-tick")

    if is_render:
        # Render-button click: use the sidebar's current settings
        fig, hier_cats, avail, _lvl = build_hierarchy(
            _S["df"], path, pt, alpha, bg,
            viz_col=viz_col, viz_mode=viz_mode,
            viz_cats=viz_cats if viz_mode == "distinct" else viz_density_cat,
            viz_highlighted=viz_highlighted if viz_mode == "highlight" else None,
        )
        hier_cat_opts = [{"label": c, "value": c} for c in hier_cats]
        return (fig, hier_cat_opts, no_update,
                no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update)

    # Path change or tab switch: render with CATH-level defaults
    # (don't pass stale sidebar state — let build_hierarchy pick the
    #  new CATH level column automatically)
    fig, hier_cats, avail, _lvl = build_hierarchy(
        _S["df"], path, pt, alpha, bg,
    )
    hier_cat_opts = [{"label": c, "value": c} for c in hier_cats]

    # Update sidebar to reflect filtered subset
    filt = _S.get("hier_filt_df")
    if filt is None or filt.empty:
        return (fig, hier_cat_opts, None,
                no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update)

    # Compute available columns in filtered subset
    filt_cols = []
    filt_card = {}
    for c in filt.columns:
        if c in _COLORBY_EXCLUDE:
            continue
        if filt[c].dtype == object or filt[c].dtype.name == "category":
            n = filt[c].nunique()
            if n > 0:
                filt_cols.append(c)
                filt_card[c] = n
        elif pd.api.types.is_numeric_dtype(filt[c]):
            n = filt[c].nunique()
            if 0 < n <= CARD_THRESHOLD:
                filt_cols.append(c)
                filt_card[c] = n

    col_opts = [{"label": f"{c}  ({filt_card[c]:,})", "value": c} for c in filt_cols]

    # Pick the current CATH level as default colour column
    default_col = avail[_lvl] if _lvl < len(avail) else (filt_cols[0] if filt_cols else None)
    if default_col not in filt_cols and filt_cols:
        default_col = filt_cols[0]

    # Compute category options for the chosen column
    if default_col and default_col in filt.columns:
        vc = filt[default_col].value_counts()
        vc = vc[vc > 0]  # drop 0-count
        cats_sorted = sorted(vc.index)
        card = len(vc)
        cat_opts = [{"label": f"{c}  ({vc[c]:,})", "value": c} for c in cats_sorted]
        mode = "distinct" if card <= CARD_THRESHOLD else "highlight"
        top_n = top_n_val if top_n_val is not None else 10
        if mode == "distinct":
            return (fig, hier_cat_opts, None,
                    col_opts, default_col, mode,
                    cat_opts, cats_sorted, cat_opts, [],
                    cat_opts, None, min(card, 50))
        else:
            top_n_list = vc.head(top_n).index.tolist()
            return (fig, hier_cat_opts, None,
                    col_opts, default_col, mode,
                    cat_opts, [], cat_opts, top_n_list,
                    cat_opts, None, min(card, 50))
    else:
        return (fig, hier_cat_opts, None,
                col_opts, default_col, no_update,
                [], [], [], [],
                [], None, no_update)


# ── Export ────────────────────────────────────────────────────────────────────
@app.callback(
    Output("status-box", "children", allow_duplicate=True),
    Input("btn-export", "n_clicks"),
    State("main-plot", "figure"),
    State("sel-col", "value"),
    State("sel-mode", "value"),
    prevent_initial_call=True,
)
def export_png(n, fig_dict, col, mode):
    if not fig_dict or not _S["paths"]:
        return "⚠️ Nothing to export."
    charts = _S["paths"]["charts"]
    os.makedirs(charts, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"explorer_{col}_{mode}_{ts}.png"
    fpath = os.path.join(charts, fname)
    try:
        fig = go.Figure(fig_dict)
        fig.write_image(fpath, width=1400, height=800, scale=2)
        return f"✅ Exported: {fname}"
    except Exception as e:
        return f"❌ Export error: {e}. Install kaleido: pip install -U kaleido"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  🧬 Protein Embedding Explorer")
    print("  Open http://localhost:8050 in your browser")
    print("=" * 60)
    app.run(debug=True, port=8050)
