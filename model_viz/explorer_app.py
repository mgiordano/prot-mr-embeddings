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
_MAX_POINTS_ZOOMED_OUT = 100_000   # cap when fully zoomed out
_MAX_POINTS_PARTIAL    = 200_000   # cap when partially zoomed in

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
          color_map={})


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _opt(cls, prefix=""):
    """Build Dash dropdown options from a constants class."""
    return [{"label": k, "value": k}
            for k in sorted(dir(cls)) if not k.startswith("_") and (not prefix or k.startswith(prefix))]


def construct_paths(ts, ds_key, filt_key, part_key, exp, combined):
    fds = getattr(dataset_names, ds_key)
    fn = getattr(filters, filt_key).name
    pn = getattr(partition_rules, part_key)["name"]
    date = corpus_utils.get_date_from_formatted_ts(ts)
    vof = os.path.join(INPUT_DATA_ROOT_PATH, fds, date, "vector_output")
    rid = f"{ts}-{fds}-{fn}-{pn}"
    mf = f"{rid}-combined-metadata.tsv" if combined else f"{rid}-metadata.tsv"
    ef = f"{exp}-combined" if combined else exp

    # When model_dim_reduction was run with --filter-col the experiment folder
    # gets a "-filtered" suffix and a dedicated metadata file is saved next to
    # the original.  Detect this and swap to the filtered metadata so that row
    # indices stay aligned with the filtered vectors.
    if ef.endswith("-filtered"):
        if "-cross-filtered" in ef:
            # Cross-model filtered: metadata is <rid>-cross_<tag>-filtered-metadata.tsv.
            # Tag is unknown here, so use a glob to find the file.
            pattern = os.path.join(vof, f"{rid}-cross_*-filtered-metadata.tsv")
            matches = glob.glob(pattern)
            if matches:
                mf = os.path.basename(matches[0])
        elif combined:
            mf = f"{rid}-combined-filtered-metadata.tsv"
        else:
            mf = f"{rid}-filtered-metadata.tsv"

    return dict(
        metadata=os.path.join(vof, mf),
        exp_folder=os.path.join(vof, "experiments", ef),
        charts=os.path.join(vof, "experiments", ef, "charts"),
        run_id=rid,
    )


def find_vector_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in glob.glob(os.path.join(folder, "*.tsv"))
        if any(t in os.path.basename(f) for t in ("vectors_tsne", "vectors_umap"))
    )


def load_data(vec_path, meta_path):
    vdf = pd.read_csv(vec_path, sep="\t", dtype=np.float32)
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
    """Return a decimated subset of *df* for the current view.

    - If *x_range*/*y_range* are given, filter to visible points first.
    - If the visible set exceeds *max_pts*, subsample evenly across spatial
      grid bins so the visual distribution stays representative.
    """
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

    # Grid-stratified subsample: pick up to k points per bin
    bin_x = _S["grid_bin_x"]
    bin_y = _S["grid_bin_y"]
    if x_range is not None and y_range is not None:
        bin_x = bin_x[mask.values] if hasattr(mask, 'values') else bin_x[mask]
        bin_y = bin_y[mask.values] if hasattr(mask, 'values') else bin_y[mask]
    # Composite bin id
    cell = bin_x.astype(np.int32) * _GRID_BINS + bin_y.astype(np.int32)
    # Shuffle and take first max_pts with even bin coverage
    rng = np.random.RandomState(42)
    idx = np.arange(len(visible))
    rng.shuffle(idx)
    # Assign shuffled order, then take first max_pts
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


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_distinct(df, col, selected, pt, alpha, bg):
    """Scatter with distinct color per category (low cardinality)."""
    all_cats = sorted(df[col].unique())
    sel = set(selected) if selected else set(all_cats)
    colors = _S["color_map"]  # precomputed stable color map
    fig = go.Figure()
    trace_index_map = []
    for cat, group in df.groupby(col, observed=True, sort=True):
        if cat not in sel:
            continue
        c = colors.get(cat, "#888888")
        fig.add_trace(go.Scattergl(
            x=group["reduced_vector_d1"].values,
            y=group["reduced_vector_d2"].values,
            mode="markers",
            marker=dict(size=pt, color=c, opacity=alpha, line_width=0),
            name=f"{cat} ({len(group):,})",
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
        hl_df = df[hl_mask]
        for cat, group in hl_df.groupby(col, observed=True, sort=True):
            c = colors.get(cat, "#888888")
            fig.add_trace(go.Scattergl(
                x=group["reduced_vector_d1"].values,
                y=group["reduced_vector_d2"].values,
                mode="markers",
                marker=dict(size=pt, color=c, opacity=alpha,
                            line_width=0),
                name=f"{cat} ({len(group):,})",
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


def build_hierarchy(df, path_json, pt, alpha, bg):
    """CATH hierarchy drill-down. path_json = [[col, val], ...]"""
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

    lvl = min(len(path_list), len(available) - 1)
    col = available[lvl]
    card = filt[col].nunique()
    cats = sorted(filt[col].unique())

    # Use stable color map for this hierarchy column (computed from ALL
    # categories in the full df so drill-down colors stay consistent).
    prev_cmap = _S["color_map"]
    all_hier_cats = sorted(df[col].unique())
    _S["color_map"] = _colors_for(all_hier_cats)

    # Apply same decimation as scatter tab to keep the browser responsive
    filt_dec = _decimate(filt, _MAX_POINTS_ZOOMED_OUT)

    if card <= CARD_THRESHOLD:
        fig = build_distinct(filt_dec, col, None, pt, alpha, bg)
    else:
        top = filt[col].value_counts().head(10).index.tolist()
        fig = build_highlight(filt_dec, col, top, pt, alpha, bg)

    _S["color_map"] = prev_cmap  # restore scatter tab color map

    bc = "All"
    for c, v in path_list:
        bc += f" → {c}: {v}"
    bc += f" → [{col} ({card:,})]"
    fig.update_layout(title=bc)
    return fig, cats, available, lvl


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

_ds_opts = _opt(dataset_names)
_fi_opts = [{"label": k, "value": k} for k in sorted(dir(filters))
            if not k.startswith("_") and k.startswith("MR_")]
_pa_opts = [{"label": k, "value": k} for k in sorted(dir(partition_rules))
            if not k.startswith("_") and k.startswith("PARTITION_")]

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

    # ── Experiment ──
    html.Div(style=S_CARD, children=[
        html.Span("EXPERIMENT", style=S_HDR),
        html.Label("Timestamp", style=S_LBL),
        dcc.Input(id="inp-ts", type="text", placeholder="20241125_102030", style=S_INPUT),
        html.Label("Dataset", style=S_LBL),
        dcc.Dropdown(id="sel-ds", options=_ds_opts,
                     value=_ds_opts[0]["value"] if _ds_opts else None),
        html.Label("MR Filter", style={**S_LBL, "marginTop": "8px"}),
        dcc.Dropdown(id="sel-filt", options=_fi_opts,
                     value="MR_FILTER_NONE" if any(o["value"] == "MR_FILTER_NONE" for o in _fi_opts) else None),
        html.Label("Partition Rule", style={**S_LBL, "marginTop": "8px"}),
        dcc.Dropdown(id="sel-part", options=_pa_opts,
                     value="PARTITION_RULE_USE_ALL" if any(o["value"] == "PARTITION_RULE_USE_ALL" for o in _pa_opts) else None),
        html.Label("Experiment Name", style={**S_LBL, "marginTop": "8px"}),
        dcc.Input(id="inp-exp", type="text", placeholder="tsne_experiment", style=S_INPUT),
        dcc.Checklist(id="chk-comb", options=[{"label": " Combined (with control)", "value": "yes"}], value=[]),
        html.Button("Load Experiment", id="btn-load-exp", style=S_BTN),
    ]),

    # ── File ──
    html.Div(style=S_CARD, children=[
        html.Span("FILE", style=S_HDR),
        dcc.Dropdown(id="sel-file", options=[], placeholder="Load experiment first…"),
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
    ]),
    # hidden stores
    dcc.Store(id="store-hier-path", data=[]),
    dcc.Store(id="store-data-loaded", data=False),
])

app.layout = html.Div(style={"display": "flex", "height": "100vh", "overflow": "hidden"},
                       children=[sidebar, main_area])


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Tab visibility ────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-scatter", "style"),
    Output("tab-hierarchy", "style"),
    Input("tabs", "value"),
)
def toggle_tabs(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (show, hide) if tab == "scatter" else (hide, show)


# ── Load experiment ───────────────────────────────────────────────────────────
@app.callback(
    Output("sel-file", "options"),
    Output("sel-file", "value"),
    Output("status-box", "children", allow_duplicate=True),
    Input("btn-load-exp", "n_clicks"),
    State("inp-ts", "value"), State("sel-ds", "value"),
    State("sel-filt", "value"), State("sel-part", "value"),
    State("inp-exp", "value"), State("chk-comb", "value"),
    prevent_initial_call=True,
)
def load_experiment(n, ts, ds_key, filt, part, exp, comb):
    if not ts or not exp:
        return [], None, "⚠️ Enter timestamp and experiment name."
    try:
        paths = construct_paths(ts.strip(), ds_key, filt, part, exp.strip(), "yes" in comb)
    except Exception as e:
        return [], None, f"❌ {e}"
    _S["paths"] = paths
    if not os.path.isfile(paths["metadata"]):
        return [], None, f"❌ Metadata not found: {paths['metadata']}"
    files = find_vector_files(paths["exp_folder"])
    if not files:
        return [], None, f"❌ No vector files in {paths['exp_folder']}"
    opts = [{"label": os.path.basename(f), "value": f} for f in files]
    return opts, files[0], f"✅ Found {len(files)} vector files."


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
        # Precompute spatial grid index for zoom-level decimation
        _S["grid_x_edges"], _S["grid_y_edges"], _S["grid_bin_x"], _S["grid_bin_y"] = \
            _build_spatial_index(_S["df"])
        _S["trace_index_map"] = None
        _S["last_fig"] = None
        _S["last_fig_args"] = None
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
    State("store-data-loaded", "data"),
    prevent_initial_call=True,
)
def on_col_change(col, loaded):
    if not loaded or not col or _S["df"] is None:
        raise PreventUpdate
    card = _S["card"].get(col, 0)
    mode = "distinct" if card <= CARD_THRESHOLD else "highlight"
    # Single value_counts() call — O(n) instead of O(n*k)
    vc = _S["df"][col].value_counts()
    cats = sorted(vc.index)
    # Precompute stable color map for ALL categories in this column.
    # This ensures the same label always gets the same color regardless
    # of which subset is selected/rendered.
    _S["color_map"] = _colors_for(cats)
    cat_opts = [{"label": f"{c}  ({vc[c]:,})", "value": c} for c in cats]
    # For distinct: pre-select all; for highlight: pre-select top 10
    if mode == "distinct":
        return (mode, cat_opts, cats, cat_opts, [], cat_opts, None, min(card, 50))
    else:
        top10 = vc.head(10).index.tolist()
        return (mode, cat_opts, [], cat_opts, top10, cat_opts, None, min(card, 50))


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


# ── Top-N auto-fill for highlight mode ────────────────────────────────────────
@app.callback(
    Output("sel-highlight", "value", allow_duplicate=True),
    Input("sld-topn", "value"),
    State("sel-col", "value"),
    State("sel-mode", "value"),
    prevent_initial_call=True,
)
def topn_fill(n, col, mode):
    if mode != "highlight" or not col or _S["df"] is None:
        raise PreventUpdate
    if n == 0:
        return []
    return _S["df"][col].value_counts().head(n).index.tolist()


# ── Main scatter plot (button-triggered) ─────────────────────────────────────
@app.callback(
    Output("main-plot", "figure"),
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
    prevent_initial_call=True,
)
def update_main_plot(_n, mode, cats, highlighted, density_cat, pt, alpha, bg, col, loaded, relayout):
    if not loaded or _S["df"] is None or not col:
        return go.Figure()
    # Apply decimation to cap point count for the initial render.
    # Zoom/pan is handled client-side by WebGL — no server round-trip needed.
    df_dec = _decimate(_S["df"], _MAX_POINTS_ZOOMED_OUT)
    if mode == "distinct":
        fig = build_distinct(df_dec, col, cats or None, pt, alpha, bg)
    elif mode == "highlight":
        fig = build_highlight(df_dec, col, highlighted or [], pt, alpha, bg)
    else:
        fig = build_density(df_dec, col, density_cat, bg)

    # Preserve previous zoom/pan so re-rendering with a new category keeps
    # the same view region (useful for comparing regions across labels).
    if relayout and "xaxis.range[0]" in relayout and "yaxis.range[0]" in relayout:
        fig.update_xaxes(range=[relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]])
        fig.update_yaxes(range=[relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]])

    return fig


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
            all_cats = sorted(_S["df"][c].cat.categories)
            col_colors = _colors_for(all_cats)
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
    Input("store-hier-path", "data"),
    Input("tabs", "value"),
    State("sld-pt", "value"),
    State("sld-alpha", "value"),
    State("sel-bg", "value"),
    State("store-data-loaded", "data"),
)
def update_hierarchy(path, tab, pt, alpha, bg, loaded):
    if tab != "hierarchy" or not loaded or _S["df"] is None:
        raise PreventUpdate
    fig, cats, avail, _lvl = build_hierarchy(_S["df"], path, pt, alpha, bg)
    cat_opts = [{"label": c, "value": c} for c in cats]
    return fig, cat_opts, None


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
