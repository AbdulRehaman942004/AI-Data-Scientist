import io
import json
import os
import re
import uuid
import glob
import shutil
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from llm import create_agent

load_dotenv()

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS – Dark Tech Theme ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-card:      #1c2128;
    --border:       #30363d;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --accent:       #1f6feb;
    --accent-glow:  #388bfd;
    --accent-lt:    #58a6ff;
    --success:      #3fb950;
    --warning:      #d29922;
    --error:        #f85149;
    --purple:       #8b5cf6;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main .block-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

header[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stToolbar"] { background-color: var(--bg-primary) !important; }
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── App header ── */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 0 14px; border-bottom: 1px solid var(--border);
    margin-bottom: 26px;
}
.app-header-icon {
    width: 44px; height: 44px; flex-shrink: 0;
    background: linear-gradient(135deg, #1f6feb 0%, #8b5cf6 100%);
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 22px;
}
.app-header-title { font-size: 1.6rem; font-weight: 700; color: var(--text-primary); margin: 0; }
.app-header-sub   { font-size: 0.82rem; color: var(--text-muted); margin: 3px 0 0; }

/* ── Sidebar brand ── */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 0 18px; border-bottom: 1px solid var(--border); margin-bottom: 18px;
}
.sidebar-brand-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #1f6feb, #8b5cf6);
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 18px; flex-shrink: 0;
}
.sidebar-brand-name    { font-size: 0.92rem; font-weight: 700; }
.sidebar-brand-version { font-size: 0.68rem; color: var(--text-muted) !important; }

/* ── Section heading ── */
.section-heading {
    display: flex; align-items: center; gap: 9px;
    font-size: 1rem; font-weight: 600; color: var(--text-primary);
    padding: 8px 0; border-bottom: 1px solid var(--border); margin-bottom: 14px;
}

/* ── Metric card grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 12px; margin: 14px 0 22px;
}
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px; position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--purple));
}
.metric-card.green::before { background: linear-gradient(90deg, var(--success), #58a6ff); }
.metric-card.red::before   { background: linear-gradient(90deg, var(--error), var(--warning)); }
.metric-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 5px; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); line-height: 1; }
.metric-sub   { font-size: 0.7rem; color: var(--text-muted); margin-top: 4px; }
.metric-delta-good { font-size: 0.72rem; color: var(--success); margin-top: 4px; }
.metric-delta-bad  { font-size: 0.72rem; color: var(--error);   margin-top: 4px; }

/* ── Step badges ── */
.step-badges { display: flex; flex-wrap: wrap; gap: 8px; padding: 6px 0 4px; }
.step-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: #161b22; border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 11px; font-size: 0.8rem;
    color: var(--text-muted); white-space: nowrap;
}
.step-badge.done   { border-color: var(--success); color: var(--success);   background: #0d1f12; }
.step-badge.active { border-color: var(--accent);  color: var(--accent-lt); background: #0d1b30; }

/* ── Viz label ── */
.viz-label {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--text-muted); margin-bottom: 5px; padding-left: 1px;
}
.group-label {
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--text-muted); margin: 20px 0 8px;
}

/* ── Feature cards (welcome) — CSS Grid siblings, always equal height ── */
.feature-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 22px 20px;
}

/* ── Cleaning action row ── */
.action-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 12px; border-radius: 7px;
    background: var(--bg-card); border: 1px solid var(--border);
    margin-bottom: 6px; font-size: 0.85rem;
}
.action-col { font-weight: 600; color: var(--accent-lt); min-width: 120px; }
.action-desc { color: var(--text-muted); }

/* ── Clarification card ── */
.clarify-card {
    background: #1a160a; border: 1px solid var(--warning);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 20px;
}
.clarify-title { color: var(--warning); font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
.clarify-sub   { color: var(--text-muted); font-size: 0.82rem; }

/* ── Buttons (regular + form submit) ── */
[data-testid="stButton"] > button,
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    letter-spacing: 0.02em !important; width: 100% !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover,
[data-testid="stFormSubmitButton"] > button:hover { opacity: 0.85 !important; }

/* Download button — keep style consistent */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    width: 100% !important; transition: opacity 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover { opacity: 0.85 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-muted) !important;
    font-weight: 500 !important; border-radius: 6px 6px 0 0 !important; padding: 8px 18px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent-lt) !important; border-bottom: 2px solid var(--accent-lt) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] label {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important; background: var(--bg-card) !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small { color: var(--text-muted) !important; }
[data-testid="stFileUploaderDropzoneInput"] + div button,
[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploader"] button {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileData"] {
    background: #21262d !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--text-primary) !important;
}
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileSize"] { color: var(--text-primary) !important; }

/* ── Text inputs & text areas ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--bg-card) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}
/* Labels above text inputs / textareas */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] [data-testid="stWidgetLabel"] p,
[data-testid="stTextArea"] label,
[data-testid="stTextArea"] [data-testid="stWidgetLabel"] p {
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
/* Placeholder text */
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #484f58 !important;
    opacity: 1 !important;
}

/* ── Radio buttons ── */
/* Step 1: force ALL text inside the radio widget to white, no-transform */
[data-testid="stRadio"] * {
    color: #e6edf3 !important;
    text-transform: none !important;
}
/* Step 2: only the widget label ("Select an option:") gets muted + uppercase */
[data-testid="stRadio"] [data-testid="stWidgetLabel"],
[data-testid="stRadio"] [data-testid="stWidgetLabel"] * {
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
}
/* Step 3: option row layout */
[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 6px !important; display: flex !important; flex-direction: column !important;
}
/* Step 4: each option card */
[data-testid="stRadio"] div[data-baseweb="radio"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    transition: border-color 0.15s, background 0.15s !important;
}
[data-testid="stRadio"] div[data-baseweb="radio"]:hover {
    border-color: var(--accent) !important;
    background: #0d1b30 !important;
}
/* Step 5: unselected circle */
[data-testid="stRadio"] [role="radio"] {
    border-color: #484f58 !important;
    background: #161b22 !important;
}
/* Step 6: selected circle — kill the base-web red/coral */
[data-testid="stRadio"] [role="radio"][aria-checked="true"],
[data-testid="stRadio"] [role="radio"][aria-checked="true"] > div {
    border-color: var(--accent) !important;
    background: var(--accent) !important;
}
/* Step 7: highlight selected row */
[data-testid="stRadio"] div[data-baseweb="radio"]:has([aria-checked="true"]) {
    border-color: var(--accent) !important;
    background: #0d1b30 !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] * { color: #e6edf3 !important; text-transform: none !important; }
[data-testid="stMultiSelect"] [data-testid="stWidgetLabel"],
[data-testid="stMultiSelect"] [data-testid="stWidgetLabel"] * {
    color: var(--text-muted) !important; font-size: 0.78rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: #0d1b30 !important;
    border: 1px solid var(--accent) !important;
    border-radius: 5px !important;
    color: var(--accent-lt) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden; }

/* ── Expander ── */
[data-testid="stExpander"], details {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary, details > summary {
    background: var(--bg-card) !important; color: var(--text-primary) !important;
    border-radius: 10px !important; padding: 10px 14px !important;
}
[data-testid="stExpanderDetails"], details > div {
    background: var(--bg-card) !important; padding: 8px 14px 12px !important;
}
[data-testid="stExpander"] *, details * { color: var(--text-primary) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; border-left-width: 3px !important; }
[data-testid="stAlert"][kind="success"],
div[data-baseweb="notification"][kind="positive"] {
    background: #0d1f12 !important; border-color: var(--success) !important; color: var(--success) !important;
}
[data-testid="stAlert"][kind="info"],
div[data-baseweb="notification"][kind="info"] {
    background: #0d1b30 !important; border-color: var(--accent) !important; color: var(--accent-lt) !important;
}
[data-testid="stAlert"][kind="warning"],
div[data-baseweb="notification"][kind="warning"] {
    background: #1f1700 !important; border-color: var(--warning) !important; color: var(--warning) !important;
}
[data-testid="stAlert"][kind="error"],
div[data-baseweb="notification"][kind="negative"] {
    background: #1f0808 !important; border-color: var(--error) !important; color: var(--error) !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: inherit !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

hr { border-color: var(--border) !important; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.pulse-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--accent); display: inline-block;
    animation: pulse 1.2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)


# ─── SVG Icon Factory ─────────────────────────────────────────────────────────
def _svg(d: str, size: int = 16, color: str = "#58a6ff") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{d}</svg>'
    )


IC_DB      = '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
IC_CHART   = '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
IC_FILE    = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>'
IC_GRID    = '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>'
IC_WAVE    = '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
IC_UPLOAD  = '<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'
IC_CHECK   = '<polyline points="20 6 9 17 4 12"/>'
IC_INFO    = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
IC_CLEAN   = '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>'
IC_DOWNLOAD= '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>'
IC_ALERT   = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
IC_CPU     = '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'


def section_heading(label: str, icon_d: str, color: str = "#58a6ff"):
    st.markdown(
        f'<div class="section-heading">{_svg(icon_d, 17, color)}&nbsp;{label}</div>',
        unsafe_allow_html=True,
    )


# ─── Tool-call label map ──────────────────────────────────────────────────────
TOOL_LABELS = {
    "load_dataset":                "Loading dataset",
    "get_dataframe_info":          "Inspecting structure",
    "get_statistics":              "Computing statistics",
    "get_missing_values":          "Analysing missing values",
    "detect_outliers":             "Detecting outliers",
    "handle_missing_values":       "Cleaning dataset",
    "generate_before_after_plots": "Creating before/after comparison",
    "generate_visualization":      "Generating visualisation",
}


# ─── Session State Defaults ───────────────────────────────────────────────────
_defaults = {
    "report":           None,
    "df":               None,
    "df_cleaned":       None,
    "viz_paths":        [],
    "comparison_paths": [],
    "tool_log":         [],
    "error":            None,
    "running":          False,
    "cleaned_path":     None,
    "cleaning_log":     [],
    "before_after_stats": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ──────────────────────────────────────────────────────────────────
def collect_visualizations(output_dir: str):
    """Return (regular_plots, comparison_plots) separately."""
    all_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        all_paths.extend(glob.glob(os.path.join(output_dir, ext)))
    all_paths = sorted(all_paths)
    regular = [p for p in all_paths if not os.path.basename(p).startswith("comparison_")]
    comparison = [p for p in all_paths if os.path.basename(p).startswith("comparison_")]
    return regular, comparison


def _metric_card(label: str, value: str, sub: str = "", delta: str = "",
                 delta_good: bool = True, variant: str = "default"):
    delta_cls = "metric-delta-good" if delta_good else "metric-delta-bad"
    card_cls = {"default": "", "green": " green", "red": " red"}.get(variant, "")
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card{card_cls}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}{delta_html}'
        f'</div>'
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">&#9670;</div>
        <div>
            <div class="sidebar-brand-name">AI Data Scientist</div>
            <div class="sidebar-brand-version">v2.0 &nbsp;&bull;&nbsp; GPT-4o-mini</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-heading" style="font-size:0.8rem;padding:4px 0 8px">'
        f'{_svg(IC_UPLOAD,14)}&nbsp;Dataset Upload</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    file_ok = False
    if uploaded_file is not None:
        size_bytes = uploaded_file.size
        size_mb = size_bytes / (1024 * 1024)
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_mb:.2f} MB"
        if size_mb > 50:
            st.error(f"File exceeds 50 MB limit ({size_str}).", icon=None)
        else:
            file_ok = True
            st.success(f"{uploaded_file.name}  ({size_str})", icon=None)

    st.divider()

    run_btn = st.button(
        "Run EDA Analysis",
        disabled=(
            not file_ok
            or st.session_state.get("workflow_status") in ("running", "waiting", "resuming")
        ),
        use_container_width=True,
    )

    # Download cleaned dataset (sidebar shortcut)
    if st.session_state.df_cleaned is not None:
        st.divider()
        cleaned_csv = st.session_state.df_cleaned.to_csv(index=False).encode("utf-8")
        fname = (
            os.path.splitext(uploaded_file.name)[0] + "_cleaned.csv"
            if uploaded_file is not None
            else "cleaned_dataset.csv"
        )
        st.download_button(
            label="Download Cleaned Dataset",
            data=cleaned_csv,
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.markdown(
        f'<div style="font-size:0.74rem;color:#8b949e;line-height:1.7">'
        f'<b style="color:#c9d1d9;font-size:0.78rem">How it works</b><br>'
        f'1. Upload a CSV dataset (max 50 MB)<br>'
        f'2. Click <b>Run EDA Analysis</b><br>'
        f'3. Answer any clarification questions<br>'
        f'4. Review AI report, charts &amp; stats<br>'
        f'5. Download the cleaned dataset'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">&#9670;</div>
    <div>
        <h1 class="app-header-title">AI Data Scientist</h1>
        <p class="app-header-sub">
            Automated Exploratory Data Analysis &nbsp;&bull;&nbsp; GPT-4o-mini &nbsp;&bull;&nbsp; Human-in-the-Loop
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Load DataFrame Preview ───────────────────────────────────────────────────
if uploaded_file is not None and file_ok:
    if st.session_state.df is None or getattr(
        st.session_state, "_last_file", None
    ) != uploaded_file.name:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state["_last_file"] = uploaded_file.name
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.session_state.df = None


# ─── Graph Initialization ─────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY", "")

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = "init"

config = {"configurable": {"thread_id": st.session_state.thread_id}}


# ─── Run Trigger ──────────────────────────────────────────────────────────────
if run_btn and file_ok:
    tmp_dir = tempfile.mkdtemp(prefix="ai_ds_")
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    out_dir = os.path.join(os.getcwd(), "outputs")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Reset all session state for a fresh run
    st.session_state.tmp_path         = tmp_path
    st.session_state.out_dir          = out_dir
    st.session_state.report           = None
    st.session_state.viz_paths        = []
    st.session_state.comparison_paths = []
    st.session_state.tool_log         = []
    st.session_state.error            = None
    st.session_state.cleaned_path     = None
    st.session_state.cleaning_log     = []
    st.session_state.before_after_stats = None
    st.session_state.df_cleaned       = None
    st.session_state.thread_id        = str(uuid.uuid4())
    config["configurable"]["thread_id"] = st.session_state.thread_id
    st.session_state.workflow_status  = "running"
    st.rerun()


# ─── Execution & Streaming ────────────────────────────────────────────────────
if st.session_state.workflow_status in ("running", "resuming"):
    try:
        agent = create_agent(api_key=api_key, memory=st.session_state.memory)

        with st.status("AI Data Scientist is thinking...", expanded=True) as status_box:
            step_area = st.empty()

            def _render_steps():
                if not st.session_state.tool_log:
                    return
                badges = [
                    f'<div class="step-badge done">'
                    f'{_svg(IC_CHECK,13,"#3fb950")}&nbsp;{TOOL_LABELS.get(t, t)}</div>'
                    for t in st.session_state.tool_log
                ]
                step_area.markdown(
                    f'<div class="step-badges">{"".join(badges)}</div>',
                    unsafe_allow_html=True,
                )

            _render_steps()

            inputs = None
            if st.session_state.workflow_status == "running":
                inputs = {
                    "messages": [(
                        "human",
                        f"Please perform a full EDA on the dataset at: {st.session_state.tmp_path}",
                    )]
                }

            for event in agent.stream(inputs, config=config, stream_mode="updates"):

                # ── Capture automated tool results ──────────────────────────
                if "automated_tools" in event:
                    for msg in event["automated_tools"]["messages"]:
                        tool_name = getattr(msg, "name", None)
                        if tool_name:
                            st.session_state.tool_log.append(tool_name)

                        # Parse handle_missing_values result
                        if tool_name == "handle_missing_values":
                            try:
                                result = json.loads(msg.content)
                                if "cleaned_file_path" in result:
                                    st.session_state.cleaned_path = result["cleaned_file_path"]
                                if "actions_applied" in result:
                                    st.session_state.cleaning_log = result["actions_applied"]
                            except Exception:
                                pass

                        # Parse generate_before_after_plots result
                        if tool_name == "generate_before_after_plots":
                            try:
                                result = json.loads(msg.content)
                                if "before_stats" in result:
                                    st.session_state.before_after_stats = result
                            except Exception:
                                pass

                    _render_steps()

                # ── Capture final report ──────────────────────────────────
                if "agent" in event:
                    msg = event["agent"]["messages"][0]
                    if getattr(msg, "content", None):
                        st.session_state.report = msg.content

            # ── Check next graph state ────────────────────────────────────
            state = agent.get_state(config)
            if state.next and "ask_human" in state.next:
                st.session_state.workflow_status = "waiting"
                status_box.update(
                    label="Paused — clarification needed",
                    state="complete",
                    expanded=False,
                )
            else:
                st.session_state.workflow_status = "complete"
                regular, comparison = collect_visualizations(st.session_state.out_dir)
                st.session_state.viz_paths        = regular
                st.session_state.comparison_paths = comparison

                # Load cleaned dataframe if it exists
                cp = st.session_state.cleaned_path
                if cp and os.path.exists(cp):
                    try:
                        st.session_state.df_cleaned = pd.read_csv(cp)
                    except Exception:
                        pass

                status_box.update(label="Analysis complete", state="complete", expanded=False)

        st.rerun()

    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.workflow_status = "error"
        st.error(f"Analysis failed: {exc}")


# ─── Human-in-the-Loop Clarification UI ──────────────────────────────────────
if st.session_state.workflow_status == "waiting":
    agent = create_agent(api_key=api_key, memory=st.session_state.memory)
    state = agent.get_state(config)
    last_msg = state.values["messages"][-1]

    ask_call = next(
        (tc for tc in last_msg.tool_calls if tc["name"] == "ask_human"), None
    )

    if ask_call:
        args = ask_call["args"]
        question      = args.get("question", "I need some clarification to proceed.")
        question_type = args.get("question_type", "single_choice")
        options       = args.get("options", [])

        st.markdown(
            '<div class="clarify-card">'
            '<div class="clarify-title">Clarification Needed</div>'
            '<div class="clarify-sub">The AI paused to ask you a question before proceeding.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:1rem;font-weight:600;color:#e6edf3;margin-bottom:16px">'
            f'{question}</div>',
            unsafe_allow_html=True,
        )

        with st.form("clarification_form"):

            answer = None

            if question_type == "yes_no":
                choice = st.radio("Your answer:", ["Yes", "No"], horizontal=True)
                answer = choice

            elif question_type == "multi_choice" and options:
                chosen = st.multiselect("Select all that apply:", options)
                answer = ", ".join(chosen) if chosen else None

            elif question_type == "text":
                answer = st.text_area(
                    "Your response:",
                    placeholder="Type your answer here…",
                    height=100,
                )

            else:  # single_choice (default)
                if options:
                    choice = st.radio("Select an option:", options)
                    answer = choice
                else:
                    answer = st.text_input(
                        "Your response:",
                        placeholder="Type your answer here…",
                    )

            # Always allow an override
            override = st.text_input(
                "Or type a custom answer (overrides selection above):",
                placeholder="Optional — leave blank to use selection",
            )

            submitted = st.form_submit_button("Submit & Resume Analysis")

            if submitted:
                final_answer = override.strip() if override.strip() else (answer or "")
                if not final_answer:
                    st.warning("Please provide an answer before continuing.")
                else:
                    tool_msg = {
                        "role": "tool",
                        "content": str(final_answer),
                        "tool_call_id": ask_call["id"],
                        "name": "ask_human",
                    }
                    agent.update_state(
                        config, {"messages": [tool_msg]}, as_node="ask_human"
                    )
                    st.session_state.workflow_status = "resuming"
                    st.rerun()


# ─── Content Area ─────────────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.df

if df is None and st.session_state.report is None and not st.session_state.error:
    # ── Welcome / empty state ──────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:52px 20px 36px">
        <div style="font-size:46px;margin-bottom:14px">&#9670;</div>
        <h2 style="font-weight:700;color:#e6edf3;margin-bottom:8px;font-size:1.45rem">
            Upload a dataset to get started
        </h2>
        <p style="color:#8b949e;max-width:500px;margin:0 auto;font-size:0.92rem;line-height:1.6">
            Upload any CSV file using the sidebar, then click
            <strong style="color:#58a6ff">Run EDA Analysis</strong>
            to generate a full AI-powered data analysis report with cleaning, visualizations, and insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    features = [
        (IC_DB,    "#58a6ff", "Full EDA Pipeline",
         "Dataset loading, structure inspection, statistical analysis, and outlier detection."),
        (IC_CLEAN, "#8b5cf6", "Intelligent Cleaning",
         "Context-aware missing value handling with before/after comparison and download."),
        (IC_CHART, "#3fb950", "Smart Visualizations",
         "Histograms, violin plots, scatter plots, heatmaps — selected intelligently."),
        (IC_FILE,  "#d29922", "AI Narrative Report",
         "LLM-generated report covering findings, correlations, and modeling recommendations."),
    ]
    cards_html = "".join(
        f'<div class="feature-card">'
        f'<div style="margin-bottom:12px">{_svg(ic, 24, clr)}</div>'
        f'<div style="font-weight:600;font-size:0.93rem;color:#e6edf3;margin-bottom:7px">{title}</div>'
        f'<div style="font-size:0.82rem;color:#8b949e;line-height:1.6">{desc}</div>'
        f'</div>'
        for ic, clr, title, desc in features
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;align-items:stretch">'
        f'{cards_html}</div>',
        unsafe_allow_html=True,
    )

else:
    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dataset Preview",
        "Analysis Report",
        "Visualizations",
        "Cleaning Report",
        "Statistics",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – Dataset Preview
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        if df is not None:
            rows, cols_ = df.shape
            num_cols = len(df.select_dtypes(include="number").columns)
            cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
            miss     = int(df.isna().sum().sum())
            miss_pct = round(miss / df.size * 100, 1) if df.size > 0 else 0
            mem_kb   = round(df.memory_usage(deep=True).sum() / 1024, 1)
            dups     = int(df.duplicated().sum())

            st.markdown(
                '<div class="metric-grid">'
                + _metric_card("Rows",            f"{rows:,}")
                + _metric_card("Columns",         f"{cols_}")
                + _metric_card("Numeric",         f"{num_cols}")
                + _metric_card("Categorical",     f"{cat_cols}")
                + _metric_card("Missing Cells",   f"{miss:,}",  sub=f"{miss_pct}% of total",
                               variant="red" if miss > 0 else "default")
                + _metric_card("Duplicates",      f"{dups:,}",
                               variant="red" if dups > 0 else "default")
                + _metric_card("Memory",          f"{mem_kb:,}", sub="KB")
                + '</div>',
                unsafe_allow_html=True,
            )

            section_heading("Data Preview — first 100 rows", IC_DB)
            st.dataframe(df.head(100), use_container_width=True, height=380)

            section_heading("Column Information", IC_GRID)
            col_info = pd.DataFrame({
                "Column":     df.columns,
                "Data Type":  df.dtypes.astype(str).values,
                "Non-Null":   df.notna().sum().values,
                "Null Count": df.isna().sum().values,
                "Null %":     (df.isna().mean() * 100).round(2).values,
                "Unique":     df.nunique().values,
                "Sample Value": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
                                 for c in df.columns],
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        else:
            st.info("Upload a dataset to see a preview.", icon=None)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – Analysis Report
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        if st.session_state.error:
            st.error(f"Analysis failed: {st.session_state.error}")

        elif st.session_state.report:
            section_heading("AI-Generated EDA Report", IC_FILE)

            # Tool execution trace
            tlog = st.session_state.tool_log
            if tlog:
                with st.expander(f"Tool execution trace  ({len(tlog)} calls)", expanded=False):
                    unique_calls = list(dict.fromkeys(tlog))
                    for tname in unique_calls:
                        lbl   = TOOL_LABELS.get(tname, tname)
                        count = tlog.count(tname)
                        badge = lbl + (f" ×{count}" if count > 1 else "")
                        st.markdown(
                            f'<div class="step-badge done" style="margin-bottom:5px">'
                            f'{_svg(IC_CHECK,13,"#3fb950")}&nbsp;{badge}</div>',
                            unsafe_allow_html=True,
                        )

            # Strip broken local image tags — charts live in the Visualizations tab
            clean_report = re.sub(r'!\[.*?\]\(.*?\)', '', st.session_state.report)
            st.markdown(clean_report)

        else:
            st.markdown(
                f'<div style="text-align:center;padding:44px;color:#8b949e">'
                f'{_svg(IC_INFO, 26, "#8b949e")}'
                f'<p style="margin-top:10px;font-size:0.92rem">'
                f'Run the analysis to see the AI-generated report here.</p></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – Visualizations
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        viz_paths = st.session_state.viz_paths
        if viz_paths:
            section_heading(f"{len(viz_paths)} Visualization(s) Generated", IC_CHART)

            groups: dict = {}
            for p in viz_paths:
                base   = os.path.basename(p)
                prefix = base.split("_")[0].capitalize()
                groups.setdefault(prefix, []).append(p)

            for group_name, paths in groups.items():
                st.markdown(
                    f'<div class="group-label">{group_name}</div>',
                    unsafe_allow_html=True,
                )
                n = min(len(paths), 3)
                cols_v = st.columns(n, gap="medium")
                for i, path in enumerate(paths):
                    with cols_v[i % n]:
                        label = (
                            os.path.basename(path)
                            .replace(".png", "")
                            .replace("_", " ")
                            .title()
                        )
                        st.markdown(
                            f'<div class="viz-label">{label}</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(path, use_container_width=True)
        else:
            st.markdown(
                f'<div style="text-align:center;padding:44px;color:#8b949e">'
                f'{_svg(IC_CHART, 26, "#8b949e")}'
                f'<p style="margin-top:10px;font-size:0.92rem">'
                f'Visualizations will appear here once the analysis is complete.</p></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – Cleaning Report
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        df_orig    = st.session_state.df
        df_cleaned = st.session_state.df_cleaned
        cleaning_log = st.session_state.cleaning_log
        ba_stats  = st.session_state.before_after_stats

        status_complete = st.session_state.workflow_status == "complete"

        if not status_complete and df_orig is not None:
            st.info("Run the analysis to see the cleaning report.", icon=None)

        elif df_orig is not None and df_cleaned is None and status_complete:
            st.success(
                "No cleaning was needed — the dataset had no missing values. It is already clean!",
                icon=None,
            )

        elif df_orig is not None and df_cleaned is not None:
            # ── Before / After metric cards ──────────────────────────────────
            section_heading("Before vs After Cleaning", IC_CLEAN, color="#3fb950")

            r_before = len(df_orig)
            r_after  = len(df_cleaned)
            c_before = len(df_orig.columns)
            c_after  = len(df_cleaned.columns)
            m_before = int(df_orig.isna().sum().sum())
            m_after  = int(df_cleaned.isna().sum().sum())

            row_delta = f"−{r_before - r_after:,} rows removed" if r_before > r_after else "No rows removed"
            col_delta = f"−{c_before - c_after} columns dropped" if c_before > c_after else "No columns dropped"
            miss_delta = f"−{m_before - m_after:,} cells fixed" if m_before > m_after else "No missing values fixed"

            st.markdown(
                '<div class="metric-grid">'
                + _metric_card("Rows Before",    f"{r_before:,}")
                + _metric_card("Rows After",     f"{r_after:,}",
                               delta=row_delta, delta_good=(r_before == r_after),
                               variant="green")
                + _metric_card("Columns Before", f"{c_before}")
                + _metric_card("Columns After",  f"{c_after}",
                               delta=col_delta, delta_good=(c_before == c_after),
                               variant="green")
                + _metric_card("Missing Before", f"{m_before:,}",
                               sub=f"{round(m_before/max(df_orig.size,1)*100,1)}% of data",
                               variant="red" if m_before > 0 else "default")
                + _metric_card("Missing After",  f"{m_after:,}",
                               sub=f"{round(m_after/max(df_cleaned.size,1)*100,1)}% of data",
                               delta=miss_delta, delta_good=True,
                               variant="green")
                + '</div>',
                unsafe_allow_html=True,
            )

            # ── Cleaning actions table ────────────────────────────────────────
            if cleaning_log:
                section_heading("Cleaning Actions Applied", IC_CHECK, color="#3fb950")
                for action in cleaning_log:
                    col_name = action.get("column", "")
                    desc     = action.get("action", "")
                    st.markdown(
                        f'<div class="action-row">'
                        f'<div class="action-col">{col_name}</div>'
                        f'<div class="action-desc">{desc}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Download cleaned dataset ──────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section_heading("Download Cleaned Dataset", IC_DOWNLOAD, color="#3fb950")

            cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8")
            fname = (
                os.path.splitext(uploaded_file.name)[0] + "_cleaned.csv"
                if uploaded_file is not None
                else "cleaned_dataset.csv"
            )

            col_dl, col_info2 = st.columns([1, 2], gap="medium")
            with col_dl:
                st.download_button(
                    label=f"Download  {fname}",
                    data=cleaned_csv,
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_info2:
                st.markdown(
                    f'<div style="font-size:0.85rem;color:#8b949e;padding-top:6px">'
                    f'{r_after:,} rows &nbsp;·&nbsp; {c_after} columns &nbsp;·&nbsp; '
                    f'{round(len(cleaned_csv)/1024,1)} KB</div>',
                    unsafe_allow_html=True,
                )

            # ── Before/After comparison visualizations ─────────────────────
            comparison_paths = st.session_state.comparison_paths
            if comparison_paths:
                st.markdown("<br>", unsafe_allow_html=True)
                section_heading(
                    f"{len(comparison_paths)} Before/After Comparison Chart(s)",
                    IC_CHART, color="#8b5cf6",
                )
                n_cols = min(len(comparison_paths), 2)
                cols_comp = st.columns(n_cols, gap="medium")
                for i, path in enumerate(comparison_paths):
                    with cols_comp[i % n_cols]:
                        label = (
                            os.path.basename(path)
                            .replace("comparison_", "")
                            .replace(".png", "")
                            .replace("_", " ")
                            .title()
                        )
                        st.markdown(
                            f'<div class="viz-label">{label}</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(path, use_container_width=True)

            # ── Cleaned dataset preview ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section_heading("Cleaned Dataset Preview", IC_DB, color="#58a6ff")
            st.dataframe(df_cleaned.head(100), use_container_width=True, height=340)

        else:
            st.markdown(
                f'<div style="text-align:center;padding:44px;color:#8b949e">'
                f'{_svg(IC_CLEAN, 26, "#8b949e")}'
                f'<p style="margin-top:10px;font-size:0.92rem">'
                f'Cleaning report will appear here after analysis is complete.</p></div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 – Statistics
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        # Use cleaned df if available, otherwise original
        stats_df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else df

        if stats_df is not None:
            num_df = stats_df.select_dtypes(include="number")
            cat_df = stats_df.select_dtypes(include=["object", "category"])

            if st.session_state.df_cleaned is not None:
                st.info(
                    "Showing statistics for the **cleaned** dataset. "
                    "Switch to Dataset Preview to see the original.",
                    icon=None,
                )

            if not num_df.empty:
                section_heading("Numeric Column Statistics", IC_WAVE)
                desc = num_df.describe().T.round(4)
                desc["skewness"] = num_df.skew().round(4)
                desc["kurtosis"] = num_df.kurt().round(4)
                desc["missing"] = num_df.isna().sum()
                st.dataframe(desc, use_container_width=True)

            if not cat_df.empty:
                section_heading("Categorical Columns", IC_GRID)
                rows_cat = []
                for c in cat_df.columns:
                    vc = cat_df[c].value_counts()
                    rows_cat.append({
                        "Column":        c,
                        "Unique Values": int(cat_df[c].nunique()),
                        "Top Value":     str(vc.index[0]) if len(vc) > 0 else "N/A",
                        "Top Frequency": int(vc.iloc[0])  if len(vc) > 0 else 0,
                        "Top %":         round(vc.iloc[0] / len(cat_df) * 100, 1) if len(vc) > 0 else 0,
                        "Null %":        round(cat_df[c].isna().mean() * 100, 2),
                    })
                st.dataframe(
                    pd.DataFrame(rows_cat),
                    use_container_width=True,
                    hide_index=True,
                )

            if not num_df.empty and len(num_df.columns) > 1:
                section_heading("Correlation Matrix", IC_CHART)
                corr = num_df.corr().round(4)
                st.dataframe(corr, use_container_width=True)

                # Highlight strongest correlations
                section_heading("Strongest Correlations", IC_WAVE, color="#8b5cf6")
                pairs = []
                cols_n = corr.columns.tolist()
                for i, c1 in enumerate(cols_n):
                    for c2 in cols_n[i+1:]:
                        pairs.append({
                            "Column A": c1,
                            "Column B": c2,
                            "Correlation": round(float(corr.loc[c1, c2]), 4),
                        })
                pairs_df = pd.DataFrame(pairs)
                pairs_df["Abs Corr"] = pairs_df["Correlation"].abs()
                pairs_df = pairs_df.sort_values("Abs Corr", ascending=False).drop(
                    columns="Abs Corr"
                ).head(15)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)

        else:
            st.info("Upload a dataset to see statistics here.", icon=None)
