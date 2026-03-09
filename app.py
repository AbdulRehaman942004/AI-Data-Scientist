import os
import glob
import shutil
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

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
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root palette ── */
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

/* ── App background ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main .block-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Streamlit top header bar ── */
header[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stToolbar"] {
    background-color: var(--bg-primary) !important;
}
/* Hide the top decoration bar */
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
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
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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
.metric-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 5px; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); line-height: 1; }
.metric-sub   { font-size: 0.7rem; color: var(--text-muted); margin-top: 4px; }

/* ── Step badges container ── */
.step-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 6px 0 4px;
}
/* ── Step badge ── */
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

/* ── Feature cards (welcome) ── */
.feature-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 22px 20px;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    letter-spacing: 0.02em !important; width: 100% !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

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

/* ── File uploader – full override ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] label {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}
/* The dashed drop-zone box */
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card) !important;
}
/* Small text inside the dropzone */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text-muted) !important;
}
/* Browse files button */
[data-testid="stFileUploaderDropzoneInput"] + div button,
[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploader"] button {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
/* Uploaded file item row */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileData"] {
    background: #21262d !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileSize"] {
    color: var(--text-primary) !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input {
    background: var(--bg-card) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}

/* ── Dataframe container ── */
[data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden; }

/* ── Expander / Status widget ──
   st.status() uses BlockProto.Expandable → renders as stExpander / <details> */
[data-testid="stExpander"],
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
/* Expander summary row (the clickable header) */
[data-testid="stExpander"] summary,
details > summary {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}
/* Expander inner content area */
[data-testid="stExpanderDetails"],
details > div {
    background: var(--bg-card) !important;
    padding: 8px 14px 12px !important;
}
/* All text inside expander */
[data-testid="stExpander"] *,
details * {
    color: var(--text-primary) !important;
}
/* Status icon & spinner inside expander header */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary span { color: var(--accent-lt) !important; }

/* ── Alert boxes (success/info/warning/error) ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 3px !important;
}
[data-testid="stAlert"][kind="success"],
div[data-baseweb="notification"][kind="positive"] {
    background: #0d1f12 !important; border-color: var(--success) !important;
    color: var(--success) !important;
}
[data-testid="stAlert"][kind="info"],
div[data-baseweb="notification"][kind="info"] {
    background: #0d1b30 !important; border-color: var(--accent) !important;
    color: var(--accent-lt) !important;
}
[data-testid="stAlert"][kind="warning"],
div[data-baseweb="notification"][kind="warning"] {
    background: #1f1700 !important; border-color: var(--warning) !important;
    color: var(--warning) !important;
}
[data-testid="stAlert"][kind="error"],
div[data-baseweb="notification"][kind="negative"] {
    background: #1f0808 !important; border-color: var(--error) !important;
    color: var(--error) !important;
}
/* Alert text color */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div { color: inherit !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

hr { border-color: var(--border) !important; }

/* ── Pulse animation ── */
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

IC_DB     = '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
IC_CHART  = '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
IC_FILE   = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>'
IC_GRID   = '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>'
IC_WAVE   = '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
IC_UPLOAD = '<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'
IC_KEY    = '<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>'
IC_CHECK  = '<polyline points="20 6 9 17 4 12"/>'
IC_INFO   = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
IC_CPU    = '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'


def section_heading(label: str, icon_d: str, color: str = "#58a6ff"):
    st.markdown(
        f'<div class="section-heading">{_svg(icon_d, 17, color)}&nbsp;{label}</div>',
        unsafe_allow_html=True,
    )


# ─── Tool-call label map ──────────────────────────────────────────────────────
TOOL_LABELS = {
    "load_dataset":           "Loading dataset",
    "get_dataframe_info":     "Inspecting structure",
    "get_statistics":         "Computing statistics",
    "get_missing_values":     "Analysing missing values",
    "handle_missing_values":  "Handling missing values",
    "generate_visualization": "Generating visualisation",
}

# ─── Session State Defaults ───────────────────────────────────────────────────
_defaults = {
    "report":    None,
    "df":        None,
    "viz_paths": [],
    "tool_log":  [],
    "error":     None,
    "running":   False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ──────────────────────────────────────────────────────────────────
def collect_visualizations(output_dir: str) -> list:
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(output_dir, ext)))
    return sorted(paths)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">&#9670;</div>
        <div>
            <div class="sidebar-brand-name">AI Data Scientist</div>
            <div class="sidebar-brand-version">v1.0 &nbsp;&bull;&nbsp; GPT-4o-mini</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload Section
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
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > 10:
            st.error(f"File exceeds 10 MB limit ({size_mb:.1f} MB).", icon=None)
        else:
            file_ok = True
            st.success(f"{uploaded_file.name}  ({size_mb:.2f} MB)", icon=None)

    st.divider()

    run_btn = st.button(
        "Run EDA Analysis",
        disabled=(not file_ok or st.session_state.running),
        use_container_width=True,
    )

    st.divider()

    # How it works
    st.markdown(
        f'<div style="font-size:0.74rem;color:#8b949e;line-height:1.7">'
        f'<b style="color:#c9d1d9;font-size:0.78rem">How it works</b><br>'
        f'1. Upload a CSV dataset (max 10 MB)<br>'
        f'2. Click <b>Run EDA Analysis</b><br>'
        f'3. Review the AI report &amp; charts'
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
            Automated Exploratory Data Analysis &nbsp;&bull;&nbsp; GPT-4o-mini
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


# ─── Run EDA ─────────────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY", "")

if run_btn and file_ok:
    # Save the uploaded file to a temporary path
    tmp_dir = tempfile.mkdtemp(prefix="ai_ds_")
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    # Clear previous outputs directory
    out_dir = os.path.join(os.getcwd(), "outputs")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Reset session results
    st.session_state.report    = None
    st.session_state.viz_paths = []
    st.session_state.tool_log  = []
    st.session_state.error     = None
    st.session_state.running   = True

    try:
        from llm import create_agent
        agent = create_agent(api_key=api_key)

        steps_done: list = []

        with st.status("Running AI Analysis...", expanded=True) as status_box:
            step_area = st.empty()

            def _render_steps():
                badges = []
                for i, tname in enumerate(steps_done):
                    lbl = TOOL_LABELS.get(tname, tname)
                    if i < len(steps_done) - 1:
                        badges.append(
                            f'<div class="step-badge done">'
                            f'{_svg(IC_CHECK,13,"#3fb950")}&nbsp;{lbl}</div>'
                        )
                    else:
                        badges.append(
                            f'<div class="step-badge active">'
                            f'<span class="pulse-dot"></span>&nbsp;{lbl} …</div>'
                        )
                html = f'<div class="step-badges">{"".join(badges)}</div>'
                step_area.markdown(html, unsafe_allow_html=True)

            final_report = ""

            for chunk in agent.stream({
                "messages": [
                    ("human", f"Please perform a full EDA on the dataset at: {tmp_path}")
                ]
            }):
                if "agent" in chunk:
                    for msg in chunk["agent"]["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                steps_done.append(tc["name"])
                                _render_steps()
                        elif hasattr(msg, "content") and msg.content:
                            final_report = msg.content

            # Mark last step as done (all green)
            if steps_done:
                last = steps_done[-1]
                steps_done[-1] = "__done__"
                all_names = [s for s in steps_done if s != "__done__"] + [last]
                badges = [
                    f'<div class="step-badge done">{_svg(IC_CHECK,13,"#3fb950")}&nbsp;{TOOL_LABELS.get(t,t)}</div>'
                    for t in all_names
                ]
                step_area.markdown(
                    f'<div class="step-badges">{"".join(badges)}</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.report    = final_report
            st.session_state.tool_log  = all_names if steps_done else []
            st.session_state.viz_paths = collect_visualizations(out_dir)
            st.session_state.running   = False
            status_box.update(label="Analysis complete", state="complete", expanded=False)

    except Exception as exc:
        st.session_state.error   = str(exc)
        st.session_state.running = False
        st.error(f"Analysis failed: {exc}")
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

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
        <p style="color:#8b949e;max-width:480px;margin:0 auto;font-size:0.92rem;line-height:1.6">
            Upload a CSV file using the sidebar,
            then click <strong style="color:#58a6ff">Run EDA Analysis</strong>
            to generate a full AI-powered data analysis report.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    features = [
        (IC_DB,    "#58a6ff", "Full EDA Pipeline",
         "Automated dataset loading, inspection, cleaning, and statistical analysis."),
        (IC_CHART, "#8b5cf6", "Smart Visualizations",
         "Histograms, boxplots, bar charts, heatmaps, and scatter plots generated intelligently."),
        (IC_FILE,  "#3fb950", "Structured Report",
         "LLM-generated narrative covering findings, outliers, correlations, and next steps."),
    ]
    for col, (ic, clr, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div style="margin-bottom:12px">{_svg(ic, 24, clr)}</div>'
                f'<div style="font-weight:600;font-size:0.93rem;color:#e6edf3;margin-bottom:7px">{title}</div>'
                f'<div style="font-size:0.82rem;color:#8b949e;line-height:1.6">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

else:
    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset Preview",
        "Analysis Report",
        "Visualizations",
        "Statistics",
    ])

    # ── Tab 1 – Dataset Preview ───────────────────────────────────────────────
    with tab1:
        if df is not None:
            rows, cols_ = df.shape
            num_cols = len(df.select_dtypes(include="number").columns)
            cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
            miss     = int(df.isna().sum().sum())
            miss_pct = round(miss / df.size * 100, 1) if df.size > 0 else 0
            mem_kb   = round(df.memory_usage(deep=True).sum() / 1024, 1)

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Rows</div>
                    <div class="metric-value">{rows:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Columns</div>
                    <div class="metric-value">{cols_}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Numeric</div>
                    <div class="metric-value">{num_cols}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Categorical</div>
                    <div class="metric-value">{cat_cols}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Missing Cells</div>
                    <div class="metric-value">{miss:,}</div>
                    <div class="metric-sub">{miss_pct}% of total</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Memory</div>
                    <div class="metric-value">{mem_kb:,}</div>
                    <div class="metric-sub">KB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        else:
            st.info("Upload a dataset to see a preview.", icon=None)

    # ── Tab 2 – Analysis Report ───────────────────────────────────────────────
    with tab2:
        if st.session_state.error:
            st.error(f"Analysis failed: {st.session_state.error}")

        elif st.session_state.report:
            section_heading("AI-Generated EDA Report", IC_FILE)

            # Tool-call trace (collapsible)
            tlog = st.session_state.tool_log
            if tlog:
                with st.expander(f"Tool execution trace ({len(tlog)} calls)", expanded=False):
                    unique_calls = list(dict.fromkeys(tlog))   # preserve order, dedupe
                    for tname in unique_calls:
                        lbl = TOOL_LABELS.get(tname, tname)
                        count = tlog.count(tname)
                        badge = f"{lbl}" + (f" ×{count}" if count > 1 else "")
                        st.markdown(
                            f'<div class="step-badge done">'
                            f'{_svg(IC_CHECK,13,"#3fb950")}&nbsp;{badge}</div>',
                            unsafe_allow_html=True,
                        )

            # Render the markdown report
            st.markdown(st.session_state.report)

        else:
            st.markdown(
                f'<div style="text-align:center;padding:44px;color:#8b949e">'
                f'{_svg(IC_INFO, 26, "#8b949e")}'
                f'<p style="margin-top:10px;font-size:0.92rem">'
                f'Run the analysis to see the AI-generated report here.</p></div>',
                unsafe_allow_html=True,
            )

    # ── Tab 3 – Visualizations ────────────────────────────────────────────────
    with tab3:
        viz_paths = st.session_state.viz_paths
        if viz_paths:
            section_heading(f"{len(viz_paths)} Visualization(s) Generated", IC_CHART)

            # Group by chart-type prefix
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

    # ── Tab 4 – Statistics ────────────────────────────────────────────────────
    with tab4:
        if df is not None:
            num_df = df.select_dtypes(include="number")
            cat_df = df.select_dtypes(include=["object", "category"])

            if not num_df.empty:
                section_heading("Numeric Column Statistics", IC_WAVE)
                desc = num_df.describe().T.round(4)
                desc["skewness"] = num_df.skew().round(4)
                desc["kurtosis"] = num_df.kurt().round(4)
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
                        "Null %":        round(cat_df[c].isna().mean() * 100, 2),
                    })
                st.dataframe(
                    pd.DataFrame(rows_cat),
                    use_container_width=True,
                    hide_index=True,
                )

            if not num_df.empty and len(num_df.columns) > 1:
                section_heading("Correlation Matrix", IC_CHART)
                st.dataframe(num_df.corr().round(4), use_container_width=True)

        else:
            st.info("Upload a dataset to see statistics here.", icon=None)
