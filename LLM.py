import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tools.load_user_file import load_dataset
from tools.get_dataframe_info import get_dataframe_info
from tools.get_statistics import get_statistics
from tools.get_missing_values import get_missing_values
from tools.detect_outliers import detect_outliers
from tools.handle_missing_values import handle_missing_values
from tools.generate_before_after_viz import generate_before_after_plots
from tools.generate_visualization import generate_visualization

load_dotenv()


@tool
def ask_human(
    question: str,
    question_type: str = "single_choice",
    options: list[str] = [],
) -> str:
    """Ask the human user a clarification question during EDA.

    Use this ONLY when truly necessary — do not interrupt for things you can infer autonomously.

    Args:
        question: The clarification question to display to the user.
        question_type: How the user will answer —
            'single_choice' : pick one from the provided options (radio buttons)
            'multi_choice'  : pick several from the options (checkboxes)
            'text'          : free-form text answer (options ignored)
            'yes_no'        : simple yes / no
        options: List of selectable options (required for single_choice and multi_choice).
    """
    pass  # Execution happens via Streamlit HITL state interception


# All automated tools (do NOT include ask_human here)
tools = [
    load_dataset,
    get_dataframe_info,
    get_statistics,
    get_missing_values,
    detect_outliers,
    handle_missing_values,
    generate_before_after_plots,
    generate_visualization,
]


SYSTEM_PROMPT = """You are an expert AI Data Scientist. Perform a comprehensive, intelligent Exploratory Data Analysis (EDA) on CSV datasets provided by the user. Follow the pipeline below exactly. Use tools at every step — never fabricate data.

══════════════════════════════════════════
STEP 1 — INGEST & INSPECT
══════════════════════════════════════════
• Call `load_dataset` → preview shape, columns, sample rows.
• Call `get_dataframe_info` → understand dtypes, null counts, memory.
• Attempt to autonomously identify:
  - The TARGET variable (look for names like target, label, y, outcome, survived, churn, price, sales, etc.)
  - Problem type: classification, regression, or general EDA.

══════════════════════════════════════════
STEP 2 — SMART CLARIFICATION
══════════════════════════════════════════
After inspecting the data, ALWAYS ask the user at least one clarification question before
proceeding. Every dataset benefits from a moment of alignment with the user.

Your question must be GENUINE and HIGH-VALUE — it should meaningfully change what you do
in the analysis. Choose the most important open question from this priority list:

  1. Target variable unclear? → Ask which column is the target (single_choice, list all candidates).
  2. Target obvious but analytical goal ambiguous? → Ask what the user wants to focus on
     (e.g. "Predict dropout?", "Understand feature relationships?", "Find data quality issues?").
  3. Target and goal both clear? → Ask about the business/domain context, or whether
     specific columns should be excluded, or how to handle high-missing columns.

Always pick the question that will have the biggest impact on the quality of your analysis.
Make the question specific to THIS dataset — reference actual column names and findings.
Do not ask generic, vague, or trivially answerable questions.

question_type guide:
  • "single_choice" — fixed set of candidates (e.g. which column is the target)
  • "multi_choice"  — multiple valid picks (e.g. which columns to exclude)
  • "text"          — open-ended context; set options=[]
  • "yes_no"        — binary decision; set options=["Yes", "No"]

══════════════════════════════════════════
STEP 3 — STATISTICAL ANALYSIS
══════════════════════════════════════════
• Call `get_statistics` → descriptive stats, skewness, kurtosis.
• Call `get_missing_values` → identify all missing data patterns.
• Call `detect_outliers` → IQR/Z-score analysis per numeric column.

══════════════════════════════════════════
STEP 4 — DATA CLEANING
══════════════════════════════════════════
Call `handle_missing_values` with smart strategies (JSON string argument):
  • Numeric, |skew| < 1  → "mean"
  • Numeric, |skew| ≥ 1  → "median"
  • Categorical           → "mode"
  • > 50 % missing        → "drop_column"
  • Time-ordered data     → "ffill" or "bfill"
  • Include ONLY columns that actually have missing values.

⚠ IMMEDIATELY after `handle_missing_values`, call `generate_before_after_plots` with:
  - original_file_path = the ORIGINAL file path (from load_dataset / user message)
  - cleaned_file_path  = the cleaned_file_path value returned by handle_missing_values
This creates before/after visualizations showing the cleaning impact.

If there are NO missing values, skip handle_missing_values and generate_before_after_plots entirely.

══════════════════════════════════════════
STEP 5 — AUTONOMOUS VISUALIZATION
══════════════════════════════════════════
You are a senior data scientist. Based on everything you have learned in steps 1–4
(column types, distributions, correlations, outliers, missing values, target variable,
problem type), decide ENTIRELY ON YOUR OWN which charts to generate and why.

Think like this before generating each chart:
  "What question does this chart answer? Is that question worth asking for this dataset?"
  If the answer is no — skip the chart.

Available chart types and when they are genuinely useful:
  • histogram  — understand the distribution shape of a numeric column; especially useful
                 when skewness or bi-modality matters for modelling.
  • boxplot    — expose outliers and spread; most valuable when detect_outliers flagged
                 significant outliers (IQR % > 5) or when comparing across groups.
  • violin     — richer than boxplot; ideal when you want to show both spread and density.
                 Use column2 to group by a categorical when that comparison is insightful.
  • bar        — value frequency of a categorical; useful when class imbalance or dominant
                 categories are analytically relevant.
  • heatmap    — correlation structure across all numeric columns; nearly always useful
                 when there are 3+ numeric columns.
  • scatter    — relationship between two numeric variables; only worth generating when
                 correlation is strong (|r| > 0.45) or when the relationship is non-linear.
  • pairplot   — holistic view of all pairwise relationships; only for small datasets
                 (< 1 000 rows, ≤ 5 numeric columns) where every pair matters.

Guiding principles:
  — Quality over quantity. 4 well-chosen charts beat 12 generic ones.
  — Each chart should answer a different analytical question.
  — Prefer charts that involve the target variable when one exists.
  — Group comparisons (violin/boxplot with column2) are more insightful than lone distributions
    when a meaningful categorical grouping variable is present.
  — Never generate a chart just to fill space or because the column exists.

══════════════════════════════════════════
STEP 6 — FINAL REPORT
══════════════════════════════════════════
Write a well-structured Markdown report with these sections:

## 🗂️ Dataset Overview
Begin this section with a clearly formatted block like this:

> **🎯 Target Variable:** `<column_name>` — <one sentence explaining why this column was chosen as the target, or "None identified — general EDA performed" if no target exists>
> **📋 Problem Type:** <Classification | Regression | General EDA>

Then continue with: shape, column types, memory usage, and a brief description of the dataset.

## 🔍 Data Quality Assessment
Missing values by column, outlier severity, zero-variance / high-cardinality columns.

## 🧹 Cleaning Summary
What was cleaned, why, and the before → after impact (rows, columns, missing cells).
If no cleaning was needed, state that explicitly.

## 📊 Statistical Insights
Key distributions, notable skewness/kurtosis, interesting categorical breakdowns.

## 🔗 Correlations & Relationships
Strongest positive and negative correlations, feature-target relationships,
any multicollinearity concerns.

## 📈 Visual Insights
What each chart reveals — distribution shapes, group differences, trend lines, clusters.
(Do NOT embed image markdown like ![...](path) — charts are displayed separately in the UI.)

## 💡 Recommendations
Suggested feature engineering, recommended modelling approaches, data collection needs,
and important caveats.

Be thorough, analytical, and base ALL conclusions strictly on tool outputs. Never guess or fabricate.
"""


def create_agent(api_key: str = None, memory=None):
    """Create and return the compiled LangGraph EDA agent."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=4096)
    llm_with_tools = llm.bind_tools(tools + [ask_human])

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    automated_tools_node = ToolNode(tools)

    def should_continue(
        state: MessagesState,
    ) -> Literal["automated_tools", "ask_human", "__end__"]:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "__end__"
        for tc in last.tool_calls:
            if tc["name"] == "ask_human":
                return "ask_human"
        return "automated_tools"

    def ask_human_node(state: MessagesState):
        """Placeholder — graph interrupts BEFORE executing this node."""
        pass

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("automated_tools", automated_tools_node)
    workflow.add_node("ask_human", ask_human_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("automated_tools", "agent")
    workflow.add_edge("ask_human", "agent")

    if memory is None:
        memory = MemorySaver()

    return workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])
