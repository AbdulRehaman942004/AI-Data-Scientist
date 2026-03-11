import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tools.load_user_file import load_dataset
from tools.get_dataframe_info import get_dataframe_info
from tools.get_statistics import get_statistics
from tools.get_missing_values import get_missing_values
from tools.handle_missing_values import handle_missing_values
from tools.generate_visualization import generate_visualization

load_dotenv()


@tool
def ask_human(question: str, options: list[str]) -> str:
    """Ask the human user a clarification question.
    Use this to ask about the target column or preferred visualizations.
    Provides a question and a list of options."""
    pass  # Execution happens via Streamlit state interception


tools = [
    load_dataset,
    get_dataframe_info,
    get_statistics,
    get_missing_values,
    handle_missing_values,
    generate_visualization,
]

SYSTEM_PROMPT = """You are an expert AI Data Scientist. Your objective is to perform a comprehensive, intelligent Exploratory Data Analysis (EDA) on datasets provided by the user.

Follow this logical sequence, exercising autonomous data science judgment:

1. INSPECT THE DATA:
   - Call `load_dataset` to preview the data (shape, columns, first rows).
   - Call `get_dataframe_info` to understand structure (dtypes, null counts, memory).
   - Attempt to autonomously identify the likely target variable based on common naming conventions

2. CONDITIONAL CLARIFICATION (HUMAN IN THE LOOP):
   - Assess if the dataset's target variable or analytical objective is obvious.
   - ONLY call the `ask_human` tool IF:
     a) The target column is highly ambiguous, or there are multiple equally plausible targets.
     b) You need specific business context or domain knowledge to proceed meaningfully.
   - If you can confidently infer the target or if a general EDA is sufficient, PROCEED AUTONOMOUSLY without interrupting the user.

3. PERFORM INTELLIGENT ANALYSIS:
   - Call `get_statistics` to evaluate descriptive stats, skewness, and kurtosis.
   - Call `get_missing_values` to identify missing data patterns.
   - Call `handle_missing_values` applying smart, context-aware strategies:
     * Numeric columns: Use 'mean' if skew < 1, else use 'median'.
     * Categorical columns: Use 'mode' (or consider 'fill_unknown' if appropriate).
     * High-missing columns (> 50%): Use 'drop_column'.
     * (Strict rule: Only include columns that actually have missing values in your JSON request).
   - Autonomously evaluate data quality issues: flag severe outliers, zero-variance columns, or extreme high-cardinality categories.

4. ADAPTIVE VISUALIZATION:
   - Call `generate_visualization` to create ONLY plots that provide genuine analytical value based on the data types and findings:
     * Always include a correlation heatmap for numeric features.
     * Generate distribution plots (histogram/bar) for the identified target variable.
     * Generate scatter plots ONLY for pairs of numeric variables that show meaningful correlation (|r| > 0.5).
     * Generate boxplots to highlight relationships between key categorical features and numeric targets (or outliers).
   - Avoid generating redundant or uninformative charts. 

5. SYNTHESIZE FINAL REPORT:
   - Compile a clear, structured markdown report summarizing:
     * Dataset Overview & Data Quality Assessment.
     * Target Variable Identification (state what you chose and why).
     * Missing Data Summary & Strategies Applied.
     * Key Statistical Findings & Outliers.
     * Meaningful Correlations & Visual Insights.
     * Actionable Recommendations for next steps (e.g., Feature Engineering, Modeling approaches).

Be thorough, analytical, and autonomous. Base all conclusions strictly on the outputs of your tools. Do not guess or fabricate data.
"""


def create_agent(api_key: str = None, memory=None):
    """Create and return the custom LangGraph StateGraph agent."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=4096,
    )

    # Bind all tools including the ask_human interrupter
    llm_with_tools = llm.bind_tools(tools + [ask_human])

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Node for running automated tools
    automated_tools_node = ToolNode(tools)

    def should_continue(state: MessagesState) -> Literal["automated_tools", "ask_human", "__end__"]:
        last_message = state["messages"][-1]
        if not getattr(last_message, "tool_calls", None):
            return "__end__"

        # Route to ask_human_node if the LLM needs clarification
        for tc in last_message.tool_calls:
            if tc["name"] == "ask_human":
                return "ask_human"

        return "automated_tools"

    def ask_human_node(state: MessagesState):
        """Placeholder node. The graph interrupts BEFORE executing this."""
        pass

    # Build Graph
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("automated_tools", automated_tools_node)
    workflow.add_node("ask_human", ask_human_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("automated_tools", "agent")
    workflow.add_edge("ask_human", "agent")  # Resumes here after user input

    if memory is None:
        memory = MemorySaver()

    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["ask_human"],
    )
