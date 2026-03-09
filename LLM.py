import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.load_user_file import load_dataset
from tools.get_dataframe_info import get_dataframe_info
from tools.get_statistics import get_statistics
from tools.get_missing_values import get_missing_values
from tools.handle_missing_values import handle_missing_values
from tools.generate_visualization import generate_visualization

load_dotenv()

SYSTEM_PROMPT = """You are an expert AI Data Scientist. Your job is to perform a full Exploratory Data Analysis (EDA) on datasets provided by the user.

Follow this exact sequence for every dataset:

1. Call load_dataset to preview the data (shape, columns, first rows).
2. Call get_dataframe_info to understand structure (dtypes, null counts, memory).
3. Call get_statistics to get descriptive stats, skewness, and kurtosis.
4. Call get_missing_values to identify and assess all missing data.
5. Call handle_missing_values with smart strategies per column:
   - Numeric columns: use 'mean' if skew < 1, else use 'median'
   - Categorical columns: use 'mode'
   - Columns with > 50% missing: use 'drop_column'
   - Only include columns that actually have missing values in the strategies JSON
   - If there are no missing values, skip this step
6. Call generate_visualization for key insights:
   - histogram for each numeric column
   - boxplot for each numeric column (outlier detection)
   - bar chart for each categorical column
   - heatmap for correlation (always include this)
   - scatter for pairs of numeric columns that are strongly correlated (|r| > 0.5)
7. Compile a clear, structured final report with:
   - Dataset Overview
   - Key Statistics & Findings
   - Missing Data Summary & How It Was Handled
   - Outliers & Distribution Notes
   - Correlations
   - List of all saved visualizations
   - Recommendations for next steps

Be thorough. Always use the tools — do not guess or fabricate data.
"""

tools = [
    load_dataset,
    get_dataframe_info,
    get_statistics,
    get_missing_values,
    handle_missing_values,
    generate_visualization,
]


def create_agent(api_key: str = None):
    """Create and return the LangGraph react agent with the configured LLM."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=4096,
    )

    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
