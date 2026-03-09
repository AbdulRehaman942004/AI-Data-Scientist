import sys
from llm import create_agent


def run_eda(file_path: str, api_key: str = None) -> str:
    """Run a full EDA on the dataset at file_path and return the final report."""
    agent = create_agent(api_key=api_key)
    result = agent.invoke({
        "messages": [("human", f"Please perform a full EDA on the dataset at: {file_path}")]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset.csv"
    report = run_eda(dataset_path)
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)
