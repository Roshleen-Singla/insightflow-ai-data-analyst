"""
Edge case tests for insightflow-ai-data-analyst, built against the real
cleaning_tools, analysis_tools, data_tools, visual_tool, and agent modules.

Run with:  pytest test_edge_cases.py -v -s
(-s so you can see the printed output on ambiguous/agent tests)
"""

import json
import pandas as pd
import pytest

from tools.cleaning_tools import (
    run_cleaning_pipeline,
    clean_tool,
    load_dataset,
    handle_missing_values,
)
from tools.analysis_tools import run_analysis, analysis_tool
from tools.data_tools import get_dataset_info
from tools.visual_tool import create_chart_tool
from langchain_core.messages import HumanMessage
from agents.agent import agent


# Minimal raw columns matching what pipeline expects BEFORE renaming
RAW_COLUMNS = [
    "Order ID", "Date", "Status", "Fulfilment", "Sales Channel ",
    "ship-service-level", "Amount", "ship-postal-code", "ship-city",
    "ship-state", "ship-country", "Qty", "Style", "SKU", "ASIN",
    "Category", "Size", "B2B", "Courier Status", "fulfilled-by",
]

# Minimal cleaned/renamed columns analysis functions expect
CLEANED_COLUMNS = [
    "order_id", "order_date", "status", "fulfilment", "sales_channel",
    "amount", "quantity", "category", "sku", "ship_state", "ship_city", "b2b",
]


# =========================================================
# 1. Empty / near-empty dataset
# =========================================================

def test_cleaning_pipeline_on_header_only_csv(tmp_path):
    """A CSV with headers but zero data rows shouldn't crash the pipeline."""
    input_file = tmp_path / "empty_raw.csv"
    output_file = tmp_path / "empty_cleaned.csv"

    pd.DataFrame(columns=RAW_COLUMNS).to_csv(input_file, index=False)

    df, report = run_cleaning_pipeline(str(input_file), str(output_file))

    assert report["dataset"]["final_rows"] == 0
    assert output_file.exists()


def test_analysis_on_empty_cleaned_dataset(tmp_path):
    """
    Analysis on zero rows: total_sales should be 0, but average_order_value
    involves .mean() on an empty group, which will be NaN. This test documents
    that behavior so you can decide if it needs a guard clause.
    """
    cleaned_file = tmp_path / "cleaned_empty.csv"
    pd.DataFrame(columns=CLEANED_COLUMNS).to_csv(cleaned_file, index=False)

    result = run_analysis(str(cleaned_file))

    assert result["sales"]["total_sales"] == 0
    assert result["sales"]["total_orders"] == 0
    # Documents current behavior — average_order_value is NaN when there are
    # no orders. If you'd rather it return 0 or None, add a guard in
    # analyze_sales() for the empty case.
    avg = result["sales"]["average_order_value"]
    assert pd.isna(avg), (
        f"Expected NaN for average_order_value on empty data, got {avg}. "
        "If you've since added a guard clause, update this assertion."
    )


def test_get_dataset_info_on_single_row(tmp_path):
    """Single-row dataset shouldn't break the summary tool."""
    file = tmp_path / "one_row.csv"
    pd.DataFrame({"amount": [100], "category": ["Shirt"]}).to_csv(file, index=False)

    info = get_dataset_info.invoke({"file_path": str(file)})

    assert "Rows: 1" in info
    assert "No missing values" in info


# =========================================================
# 2. Missing / null values in critical columns
# =========================================================

def test_handle_missing_values_numeric_uses_median():
    df = pd.DataFrame({"amount": [100.0, None, 300.0]})
    cleaned_df, before, after, report = handle_missing_values(df.copy())

    assert after == 0
    assert cleaned_df["amount"].iloc[1] == 200.0  # median of [100, 300]
    assert report["amount"]["method"] == "median"


def test_handle_missing_values_categorical_uses_unknown():
    df = pd.DataFrame({"category": ["Shirt", None, "Pants"]})
    cleaned_df, before, after, report = handle_missing_values(df.copy())

    assert after == 0
    assert cleaned_df["category"].iloc[1] == "Unknown"
    assert report["category"]["method"] == "constant"


def test_cleaning_pipeline_missing_value_counts_match(tmp_path):
    """Full pipeline: missing_before should equal the nulls actually present."""
    input_file = tmp_path / "raw_with_nulls.csv"
    output_file = tmp_path / "cleaned_with_nulls.csv"

    df = pd.DataFrame({col: [None, None] for col in RAW_COLUMNS})
    df["Order ID"] = ["A1", "A2"]
    df["Amount"] = [100.0, None]
    df.to_csv(input_file, index=False)
    
    _, report = run_cleaning_pipeline(str(input_file), str(output_file))

    assert report["missing_values"]["total_after"] == 0
    assert report["missing_values"]["total_before"] > 0


# =========================================================
# 3. Malformed / unexpected file types
# =========================================================

def test_load_dataset_missing_file_raises():
    """A nonexistent path should raise, not silently return something wrong."""
    with pytest.raises(FileNotFoundError):
        load_dataset("data/this_file_does_not_exist.csv")


def test_clean_tool_missing_file_currently_raises():
    """
    Documents CURRENT behavior: clean_tool has no try/except, so a missing
    file bubbles up as a raw FileNotFoundError instead of a friendly message
    the agent could relay. If you add error handling (recommended, to match
    create_chart_tool's pattern), update this test to check for a graceful
    error string in the return value instead.
    """
    with pytest.raises(FileNotFoundError):
        clean_tool.invoke({"input_path": "data/nope.csv"})


def test_analysis_tool_missing_file_currently_raises():
    """Same gap as above, but for analysis_tool."""
    with pytest.raises(FileNotFoundError):
        analysis_tool.invoke({"file_path": "data/nope_cleaned.csv"})


def test_get_dataset_info_malformed_csv(tmp_path):
    """A badly-formed CSV (ragged rows) — check it doesn't crash silently."""
    bad_file = tmp_path / "malformed.csv"
    bad_file.write_text("col_a,col_b\n1,2,3,4\nonly_one_value\n")

    # pandas is often tolerant of ragged rows; this test documents whatever
    # actually happens rather than assuming a crash.
    try:
        result = get_dataset_info.invoke({"file_path": str(bad_file)})
        assert result is not None
    except Exception as e:
        pytest.fail(
            f"get_dataset_info raised on a malformed-but-common CSV shape: {e}"
        )


def test_create_chart_tool_missing_required_args():
    """
    create_chart_tool requires chart_type and metric with no defaults.
    This documents the bug seen in test_visual.py, where only file_path
    was passed — that call is expected to fail with a missing-argument
    error until chart_type/metric are given defaults or made optional.
    """
    with pytest.raises(Exception):
        create_chart_tool.invoke({"file_path": "data/cleaned_amazon_sales.csv"})


def test_create_chart_tool_invalid_metric(tmp_path):
    """An unrecognized metric should return the tool's own error message, not crash."""
    cleaned_file = tmp_path / "cleaned.csv"
    pd.DataFrame({
        "amount": [100, 200],
        "order_date": ["2024-01-01", "2024-01-02"],
        "sku": ["A", "B"],
        "ship_state": ["Punjab", "Delhi"],
        "category": ["Shirt", "Pants"],
    }).to_csv(cleaned_file, index=False)

    result = create_chart_tool.invoke({
        "chart_type": "bar",
        "metric": "not_a_real_metric",
        "file_path": str(cleaned_file),
    })

    assert "Could not create chart" in result


# =========================================================
# 4. Prompt with no matching tool (real agent call)
# =========================================================

def test_agent_does_not_hallucinate_tool_for_unrelated_prompt():
    """
    A prompt with nothing to do with the dataset should not trigger a
    fabricated tool call / fabricated data in the final answer.
    """
    result = agent.invoke({
        "messages": [HumanMessage(content="Tell me a joke.")]
    })

    tool_messages = [
        m for m in result["messages"]
        if type(m).__name__ == "ToolMessage"
    ]
    print(f"\n[unrelated prompt] Tool messages triggered: {len(tool_messages)}")
    assert len(tool_messages) == 0, (
        "Agent invoked a tool for a prompt unrelated to the dataset — "
        "check the system prompt's tool-selection guidance."
    )


# =========================================================
# 5. Ambiguous prompt (real agent call)
# =========================================================

def test_agent_ambiguous_prompt_tool_choice():
    """
    Prompt that could plausibly map to cleaning OR analysis. There's no
    single correct answer here — print which tool(s) got called so you
    can judge if the choice is reasonable given your system prompt.
    """
    result = agent.invoke({
        "messages": [HumanMessage(
            content="Something seems off with this data, can you fix it and tell me what's wrong?"
        )]
    })

    tool_calls_seen = [
        m.name for m in result["messages"]
        if type(m).__name__ == "ToolMessage"
    ]
    print(f"\n[ambiguous prompt] Tools called: {tool_calls_seen}")
    assert len(tool_calls_seen) >= 1