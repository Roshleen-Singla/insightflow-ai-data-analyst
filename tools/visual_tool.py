import plotly.express as px
from langchain_core.tools import tool
import pandas as pd
from tools.analysis_tools import load_cleaned_data


# Simple in-memory store so the UI can retrieve the actual figure
# object, separately from the text summary sent to the LLM.
LATEST_CHART = {"figure": None, "title": None}


def _build_figure(df, chart_type: str, metric: str):

    if metric == "sales_trend":
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        data = df.groupby(df["order_date"].dt.to_period("M"))["amount"].sum()
        data.index = data.index.astype(str)
        fig = px.line(data, x=data.index, y=data.values,
                      title="Sales Trend Over Time",
                      labels={"x": "Month", "y": "Sales (₹)"})

    elif metric == "top_products":
        data = df.groupby("sku")["amount"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(data, x=data.index, y=data.values,
                     title="Top 10 Products by Revenue",
                     labels={"x": "SKU", "y": "Sales (₹)"})

    elif metric == "sales_by_state":
        data = df.groupby("ship_state")["amount"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(data, x=data.index, y=data.values,
                     title="Top 10 States by Sales",
                     labels={"x": "State", "y": "Sales (₹)"})

    elif metric == "sales_by_category":
        data = df.groupby("category")["amount"].sum().sort_values(ascending=False)
        fig = px.pie(data, names=data.index, values=data.values,
                     title="Sales by Category")

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return fig 


@tool
def create_chart_tool(
    chart_type: str,
    metric: str,
    file_path: str = "data/cleaned_amazon_sales.csv"
) -> str:
    """
    Creates a chart from the cleaned Amazon sales dataset.

    metric: one of "sales_trend" (line chart), "top_products" (bar chart),
    "sales_by_state" (bar chart), "sales_by_category" (pie chart).
    Each metric produces its own fixed, appropriate chart type.

    chart_type is currently informational only — the chart shape is
    determined automatically by the metric chosen.

    Use this when the user asks to see, plot, chart, or visualize
    sales data, trends, top products, or state/category breakdowns.
    """
    try:
        df = load_cleaned_data(file_path)
        fig = _build_figure(df, chart_type, metric)
        LATEST_CHART["figure"] = fig
        LATEST_CHART["title"] = fig.layout.title.text
        return f"Chart created: {fig.layout.title.text}. It will be displayed to the user."
    except Exception as e:
        return f"Could not create chart: {e}. Valid metrics are: sales_trend, top_products, sales_by_state, sales_by_category."
