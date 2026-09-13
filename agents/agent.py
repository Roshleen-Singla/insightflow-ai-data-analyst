'''from agents.model import model
from tools.data_tools import get_dataset_info


tools = [get_dataset_info]

model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = """
You are InsightFlow, an AI data analyst.

You are working with the Amazon Sales dataset.

The dataset is located at:
data\Amazon Sale Report.csv

When you need information about the dataset, use the
get_dataset_info tool with exactly this file path.
"""
'''

from agents.model import model
from langchain.agents import create_agent
from tools.data_tools import get_dataset_info
from tools.cleaning_tools import clean_tool
from tools.analysis_tools import analysis_tool
from tools.visual_tool import create_chart_tool


tools = [
    get_dataset_info,
    clean_tool,
    analysis_tool,
    create_chart_tool
]

SYSTEM_PROMPT = """
You are InsightFlow, an AI data analyst.

You are working with the Amazon Sales dataset.

The dataset is located at:

data/Amazon Sale Report.csv

Available tools:

1. get_dataset_info
   Use this when you need information about the
   dataset structure, columns, data types, or
   missing values.

2. clean_tool
   Use this when the user asks to clean the dataset,
   including:
   - handling missing values
   - removing useless columns
   - standardizing city names
   - standardizing state names

3. analysis_tool
   Use this when the user asks for analysis of the dataset,
   including:
   - total sales by product
   - sales trends over time
   - top performing products
   - top cities and states for sales
   - location insights based on sales data
   - b2b insights based on sales data

   4. create_chart_tool
   Use this when the user asks to see, plot, chart, or visualize
   sales data, trends, top products, or state/category breakdowns.

   
The cleaned dataset is saved as: data/cleaned_amazon_sales.csv

When you need information about the analysis data, use the analysis_tool with exactly this file path: data/cleaned_amazon_sales.csv
this tool will provide insights on the analysis of the dataset, including total sales by product, sales trends over time, top performing products, top cities and states for sales, location insights based on sales data, and b2b insights based on sales data.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


