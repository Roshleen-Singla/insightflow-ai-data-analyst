# InsightFlow — AI Data Analyst Agent

InsightFlow is an agentic AI data analyst built on top of the Amazon Sales dataset. Instead of writing SQL, pandas code, or chart scripts by hand, you ask InsightFlow a question in plain English — and it decides on its own which tool to call, runs it, and answers you.

> "What's my total revenue?" → it calls the analysis tool and tells you.
> "Show me a chart of sales by state" → it calls the visualization tool and renders it.
> "Clean this dataset" → it runs the full cleaning pipeline and reports what changed.

## Demo

[Add your demo video / GIF here]

## How it works

InsightFlow follows the **ReAct (Reason + Act)** pattern: the model reasons about what the user needs, takes an action (calls a tool), observes the result, and either responds or takes another action.

![Architecture diagram](flowchart_understanding.drawio (1))

1. **User** sends a prompt (e.g. *"What's the total revenue?"*)
2. **Model** decides which tool the task needs, and why
3. **Tool** runs and returns a real result back to the model (the "observation")
4. **Model** turns that result into a plain-language answer

One design detail worth calling out: the loop above works cleanly for anything the model can read as text. Charts are the exception — an LLM can't return a chart object directly, so `create_chart_tool` stores the actual figure in a shared in-memory store and only sends the model a short text confirmation. The Streamlit UI reads the real chart directly from that store, bypassing the model entirely for the visual itself.

## Tools

| Tool | What it does |
|---|---|
| `get_dataset_info` | Returns a readable summary of the dataset — shape, columns, dtypes, missing values, duplicate rows |
| `clean_tool` | Runs the full cleaning pipeline: removes useless columns, handles missing values (median for numeric, "Unknown" for categorical), standardizes city/state names, renames columns, saves the cleaned file |
| `analysis_tool` | Runs sales, product, location, and order analysis on the cleaned dataset and returns structured JSON |
| `create_chart_tool` | Builds a Plotly chart (line, bar, or pie depending on the metric) — sales trend, top products, sales by state, or sales by category |

## Tech stack

- **Model**: Gemini (via `langchain-google-genai`)
- **Agent framework**: LangChain (`create_agent`)
- **UI**: Streamlit
- **Data**: pandas
- **Visualization**: Plotly
- **Testing**: pytest

## Project structure

```
insightflow-ai-data-analyst/
├── agents/
│   ├── agent.py          # Agent definition, system prompt, tool registry
│   └── model.py          # Gemini model configuration
├── tools/
│   ├── data_tools.py      # get_dataset_info
│   ├── cleaning_tools.py  # clean_tool, run_cleaning_pipeline
│   ├── analysis_tools.py  # analysis_tool, run_analysis
│   └── visual_tool.py     # create_chart_tool
├── utils/
│   └── message_utils.py
├── tests/
│   ├── test_tools.py
│   ├── test_clean.py
│   ├── test_analysis.py
│   ├── test_visual.py
│   ├── test_agent.py
│   ├── test_model.py
│   └── test_edge_cases.py
├── data/
│   ├── Amazon Sale Report.csv
│   └── cleaned_amazon_sales.csv
├── app.py                 # Streamlit chat interface
└── requirements.txt
```

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/<your-username>/insightflow-ai-data-analyst.git
   cd insightflow-ai-data-analyst
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Add your Gemini API key to a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Running tests

```bash
python -m pytest -v
```

Test coverage includes tool-level tests, agent tool-selection tests, and a dedicated edge case suite (`test_edge_cases.py`) covering empty datasets, missing/null values, prompts with no matching tool, ambiguous prompts, and malformed/missing files.

**Bug caught by the edge case tests:** the cleaning pipeline's missing-value handler computed a column's median to fill nulls — but for a column that was entirely empty, the median itself is `NaN`, so `fillna(NaN)` silently did nothing. The pipeline reported "missing values handled" while 36 cells stayed null in testing. Fixed by falling back to a constant value when the median isn't a real number.

## Known limitations / roadmap

- No caching — repeated identical questions re-run the full tool call each time
- Visualization is limited to four fixed metrics (`sales_trend`, `top_products`, `sales_by_state`, `sales_by_category`)
- Currently scoped to a single dataset (Amazon Sales); not yet generalized to arbitrary uploaded CSVs

