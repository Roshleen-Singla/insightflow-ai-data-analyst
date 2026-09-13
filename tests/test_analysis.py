from tools.analysis_tools import analysis_tool


FILE_PATH = (
    r"C:/Users/Acer/Desktop/data analyst me/"
    r"ai-agent-project/insightflow-ai-data-analyst/"
    r"data/cleaned_amazon_sales.csv"
)


result = analysis_tool.invoke({
    "file_path": FILE_PATH
})

print("\n==============================")
print("ANALYSIS TOOL RESULT")
print("==============================")
print(result)