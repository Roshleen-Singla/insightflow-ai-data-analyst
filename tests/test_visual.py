from tools.visual_tool import create_chart_tool

FILE_PATH = (
    r"C:/Users/Acer/Desktop/data analyst me/"
    r"ai-agent-project/insightflow-ai-data-analyst/"
    r"data/cleaned_amazon_sales.csv"
)


result = create_chart_tool.invoke({
    "file_path": FILE_PATH
})

print("\n==============================")
print("VISUALIZATION TOOL RESULT")
print("==============================")
print(result)