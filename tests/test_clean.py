from tools.cleaning_tools import run_cleaning_pipeline


INPUT_FILE = (
    r"C:/Users/Acer/Desktop/data analyst me/"
    r"ai-agent-project/insightflow-ai-data-analyst/"
    r"data/Amazon Sale Report.csv"
)

OUTPUT_FILE = (
    r"C:/Users/Acer/Desktop/data analyst me/"
    r"ai-agent-project/insightflow-ai-data-analyst/"
    r"data/cleaned_amazon_sales.csv"
)


df, report = run_cleaning_pipeline(
    INPUT_FILE,
    OUTPUT_FILE
)


print("\n==============================")
print("CLEANING REPORT")
print("==============================")

for key, value in report.items():

    print(f"\n{key}:")

    if isinstance(value, dict):

        for sub_key, sub_value in value.items():

            print(f"  {sub_key}: {sub_value}")

    else:

        print(f"  {value}")


print("\n==============================")
print("CLEANING COMPLETE")
print("==============================")