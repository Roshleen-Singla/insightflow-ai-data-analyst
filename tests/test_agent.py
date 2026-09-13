"""from agents.agent import model_with_tools, SYSTEM_PROMPT
from tools.data_tools import get_dataset_info


messages = [
    ("system", SYSTEM_PROMPT),
    ("human", "Tell me what information you can get about my dataset.")
]

response = model_with_tools.invoke(messages)

print("MODEL RESPONSE:")
print(response)

if response.tool_calls:
    tool_call = response.tool_calls[0]

    print("\nTOOL SELECTED:")
    print(tool_call["name"])

    print("\nTOOL ARGUMENTS:")
    print(tool_call["args"])

    tool_result = get_dataset_info.invoke(tool_call["args"])

    print("\nTOOL RESULT:")
    print(tool_result)
"""

"""from agents.agent import model_with_tools, SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from tools.data_tools import get_dataset_info
from tools.cleaning_tools import clean_tool
from tools.analysis_tools import analysis_tool
from tools.visual_tool import create_chart_tool


TOOL_REGISTRY = {
    "get_dataset_info": get_dataset_info,
    "clean_tool": clean_tool,
    "analysis_tool": analysis_tool,
    "create_chart_tool": create_chart_tool
}

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="  Plot sales trends over time.")
]

response = model_with_tools.invoke(messages)

print("\n==============================")
print("MODEL RESPONSE (initial)")
print("==============================")
print(response)

if response.tool_calls:

    # IMPORTANT: keep the model's own tool-call message in the conversation
    messages.append(response)

    for tool_call in response.tool_calls:

        print("\nTOOL SELECTED:", tool_call["name"])
        print("TOOL ARGUMENTS:", tool_call["args"])

        tool_fn = TOOL_REGISTRY.get(tool_call["name"])

        if tool_fn is None:
            print(f"\nWARNING: no tool registered for '{tool_call['name']}'")
            continue

        tool_result = tool_fn.invoke(tool_call["args"])

        # Feed the result back as a ToolMessage, tagged with the matching call id
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    # SECOND call — now the model actually answers your question using the tool results
    final_response = model_with_tools.invoke(messages)

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print("RAW CONTENT TYPE:", type(final_response.content))
    print("RAW CONTENT:", repr(final_response.content))

else:
    print("\nNo tool calls were made.")
    print(response.content)"""


from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from agents.agent import agent
from agents.model import model
from tools.data_tools import get_dataset_info
from tools.cleaning_tools import clean_tool
from tools.analysis_tools import analysis_tool
from tools.visual_tool import create_chart_tool
from utils.message_utils import get_text


tools = [get_dataset_info, clean_tool, analysis_tool, create_chart_tool]

SYSTEM_PROMPT = """
You are InsightFlow, an AI data analyst for the Amazon Sales dataset.
Use your tools to inspect, clean, analyze, or visualize the data.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


result = agent.invoke({
    "messages": [HumanMessage(content="Show a chart of sales trends over time.")]
})

final_message = result["messages"][-1]

print("\n==============================")
print("FINAL ANSWER")
print("==============================")
print(get_text(final_message))