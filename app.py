import streamlit as st
from langchain_core.messages import HumanMessage

from agents.agent import agent
from tools.visual_tool import LATEST_CHART
from utils.message_utils import get_text


st.set_page_config(page_title="InsightFlow", page_icon="📊")
st.title("InsightFlow — AI Data Analyst")

# Persist chat history across reruns (Streamlit reruns the whole script on every interaction)
if "history" not in st.session_state:
    st.session_state.history = []

# Render past messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart") is not None:
            st.plotly_chart(msg["chart"], key=msg.get("chart_key"))

# Chat input box
user_input = st.chat_input("Ask about your sales data...")

if user_input:

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input, "chart": None})

    # Reset chart slot before this turn, so we only pick up a NEW chart if one was made
    LATEST_CHART["figure"] = None

    with st.spinner("Thinking..."):
        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

    final_message = result["messages"][-1]
    answer_text = get_text(final_message)

    chart = LATEST_CHART["figure"]

    with st.chat_message("assistant"):
        st.markdown(answer_text)
        if chart is not None:
            st.plotly_chart(chart, key=f"chart_{len(st.session_state.history)}")

    st.session_state.history.append({
        "role": "assistant",
        "content": answer_text,
        "chart": chart,
        "chart_key": f"chart_{len(st.session_state.history)}"
    })