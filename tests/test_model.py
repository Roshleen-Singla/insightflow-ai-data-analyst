from agents.model import model

response = model.invoke(
    "You are a data analyst. Explain what revenue means in one sentence."
)

print(response.content)