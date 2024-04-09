from langgraph.graph import MessageGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from decouple import config
from langchain_openai import ChatOpenAI

from typing import List


model = ChatOpenAI(api_key=config("OPENAI_API_KEY"))


def call_model(state: MessageGraph):
    # print(state)
    messages = state["messages"]
    response = model.invoke(messages)

    return state.append(response)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return "reflect"


graph = MessageGraph()

graph.add_node("generate", model)
graph.add_node("reflect", model)
graph.set_entry_point("generate")


graph.add_conditional_edges("generate", should_continue)
graph.add_edge("reflect", "generate")

app = graph.compile()

inputs = HumanMessage(
    content="Write me a research paper on climate change.")
response = app.invoke(inputs)

print(response)