from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from decouple import config
from langchain_openai import ChatOpenAI
from typing import List, Sequence
from langgraph.graph import MessageGraph, END
import asyncio


from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


prompt = ChatPromptTemplate.from_messages(
    [
        #  this needs to be a tuple so do not forget the , at the end and do not include any , in-between
        (
            "system",
            "You are an AI assistant researcher tasked with researching on a variety of topics in a short summary of 5 paragraphs."
            " Generate the best research possible as per user request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(api_key=config("OPENAI_API_KEY"))

generate = prompt | llm


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        #  this needs to be a tuple so do not forget the , at the end
        (
            "system",
            "You are a senior researcher"
            " Provide detailed recommendations, including requests for length, depth, style, etc."
            " to an asistant researcher to help improve his/her researches",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect = reflection_prompt | llm


async def generation_node(state: Sequence[BaseMessage]):
    return await generate.ainvoke({"messages": state})


async def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    class_map = {"ai": HumanMessage, "human": AIMessage}

    translated = [messages[0]] + [
        class_map[msg.type](content=msg.content) for msg in messages[1:]
    ]

    response = await reflect.ainvoke({"messages": translated})

    return HumanMessage(content=response.content)


graph = MessageGraph()

graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.set_entry_point("generate")


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return "reflect"


graph.add_conditional_edges("generate", should_continue)
graph.add_edge("reflect", "generate")

app = graph.compile()


async def stream_reponses():
    async for event in app.astream(
        [
            HumanMessage(
            content="Research on climate change"
            )
        ]
    ):
        print(event)
        print("================================")
        print("================================")


asyncio.run(stream_reponses())
