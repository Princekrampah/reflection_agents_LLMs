from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from collections import defaultdict
from typing import List

from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.graph import END, MessageGraph


import json
import os

from decouple import config

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = config("TAVILY_API_KEY")

# tools
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

# tool executor
tool_executor = ToolExecutor([tavily_tool])

# parser
parser = JsonOutputToolsParser(return_id=True)


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)
    ids = []
    tool_invocations = []
    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_results_json",
                    tool_input=query,
                )
            )
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations)
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    return [
        ToolMessage(content=json.dumps(query_outputs), tool_call_id=id_)
        for id_, query_outputs in outputs_map.items()
    ]


################# Initial responder #################


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(
        description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(
        description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


# llm
llm = ChatOpenAI(model="gpt-4-turbo-preview")

initial_answer_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed approximately 300 word answer."
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

validator = PydanticToolsParser(tools=[AnswerQuestion])


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    @traceable
    def respond(self, state: List[BaseMessage]):
        response = []
        for attempt in range(3):
            try:
                response = self.runnable.invoke({"messages": state})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response


first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)


example_question = "How should we handle the climate crisis?"
initial = first_responder.respond([HumanMessage(content=example_question)])

# print(initial)
# print("✦✧✦✧✦✧✧✦✧✦✧ Initial Responder ✦✧✦✧✦✧✧✦✧✦✧")
# parsed = parser.invoke(initial)
# print(parsed)

################# Revisor #################
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )


revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(
    runnable=revision_chain, validator=revision_validator)


# revised = revisor.respond(
#     [
#         HumanMessage(content=""),
#         initial,
#         ToolMessage(
#             tool_call_id=initial.additional_kwargs["tool_calls"][0]["id"],
#             content=json.dumps(
#                 tavily_tool.invoke(str(parsed[0]["args"]["search_queries"]))
#             ),
#         ),
#     ]
# )

# parsed = parser.invoke(revised)

# print("✦✧✦✧✦✧✧✦✧✦✧ First Revised Respond ✦✧✦✧✦✧✧✦✧✦✧")
# print(parsed)

MAX_ITERATIONS = 5

builder = MessageGraph()

builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor.respond)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

def _get_num_iterations(state: List[BaseMessage]):
    i = 0
    for m in state[::-1]:
        if not isinstance(m, (ToolMessage, AIMessage)):
            break
        i += 1
    return i


def should_continue(state: List[BaseMessage]) -> str:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", should_continue)
builder.set_entry_point("draft")

graph = builder.compile()

events = graph.stream(
    [HumanMessage(content=example_question)]
)

for i, step in enumerate(events):
    node, output = next(iter(step.items()))
    print(f"## {i+1}. {node}")
    try:
        print(f'Answer: {parser.invoke(output)[0]["args"]["answer"]}')
        print(
            f'Reflection | Missing: {parser.invoke(output)[0]["args"]["reflection"]["missing"]}')
        print(
            f'Reflection | Missing: {parser.invoke(output)[0]["args"]["reflection"]["superfluous"]}')
        print('Reflection | Search Queries:')
        
        for y, sq in enumerate(parser.invoke(output)[0]["args"]["search_queries"]):
            print(f"{y+1}: {sq}")
        print("✦✧✦✧✦✧✧✦✧✦✧ Node Output ✦✧✦✧✦✧✧✦✧✦✧")
        continue

    except Exception as e:
        print(str(output)[:100] + " ...")

print("\n\n✦✧✦✧✦✧✧✦✧✦✧ Final Generated Response ✦✧✦✧✦✧✧✦✧✦✧\n\n")
print(parser.invoke(step[END][-1])[0]["args"]["answer"])