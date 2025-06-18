import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter you password")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    trim_messages,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict, Sequence


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

trimmer = trim_messages(
    max_tokens=30,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
# trimmer.invoke(messages)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# Define a new graph
workflow = StateGraph(state_schema=State)


# Define the function that calls the model
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc456"}}
# query = "What is my name"

# language = "Spanish"
# input_messages = messages + [HumanMessage(query)]

# output = app.invoke({"messages": input_messages, "language": language}, config)

# output["messages"][-1].pretty_print()


query = "What is my name"
language = "English"
input_messages = messages + [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):
        print(chunk.content, end="|")
# output = app.invoke({"messages": input_messages, "language": language}, config)
# output["messages"][-1].pretty_print()
