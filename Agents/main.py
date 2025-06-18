import getpass
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter you password")
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# model = init_chat_model("gemini-2.0-flash", model_provider="google-genai")
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

search = TavilySearchResults(max_results=2)
tools = [search]
models_with_tools = llm.bind_tools(tools)

memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="What is the weather in sf")]}, config=config
)
pprint(response)

# for step in agent_executor.stream(
#     {"messages": [HumanMessage(content="What is the weather in sf?")]},
#     stream_mode="messages",
# ):
#     print(step)
