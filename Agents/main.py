import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter you password")
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

model = init_chat_model("gemini-2.0-flash", model_provider="google-genai")

search = TavilySearchResults(max_results=2)
tools = [search]
models_with_tools = model.bind_tools(tools)


agent_executor = create_react_agent(model, tools)

# response = agent_executor.invoke(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]}
# )

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="What is the weather in sf?")]}
):
    step["messages"][-1].pretty_print()
