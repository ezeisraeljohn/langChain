import ast
import re
from langchain_community.utilities import SQLDatabase
from pprint import pprint
from typing_extensions import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import getpass

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter you password")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# db = SQLDatabase.from_uri(
#     "postgresql+psycopg2://postgres:*#1357Eze@localhost:5433/trackify"
# )
# print(db.dialect)
# print(db.get_usable_table_names())
# pprint(db.run("SELECT * FROM linked_accounts LIMIT 10;"))
# print(db.get_table_info())
# pprint(db.run("SELECT * FROM linked_accounts LIMIT 10;"))


# llm = ChatOllama(model="llama3.1", temperature=1)
llm = init_chat_model("gemini-2.0-flash", model_provider="google-genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

# print(tools)


# class State(TypedDict):
#     question: str
#     query: str
#     result: str
#     answer: str


system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect="SQLite",
    top_k=5,
)

agent_executor = create_react_agent(model=llm, tools=tools, prompt=system_message)

# question = "How is my financial health"

# for step in agent_executor.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

# user_prompt = "Question: {input}"


# query_prompt_template = ChatPromptTemplate(
#     [("system", system_message), ("user", user_prompt)]
# )

# for message in query_prompt_template.messages:
#     message.pretty_print()


# class QueryOutput(TypedDict):
#     """Generated SQL query"""

#     query: Annotated[str, ..., "Syntactically valid SQL query."]


# def write_query(state: State):
#     """Generate SQL query to fetch information."""
#     prompt = query_prompt_template.invoke(
#         {
#             "dialect": "PostgreSQL",
#             "table_info": db.get_table_info(),
#             "input": state["question"],
#         }
#     )
#     structured_llm = llm.with_structured_output(QueryOutput)
#     result = structured_llm.invoke(prompt)
#     return {"query": result["query"]}


# def execute_query(state: State):
#     "Execute SQL Query"
#     execute_query_tool = QuerySQLDataBaseTool(db=db)
#     return {"result": execute_query_tool.invoke(state["query"])}


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")
print(albums[:5])

vector_store.add_texts(artists + albums)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
description = (
    "Use to look up values to filter on. Input is an approximate spelling "
    "of the proper noun, output is valid proper nouns. Use the noun most "
    "similar to the search."
)

retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

suffix = (
    "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
    "the filter value using the 'search_proper_nouns' tool! Do not try to "
    "guess at the proper name - use this function to find similar ones."
)


system = f"{system_message}\n\n{suffix}"
tools.append(retriever_tool)

agent = create_react_agent(llm, tools, prompt=system)

question = "How many albums does alis in chain have?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


# def generate_answer(state: State):
#     """Answer question using retrieved information as context."""
#     prompt = (
#         "As a financial assistant, Given the following user question, corresponding SQL query,"
#         "and result, answer the user question."
#         "Please just give the answers to the human question like someone who do not know anything about sql "
#         "And lastly, if the query returns an error, kindly answer 'Sorry I cannot answer this now' \n\n"
#         f'Question: {state["question"]}\n'
#         f'SQL Query: {state["query"]}\n'
#         f'SQL Result: {state["result"]}'
#     )
#     response = llm.invoke(prompt)
#     return {"answer": response.content}


# # print(write_query({"question": "How many Employees are there?"}))
# # print(
# #     execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})
# # )

# graph_builder = StateGraph(State).add_sequence(
#     [write_query, execute_query, generate_answer]
# )
# graph_builder.add_edge(START, "write_query")

# memory = MemorySaver()

# config = {"configurable": {"thread_id": "1"}}

# graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# # print(display(Image(graph.get_graph().draw_mermaid_png())))

# for step in graph.stream(
#     {"question": "What is my financial health"},
#     config,
#     stream_mode="updates",
# ):
#     print(step)

# try:
#     user_approval = input("Do you want to go to execute query? (yes/no): ")
# except Exception:
#     user_approval = "no"

# if user_approval.lower() == "yes":
#     # If approved, continue the graph execution
#     for step in graph.stream(None, config, stream_mode="updates"):
#         print(step)
# else:
#     print("Operation cancelled by user.")
