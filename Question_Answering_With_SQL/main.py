from IPython.display import Image, display
from langchain_community.utilities import SQLDatabase
from pprint import pprint
from typing_extensions import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
import os
import getpass

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter you password")

# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db = SQLDatabase.from_uri(
    "postgresql+psycopg2://postgres:*#1357Eze@localhost:5433/trackify"
)
# print(db.dialect)
# print(db.get_usable_table_names())
# pprint(db.run("SELECT * FROM linked_accounts LIMIT 10;"))
print(db.get_table_info())
# pprint(db.run("SELECT * FROM linked_accounts LIMIT 10;"))


# llm = ChatOllama(model="llama3.1", temperature=1)
llm = init_chat_model("gemini-2.0-flash", model_provider="google-genai")


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


system_message = """Given an input question, create a syntactically correct {dialect} query to
run to help find the answer.
You can order the results by a relevant column to return the most interesting examples.

Only use the following tables:
{table_info}

--- Additional Notes ---
- Use only column names you see in the schema.
- Be careful to avoid non-existent columns or tables.
- For PostgreSQL date operations, use:
  - CURRENT_DATE or NOW()
  - Subtract intervals using `CURRENT_DATE - INTERVAL '6 months'`
  - Example: `transaction_date >= CURRENT_DATE - INTERVAL '6 months'`
"""


user_prompt = "Question: {input}"


query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

for message in query_prompt_template.messages:
    message.pretty_print()


class QueryOutput(TypedDict):
    """Generated SQL query"""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": "PostgreSQL",
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


def execute_query(state: State):
    "Execute SQL Query"
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "As a financial assistant, Given the following user question, corresponding SQL query,"
        "and result, answer the user question."
        "Please just give the answers to the human question like someone who do not know anything about sql \n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


# print(write_query({"question": "How many Employees are there?"}))
# print(
#     execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})
# )

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")

memory = MemorySaver()

config = {"configurable": {"thread_id": "abc456"}}

graph = graph_builder.compile(checkpointer=memory)

# print(display(Image(graph.get_graph().draw_mermaid_png())))

result = graph.invoke(
    {"question": "How is my financial life?"},
    config=config,
    # stream_mode="updates",
)
print(result)
