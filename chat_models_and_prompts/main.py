import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        "Please enter your LangSmith API key: "
    )

if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        "Please enter your LangSmith project name: "
    )
    if not os.environ["LANGSMITH_PROJECT"]:
        os.environ["LANGSMITH_PROJECT"] = "default"

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Please enter your Google API key: ")


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


# messages = [
#     SystemMessage("Translate the following from English to Italian"),
#     HumanMessage("Hi"),
# ]
# response = model.invoke(messages)
# print("Response:", response.content)

system_template = "Translate the following to {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "Good morning!"})
response = model.invoke(prompt)
print("Response:", response.content)
