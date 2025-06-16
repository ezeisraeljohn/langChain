from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import getpass
from typing import List

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter GOOGLE API KEY")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


class Person(BaseModel):
    """Information about a person"""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    name: Optional[str] = Field(
        default=None, description="The actual name of the person"
    )
    hair_color: Optional[str] = Field(
        default=None, description="The name of the person"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height Measured in meters"
    )


class Data(BaseModel):
    """Extracted data about people"""

    people: List[Person]


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

structured_llm = llm.with_structured_output(schema=Data)

text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me and height."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)
print(response)
