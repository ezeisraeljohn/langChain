import getpass
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API KEY")

llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

tagging_prompt = ChatPromptTemplate.from_template(
    template="""
Extract the desired information from the following package.
Only Extract the properties mentioned in the "Classification" function.

passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(..., description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale of 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


structured_llm = llm.with_structured_output(Classification)

inp = "Are you okay little wisdom onuu?"
prompt = tagging_prompt.invoke({"input": inp})
# response = structured_llm.invoke(prompt)

for chunk in structured_llm.stream(prompt):
    print(f"{chunk}")

# print(response.model_dump())
