from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
import getpass
from pprint import pprint
from langchain_core.runnables import chain
from typing import List
from langchain import hub

load_dotenv()

file_path = "~/langChain/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path=file_path)

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API Key")

# documents = [
#     Document(
#         page_content="Animals are good friends.",
#         metadata={"source": "mammals-pet.docx"},
#     ),
#     Document(
#         page_content="Cats are independent pets that often enjoy their space",
#         metadata={"source": "mammals-pet.docx"},
#     ),
# ]

docs = loader.load()

# print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_split = text_splitters.split_documents(docs)

# print(len(all_split))

docs = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# vector_1 = embeddings.embed_query(all_split[0].page_content)
# vector_2 = embeddings.embed_query(all_split[0].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length{len(vector_1)}\n")
# print(vector_1[:10])

vector_store = InMemoryVectorStore(embedding=embeddings)

ids = vector_store.add_documents(documents=all_split)
# results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")

doc, score = results[1]
print(f"score {score}\n")
print(doc)
print(len(results))
