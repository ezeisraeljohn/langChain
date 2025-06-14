from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

file_path = "~/langChain/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path=file_path)

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
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)
