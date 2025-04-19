import os
import sys
import constant

from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = constant.OPENAI_API_KEY

# Load the text file
loader = TextLoader("data.txt")

# Create an index (embeddings + vector search)
index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=OpenAIEmbeddings()
).from_loaders([loader])

query = sys.argv[1]

answer = index.query(query, llm=OpenAI())
print(answer)
