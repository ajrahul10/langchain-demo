import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Set your OpenAI API key
load_dotenv()

# Access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Load vector index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)


# Set up LLM and QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

query = sys.argv[1]

# Ask and answer
response = qa_chain.invoke(query)
print(response["result"])
