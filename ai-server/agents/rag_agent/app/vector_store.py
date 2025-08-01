from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store
