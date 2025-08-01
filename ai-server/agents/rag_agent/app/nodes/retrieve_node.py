from app.vector_store import create_vector_store
from app.loader import load_documents, split_documents

from app.store import global_vector_store

def retrieve_node(state):
    question = state.question
    docs = global_vector_store.similarity_search(question, k=3)
    return {**state.dict(), "retrieved_docs": docs}
