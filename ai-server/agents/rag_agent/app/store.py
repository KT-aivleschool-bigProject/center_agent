from app.loader import load_documents, split_documents
from app.vector_store import create_vector_store

documents = load_documents("data/")
split_docs = split_documents(documents)
global_vector_store = create_vector_store(split_docs)

def update_vector_store():
    global global_vector_store
    docs = load_documents("data/")
    split_docs = split_documents(docs)
    global_vector_store = create_vector_store(split_docs)
