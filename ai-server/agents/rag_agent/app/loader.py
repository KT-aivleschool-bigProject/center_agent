from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def load_documents(data_dir="data/"):
    all_docs = []
    for file in Path(data_dir).glob("*"):
        if file.suffix == ".txt":
            loader = TextLoader(str(file), encoding='utf-8')
        elif file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        else:
            continue
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)
