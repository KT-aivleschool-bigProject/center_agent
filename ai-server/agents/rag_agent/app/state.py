from typing import Optional, List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    question: Optional[str] = None
    retrieved_docs: Optional[List[Document]] = None
    answer: Optional[str] = None
