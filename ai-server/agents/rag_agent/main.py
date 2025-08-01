from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.graph import build_graph
from fastapi import UploadFile, File
from app.loader import load_documents, split_documents
from app.vector_store import create_vector_store
from app.store import global_vector_store, update_vector_store
import os

class QuestionRequest(BaseModel):
    question: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_graph = build_graph()


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if not hasattr(global_vector_store, "index") or global_vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="아직 업로드된 문서가 없습니다.")
    result = rag_graph.invoke({"question": req.question})
    return {
        "question": req.question,
        "answer": result["answer"]
    }

# global_vector_store = create_vector_store(split_documents(load_documents("data/")))

# upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = f"data/{file.filename}"

    if os.path.exists(save_path):
        raise HTTPException(status_code=400, detail="동일한 이름의 파일이 이미 존재합니다.")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    update_vector_store()

    return {"message": f"{file.filename} 업로드 및 벡터 인덱싱 완료"}

# list docs
@app.get("/list_docs")
async def list_docs():
    doc_ids = list(global_vector_store.docstore._dict.keys())
    return {"documents": doc_ids}


# clear docs
@app.post("/clear_docs")
async def clear_docs():
    global_vector_store.index.reset()
    global_vector_store.docstore._dict.clear()
    global_vector_store._collection.clear()
    for filename in os.listdir("data/"):
        file_path = os.path.join("data/", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return {"message": "벡터스토어 및 업로드 문서 전체 초기화 완료"}