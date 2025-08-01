from app.loader import load_documents, split_documents
from app.vector_store import create_vector_store

from dotenv import load_dotenv
load_dotenv()

# 문서 불러오기 및 분할
print("1")
docs = load_documents("data/")
split_docs = split_documents(docs)
print("2")
# 벡터스토어 생성
vector_store = create_vector_store(split_docs)
print("3")
# 유사 문서 검색 테스트
query = "이 프로젝트의 핵심 기능은 무엇인가요?"
docs = vector_store.similarity_search(query, k=2)
print("4")
for d in docs:
    print(d.page_content)
