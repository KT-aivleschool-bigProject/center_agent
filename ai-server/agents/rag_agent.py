from rag_agent.app.loader import load_documents, split_documents
from rag_agent.app.vector_store import create_vector_store
from rag_agent.app.store import global_vector_store  # 벡터스토어는 전역 사용 가능
from rag_agent.app.graph import build_graph  # 그래프 빌더 import

class RAGAgent:
    def __init__(self):
        self.name = "RAG Document QA Agent"
        self.graph = build_graph()  # LangGraph로 구성된 DAG

    async def process(self, message: str) -> str:
        try:
            result = self.graph.invoke({"question": message})
            return result.get("answer", "답변을 생성하지 못했습니다.")
        except Exception as e:
            return f"[오류] RAG Agent 처리 중 문제 발생: {str(e)}"