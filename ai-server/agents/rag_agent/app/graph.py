from langgraph.graph import StateGraph
from app.state import RAGState

# 노드 불러오기
from app.nodes.input_node import input_node
from app.nodes.retrieve_node import retrieve_node
from app.nodes.generate_node import generate_node

def build_graph():
    builder = StateGraph(RAGState)

    # 노드 등록
    builder.add_node("Input", input_node)
    builder.add_node("Retrieve", retrieve_node)
    builder.add_node("Generate", generate_node)

    # 흐름 정의
    builder.set_entry_point("Input")
    builder.add_edge("Input", "Retrieve")
    builder.add_edge("Retrieve", "Generate")
    builder.set_finish_point("Generate")

    # 컴파일된 graph 반환
    return builder.compile()
