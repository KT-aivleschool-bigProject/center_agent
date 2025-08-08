# langgraph_agent_definition.py
"""
Langgraph Agent 정의 및 노드 설정
"""

# Langgraph 관련 임포트
from langgraph.graph import StateGraph, END

# 커스텀 모듈 임포트 (AgentState와 Agent 노드 함수들이 필요)
# 이 파일들이 같은 디렉토리에 있다고 가정합니다.
from .agent_nodes import common_message_parser, llm_intent_classifier, llm_calendar_extractor, \
                        add_google_calendar_event, google_calendar_searcher, \
                        llm_calendar_modifier_extractor, google_calendar_updater, \
                        llm_calendar_deleter_extractor, google_calendar_deleter, \
                        route_by_intent, route_after_search
from .agent_state import ScheduleAgentState

# ========= LangGraph Agent 정의 ============== #

# Graph Builder 초기화
workflow = StateGraph(ScheduleAgentState)

# 노드 추가
workflow.add_node("parse_message", common_message_parser)          # 범용 메시지 파서
workflow.add_node("classify_intent", llm_intent_classifier)              # 의도 분류
workflow.add_node("extract_calendar_info", llm_calendar_extractor)       # 일정 추가 정보 추출
workflow.add_node("add_google_calendar_event", add_google_calendar_event)    # 구글 캘린더 일정 추가
workflow.add_node("search_google_calendar", google_calendar_searcher)        # 일정 검색
workflow.add_node("extract_modification_info", llm_calendar_modifier_extractor) # 변경 정보 추출
workflow.add_node("update_google_calendar_event", google_calendar_updater)       # 일정 변경
workflow.add_node("extract_deletion_info", llm_calendar_deleter_extractor)       # 삭제 정보 추출
workflow.add_node("delete_google_calendar_event", google_calendar_deleter)       # 일정 삭제

# 시작점 설정
workflow.set_entry_point("parse_message")

# 엣지 정의
# 1. Slack 메시지 파싱 후 항상 의도 분류 노드로
workflow.add_edge("parse_message", "classify_intent")

# 2. 의도 분류 결과에 따라 분기 (조건부 엣지)
workflow.add_conditional_edges(
    "classify_intent", # classify_intent 노드 이후
    route_by_intent,   # route_by_intent 함수를 호출하여 다음 노드 결정
    {
        "add": "extract_calendar_info", # 의도가 "add"일 때
        "search_calendar": "search_google_calendar", # 의도가 "modify", "delete", "search"일 때
        "__END__": END # 의도가 "unknown"이거나 오류 발생 시 종료
    }
)

# 3. 일정 추가 정보 추출 후 캘린더 추가 노드로
workflow.add_edge("extract_calendar_info", "add_google_calendar_event")

#  4. 일정 추가 후 종료
workflow.add_edge("add_google_calendar_event", END)

# 5. 일정 검색 후 다음 노드 결정 (조건부 엣지)
workflow.add_conditional_edges(
    "search_google_calendar", # search_google_calendar 노드 이후
    route_after_search, # route_after_search 함수를 호출하여 다음 노드 결정
    {
        "modify": "extract_modification_info", # 의도가 "modify"일 때
        "delete": "extract_deletion_info", # 의도가 "delete"일 때
        "__END__": END # 검색 결과가 없거나, "search" 의도일 때 종료
    }
)

# 6. 변경 정보 추출 후 캘린더 변경 노드로
workflow.add_edge("extract_modification_info", "update_google_calendar_event")

# 7. 캘린더 변경 후 종료
workflow.add_edge("update_google_calendar_event", END)

# 8. 삭제 정보 추출 후 캘린더 삭제 노드로
workflow.add_edge("extract_deletion_info", "delete_google_calendar_event")

# 9. 캘린더 삭제 후 종료
workflow.add_edge("delete_google_calendar_event", END)

# 그래프 컴파일 및 외부로 노출
langgraph_app = workflow.compile()

# 이 파일에서 langgraph_app을 임포트하여 사용할 수 있도록 합니다.