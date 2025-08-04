# agent_state.py
"""
Langgraph Agent의 상태를 정의하는 파일
이 파일은 Langgraph Agent에서 사용하는 상태를 정의합니다.
각 노드에서 이 상태를 읽고 업데이트할 수 있습니다.
"""

from typing import TypedDict, Optional
import slack_sdk # Slack 클라이언트 객체를 타입 힌트로 사용하기 위함

class AgentState(TypedDict):
    """
    Langgraph Agent의 현재 상태를 나타내는 TypedDict.
    Agent의 각 노드에서 이 상태를 읽고 업데이트할 수 있습니다.
    """
    slack_message: str              # 원본 Slack 메시지 텍스트
    cleaned_message: Optional[str]  # 봇 멘션 등이 제거된 정제된 메시지
    channel_id: str                 # 메시지가 발생한 Slack 채널 ID
    user_id: str                    # 메시지를 보낸 사용자 ID
    bot_client: slack_sdk.WebClient # Slack WebClient 인스턴스 (메시지 전송 등에 사용)

    calendar_subject: Optional[str]         # LLM이 추출한 일정 제목
    calendar_start_datetime: Optional[str]  # LLM이 추출한 일정 시작 시간 (ISO 8601)
    calendar_end_datetime: Optional[str]    # LLM이 추출한 일정 종료 시간 (ISO 8601)
    
    llm_raw_response: Optional[str]     # LLM의 원본 응답
    llm_error: Optional[str]            # LLM 처리 중 발생한 오류 메시지

    calendar_add_success: Optional[bool]    # 캘린더 추가 성공 여부
    calendar_add_error: Optional[str]       # 캘린더 추가 중 발생한 오류 메시지

    intent: Optional[str]           # 사용자의 의도: "add", "modify", "delete", "unknown"
    search_query: Optional[str]     # 일정 검색을 위한 키워드 (변경/삭제 시)
    search_date: Optional[str]      # 일정 검색을 위한 날짜 (YYYY-MM-DD)

    # 검색된 일정 목록 (Google Calendar Event 객체 리스트)
    # 각 Event는 'id', 'summary', 'start', 'end' 등의 필드를 가짐
    found_events: Optional[list[dict]] 
    
    target_event_id: Optional[str] # 변경/삭제 대상 일정의 고유 ID (사용자가 선택)

    # 변경될 일정의 새로운 정보 (변경 시)
    modified_subject: Optional[str]
    modified_start_datetime: Optional[str]
    modified_end_datetime: Optional[str]

    calendar_action_success: Optional[bool] # 캘린더 변경/삭제 성공 여부
    calendar_action_error: Optional[str]    # 캘린더 변경/삭제 중 발생한 오류 메시지