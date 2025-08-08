# schedule_agent.py
"""
일정관리 Agent 클래스
Slack과 Web 챗봇 모두에서 사용할 수 있는 공통 처리 로직을 캡슐화합니다.
"""
from datetime import datetime

# LangGraph 및 상태 관리 관련 임포트
from .agent_state import ScheduleAgentState  # agent_state.py에서 정의한 상태
from .langgraph_agent_definition import langgraph_app # langgraph_agent_definition.py에서 컴파일된 LangGraph 앱 임포트

# Adpater 임포트
from .adapter.base_adapter import ChannelAdapter


class ScheduleAgent:
    """
    Slack 및 Web 채널을 위한 일정 관리 에이전트 클래스.
    하나의 LangGraph 인스턴스를 공유하며 메시지 처리를 담당합니다.
    """

    def __init__(self, channel: str):
        self.name = "Schedule Agent"
        # LangGraph Agent 인스턴스를 클래스 멤버 변수로 저장
        self.agent_executor = langgraph_app

        print(f"✅ ScheduleAgent 구성 요소가 초기화되었습니다. (채널: {channel})")

            
    def process(self, message: str, adapter: ChannelAdapter, user_id: str, channel_id: str):
        """
        사용자 메시지를 받아 LangGraph Agent를 실행합니다.
        
        Args:
            message (str): 사용자 입력 메시지
            adapter (ChannelAdapter): 채널별로 메시지 응답을 처리할 어댑터
            user_id (str): 메시지를 보낸 사용자 ID
            channel_id (str): 메시지가 발생한 채널 ID
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ScheduleAgent 실행 시작 (채널: {channel_id})")

        # 1. Agent 상태 구성
        # 모든 채널에 공통적인 상태를 만듭니다.
        initial_state = ScheduleAgentState(
            raw_message=message,
            cleaned_message=None,  # 초기 상태에서는 None으로 설정
            channel_id=channel_id,
            user_id=user_id,
            adapter=adapter,

            # 나머지 필드는 Optional이므로 초기화시 비워둠
            calendar_subject=None,
            calendar_start_datetime=None,
            calendar_end_datetime=None,
            
            llm_raw_response=None,
            llm_error=None,

            calendar_add_success=None,
            calendar_add_error=None,

            intent=None,
            search_query=None,
            search_date=None,

            found_events=None,
                       
            target_event_idt=None,
            
            modified_subjectt=None,
            modified_start_datetimet=None,
            modified_end_datetimet=None,

            calendar_action_successt=None,
            calendar_action_errort=None
        )

        try:
            # 2. LangGraph Agent 실행
            # run() 대신 stream()을 사용하여 중간 결과를 실시간으로 처리할 수도 있습니다.
            final_state = self.agent_executor.invoke(initial_state)

            # 3. 최종 결과 처리
            # 최종 상태에서 LLM 오류가 있었는지 확인하여 추가적인 로그를 남길 수 있습니다.
            if final_state.get("llm_error"):
                print(f"Agent 실행 중 오류 발생: {final_state['llm_error']}")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ScheduleAgent 실행 완료.")
        except Exception as e:
            print(f"[ERROR] LangGraph Agent 실행 중 예외 발생: {e}")
            adapter.send_message(text=f"일정 처리에 실패했습니다. 잠시 후 다시 시도해주세요. (오류: {e})")


# 이 파일 하단에 Slack Bolt 관련 코드는 별도의 `slack_app.py`로 분리하는 것이 좋습니다.
# 여기서는 Web FastAPI와 결합하여 사용할 것이므로 `main.py`에서 직접 ScheduleAgent 클래스를 사용합니다.