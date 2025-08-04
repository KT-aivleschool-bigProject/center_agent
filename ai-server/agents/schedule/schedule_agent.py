# schedule_agent.py
"""
일정관리 Agent의 메인 실행 파일
이 파일은 Slack 이벤트를 수신하고, LangGraph Agent를 실행하여
일정을 추가/변경/삭제하는 기능을 제공합니다
"""

import slack_sdk
from slack_sdk.errors import SlackApiError
from datetime import datetime
import os
import json

# Slack Bolt 관련 임포트 (Socket Mode 전용)
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent_state import AgentState                   # agent_state.py에서 정의한 상태
from langgraph_agent_definition import langgraph_app # langgraph_agent_definition.py에서 컴파일된 LangGraph 앱 임포트

# 환경 변수 설정
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN') # Slack 봇 토큰 (xoxb- 로 시작)
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN') # Slack 앱 토큰 (xapp- 로 시작, Socket Mode에 필요)
SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET') # Slack 이벤트 서명 검증을 위한 시크릿

# 전역 변수로 봇 사용자 ID 저장
BOT_USER_ID = None

# Slack Bolt 앱 초기화 (Socket Mode 전용)
# 봇 토큰과 서명 시크릿을 사용하여 앱을 구성합니다.
app_bolt = App(token=SLACK_BOT_TOKEN, signing_secret=SIGNING_SECRET)


# 봇 사용자 ID를 가져오는 함수
def get_bot_user_id(client):
    """
    Slack API를 사용하여 봇의 사용자 ID를 가져옵니다.
    """
    try:
        auth_test = client.auth_test()
        return auth_test["user_id"]
    except SlackApiError as e:
        print(f"Error getting bot user ID: {e}")
        return None


# LangGraph Agent를 실행하는 함수
def process_slack_message(message: str, channel: str, user: str, bot_client: slack_sdk.WebClient):
    """
    수신된 Slack 메시지를 기반으로 LangGraph Agent를 실행하는 메인 함수
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] LangGraph Agent 실행 시작: 채널({channel}), 사용자({user})")

    # 1. AgentState 초기화
    initial_state = AgentState(
        slack_message=message,
        channel_id=channel,
        user_id=user,
        bot_client=bot_client,

        # 나머지 필드는 Optional이므로 초기화 시 비워둡니다.
        cleaned_message=None,
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
        target_event_id=None,
        modified_subject=None,
        modified_start_datetime=None,
        modified_end_datetime=None,
        calendar_action_success=None,
        calendar_action_error=None
    )

    # 2. LangGraph 실행
    # .stream()을 사용하면 각 노드의 출력을 실시간으로 확인할 수 있어 디버깅에 유용합니다.
    try:
        # 최종 결과만 필요한 경우 .invoke() 사용
        # final_state = langgraph_app.invoke(initial_state)
        # print("최종 상태:", final_state)

        # 각 스텝을 로깅하며 실행하고 싶을 때 .stream() 사용
        for s in langgraph_app.stream(initial_state, {"recursion_limit": 20}):
            # 각 스텝의 상태를 출력 (디버깅용)
            print("----------------- 스트림 출력 -----------------")
            print(json.dumps(s, indent=2, ensure_ascii=False)) 
            print("----------------------------------------------")

    except Exception as e:
        print(f"LangGraph 실행 중 오류 발생: {e}")
        bot_client.chat_postMessage(channel=channel, text=f"죄송합니다. 요청을 처리하는 중에 예상치 못한 오류가 발생했습니다: {e}")


# 'message' 이벤트 핸들러
# 봇이 참여하고 있는 채널의 모든 메시지를 수신
# 봇 멘션 없는 일반 메시지에서도 특정 키워드를 감지해서 처리하고 싶다면 -> 이 핸들러를 써야 함.
# 예시) DM에서 멘션 없이 항상 반응하도록
@app_bolt.event("message")
def handle_message_events(body, logger, client):
    logger.info(body) # 수신된 메시지 정보 로깅 (디버깅용)
    event = body.get('event', {})
    message_text = event.get('text')
    channel_id = event.get('channel')


# 'app_mention' 이벤트 핸들러
# 봇이 직접 멘션(@봇이름)되었을 때만 트리거됩니다.
@app_bolt.event("app_mention")
def handle_app_mention_events(body, logger, client):
    logger.info(body) # 수신된 멘션 정보 로깅 (디버깅용)
    event = body.get('event', {})
    message_text = event.get('text')
    channel_id = event.get('channel')
    user_id = event.get('user') # 사용자 ID 추가

    # 여기서 우리 봇이 멘션되었는지 명시적으로 확인합니다.
    if message_text and channel_id and f'<@{BOT_USER_ID}>' in message_text:
        # 이전 답변에서 제안한 process_slack_message 함수를 호출합니다.
        process_slack_message(message=message_text, channel=channel_id, user=user_id, bot_client=client)
    else:
        # 우리 봇의 멘션이 없는 메시지는 무시합니다.
        logger.info(f"봇 멘션이 없어 메시지를 건너뜜: {message_text}")


# 메인 실행 블록
# Socket Mode 핸들러를 시작하여 Slack 이벤트 수신을 시작합니다.
if __name__ == "__main__":
    # Slack WebClient 초기화 (봇 ID 획득용)
    # SocketModeHandler가 시작되기 전에 봇 ID를 미리 가져옵니다.
    slack_web_client_for_init = slack_sdk.WebClient(token=SLACK_BOT_TOKEN)
    BOT_USER_ID = get_bot_user_id(slack_web_client_for_init)

    if BOT_USER_ID is None:
        print("Fatal Error: 봇 사용자 ID를 가져올 수 없습니다. 애플리케이션을 종료합니다.")
        exit(1) # 봇 ID 없으면 실행 중단

    print("Slack Agent가 Socket Mode로 시작됩니다.")
    print("슬래시 커맨드(/schedule)는 지원되지 않습니다. Agent를 멘션(@Agent이름)하여 일정을 추가해주세요.")
    
    # Socket Mode 핸들러 시작
    handler = SocketModeHandler(app_bolt, SLACK_APP_TOKEN)
    handler.start()