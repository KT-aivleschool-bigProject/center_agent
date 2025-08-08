
# slack_app.py
# print(f"==== [TEST] slack_app.py의__package__ = {__package__}")
"""
Slack Bolt App을 위한 메인 실행 파일.
Slack 이벤트(멘션, 메시지 등)를 수신하고, ScheduleAgent 클래스를 호출하여
메시지를 처리하는 역할을 담당합니다.
"""

import os
import slack_sdk
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Schedule Agent 클래스 및 어댑터 임포트
from agents.schedule.schedule_agent import ScheduleAgent
from .adapter.slack_adapter import SlackAdapter

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN') # Slack 봇 토큰 (xoxb- 로 시작)
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN') # Slack 앱 토큰 (xapp- 로 시작, Socket Mode에 필요)
SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET') # Slack 이벤트 서명 검증을 위한 시크릿

# Slack Bolt 앱 초기화 (Socket Mode 전용)
app_bolt = App(token=SLACK_BOT_TOKEN, signing_secret=SIGNING_SECRET)

# ScheduleAgent 인스턴스 저장
slack_schedule_agent = ScheduleAgent(channel="slack")  # Slack 채널용 ScheduleAgent 인스턴스 생성

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

# ======================= Slack 이벤트 핸들러 ======================= #
# 'message' 이벤트 핸들러
# 봇이 참여하고 있는 채널의 모든 메시지를 수신
# 봇 멘션 없는 일반 메시지에서도 특정 키워드를 감지해서 처리하고 싶다면 -> 이 핸들러를 써야 함.
# 예시) DM에서 멘션 없이 항상 반응하도록
@app_bolt.event("message")
def handle_message_events(body, logger, client):
    logger.info(body) # 수신된 멘션 정보 로깅 (디버깅용)

    event = body.get('event', {})
    message_text = event.get('text')
    channel_id = event.get('channel')

    logger.info(f"[Slack message 이벤트] {message_text} (channel: {channel_id})")
    # 멘션 아닌 일반 메시지는 현재는 무시 처리

# 'app_mention' 이벤트 핸들러
# 봇이 직접 멘션(@봇이름)되었을 때만 트리거됩니다.
@app_bolt.event("app_mention")
def handle_app_mention_events(body, logger, client):
    logger.info(body) # 수신된 멘션 정보 로깅 (디버깅용)

    event = body.get('event', {})
    message_text = event.get('text')
    channel_id = event.get('channel')
    user_id = event.get('user')

    # 여기서 우리 봇이 멘션되었는지 명시적으로 확인
    if message_text and channel_id and f'<@{BOT_USER_ID}>' in message_text:
        # Slack 어댑터를 생성하여 전달
        adapter = SlackAdapter(client, channel_id)
        
        # ScheduleAgent 클래스의 run 메서드 호출
        slack_schedule_agent.process(
            message=message_text,
            adapter=adapter,
            user_id=user_id,
            channel_id=channel_id
        )
    else:
        logger.info(f"봇 멘션이 없어 메시지를 건너뜀: {message_text}")

# ======================= 슬랙용 메인 처리 로직 ======================= #
def run_slack_bot():
    global BOT_USER_ID # 전역 변수로 봇 사용자 ID 저장

    # Slack Webclient 초기화 (봇 ID 획득용)
    slack_web_client_for_init = slack_sdk.WebClient(token=SLACK_BOT_TOKEN)
    BOT_USER_ID = get_bot_user_id(slack_web_client_for_init)

    if BOT_USER_ID is None:
        print("Fatal Error: 봇 사용자 ID를 가져올 수 없습니다. 애플리케이션을 종료합니다.")
        exit(1)

    print("Slack Agent가 Socket Mode로 시작됩니다.")
    print("슬래시 커맨드(/schedule)는 지원되지 않습니다. Agent를 멘션(@Agent이름)하여 일정을 추가해주세요.")

    # Socket Mode 핸들러 시작
    handler = SocketModeHandler(app_bolt, SLACK_APP_TOKEN)
    handler.connect()

if __name__ == "__main__":
    run_slack_bot()
