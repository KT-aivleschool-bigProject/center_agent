# agent_nodes.py
"""
Langgraph Agent 노드 함수 정의
이 파일은 Langgraph Agent에서 사용하는 각 노드의 비즈니스 로직을 정의합니다.
각 노드는 AgentState를 입력으로 받아 처리 결과를 반환합니다.
"""
import re
import json
from datetime import datetime, timedelta
import os
import subprocess
import sys

# Google 엑세스 토큰 및 인증 관련 임포트
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google Calendar API 설정
GOOGLE_CALENDAR_SCOPES = [os.getenv('GOOGLE_CALENDAR_SCOPES')]  # google_auth_setup.py에서 사용한 것과 동일한 스코프
TARGET_CALENDAR_ID = os.getenv('GOOGLE_CALENDAR_ID')            # 캘린더 이메일 주소 (일정을 추가할 대상 캘린더의 ID)

# 스크립트 경로 정의
BASE_DIR = os.path.dirname(__file__)
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
CLIENT_SECRET_PATH = os.path.join(BASE_DIR, "client_secret.json")
AUTH_SCRIPT = os.path.join(BASE_DIR, "google_auth_setup.py")  # 인증 스크립트 경로

# LangGraph Agent 상태 및 OpenAI 클라이언트 임포트
from agent_state import AgentState
from langchain_openai import ChatOpenAI

# ChatOpenAI 초기화
llm_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature=0.2
)

# 현재 날짜 및 시간 정보 (프롬프트 입력용)
now = datetime.now().astimezone() # 현재 날짜와 시간
current_datetime_str = now.strftime('%Y-%m-%d %H:%M:%S %Z')
this_week_monday = now - timedelta(days=now.weekday())

# 상대적 날짜 해석 규칙 (프롬프트 입력용)
relative_date_guidelines = f"""
    **상대적 날짜 표현을 아래 기준으로 해석해줘:**
    - '오늘'은 '{now.strftime('%Y-%m-%d')}'
    - '내일'은 '{(now + timedelta(days=1)).strftime('%Y-%m-%d')}'
    - '모레'는 '{(now + timedelta(days=2)).strftime('%Y-%m-%d')}'
    - '3일 뒤'는 '{(now + timedelta(days=3)).strftime('%Y-%m-%d')}'
    - '이번주 월요일'은 '{(this_week_monday).strftime('%Y-%m-%d')}'
    - '이번주 금요일'은 '{(this_week_monday + timedelta(days=4)).strftime('%Y-%m-%d')}'
    - '다음주 화요일'은 '{(this_week_monday + timedelta(days=8)).strftime('%Y-%m-%d')}'
    - '다음주 목요일'은 '{(this_week_monday + timedelta(days=10)).strftime('%Y-%m-%d')}'
"""

# 현재 FastAPI가 실행 중인 Python 인터프리터 경로를 그대로 이용
current_python = sys.executable


def run_google_auth_setup():
    """
    google_auth_setup.py를 실행하여 인증 토큰을 생성합니다.
    """
    try:
        subprocess.run([sys.executable, AUTH_SCRIPT], cwd=BASE_DIR, check=True)
        return os.path.exists(TOKEN_PATH)
    except Exception as e:
        print(f"[ERROR] 인증 스크립트 실행 실패: {e}")
        return False


# Google Calendar 서비스 객체 초기화 함수
def get_google_calendar_service():
    """
    Google Calendar API 서비스 객체를 반환합니다.
    토큰이 없거나 만료되었을 경우 자동으로 인증 스크립트를 실행합니다.
    """
    creds = None

    # 1. token.json이 존재하면 로드
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, GOOGLE_CALENDAR_SCOPES)
        except Exception as e:
            print(f"[ERROR] token.json 로딩 중 오류 발생: {e}. 파일이 손상되었을 수 있습니다.")
            creds = None
            return None
    
    # 2. 토큰이 없거나 유효하지 않으면 인증 실행
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # 리프레시 토큰이 유효하면 액세스 토큰 갱신
            print("Google 액세스 토큰이 만료되어 갱신을 시도합니다...")

            try:
                creds.refresh(Request())

                # 갱신된 자격 증명을 token.json 파일에 저장
                with open(TOKEN_PATH, "w") as token:
                    token.write(creds.to_json())
                print("Google 액세스 토큰 갱신 성공.")
            except Exception as e:
                print(f"[ERROR] 토큰 갱신 실패: {e}")

                print("인증 스크립트를 다시 실행합니다...")
                if not run_google_auth_setup():
                    return None
                creds = Credentials.from_authorized_user_file(TOKEN_PATH, GOOGLE_CALENDAR_SCOPES)
        else:
            print("🔐 인증 정보 없음 또는 유효하지 않음 → 인증 시작")
            if not run_google_auth_setup():
                return None
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, GOOGLE_CALENDAR_SCOPES)

    # 3. 서비스 객체 생성
    try:
        service = build("calendar", "v3", credentials=creds)
        return service
    except Exception as e:
        print(f"[ERROR] Google Calendar 서비스 객체 생성 실패: {e}")
        return None
    

# ============== 노드 함수 정의 ================ #
def slack_message_parser(state: AgentState):
    """
    Slack 메시지에서 봇 멘션을 제거하고 정제된 텍스트를 추출하는 노드.
    """
    slack_message = state["slack_message"]
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    # 봇 멘션 부분을 제거하여 실제 일정 내용만 추출
    cleaned_text = re.sub(r'<@\w+>\s*', '', slack_message).strip()

    # 추출된 내용이 비어있을 경우 사용자에게 안내
    if not cleaned_text:
        bot_client.chat_postMessage(channel=channel_id, text=f"일정 추가를 위한 내용을 입력해주세요. 예시: `@Agent이름 내일 10시 팀 회의`")
        return {"cleaned_message": None, "intent": "unknown", "llm_error": "No content after mention removal"} 
    
    # 여기서 /schedule 커맨드 예외 처리. 실제 슬래시 커맨드는 별도 핸들러에서 처리될 수 있지만,
    # 멘션 안에 포함된 경우를 대비하여 처리.
    if cleaned_text.lower().startswith('/schedule'):
        bot_client.chat_postMessage(channel=channel_id, text="`/schedule` 커맨드는 현재 지원되지 않습니다. Agent를 멘션하여 일정을 입력해주세요. 예시: `@Agent이름 내일 10시 팀 회의`")
        return {"cleaned_message": None, "intent": "unknown", "llm_error": "Unsupported command detected"} 
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: slack_message_parser - 정제된 메시지: '{cleaned_text}'")
    return {"cleaned_message": cleaned_text}


def llm_intent_classifier(state: AgentState):
    """
    LLM을 사용하여 사용자의 의도(추가, 변경, 삭제)를 분류하는 노드.
    """
    cleaned_message = state.get("cleaned_message")
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    # 정제된 메시지 내용물 확인
    if cleaned_message is None:
        return {"intent": "unknown", "llm_error": "No cleaned message for intent classification."}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_intent_classifier - 의도 분류 시작")

    try:
        prompt = f"""
            현재 시각은 {current_datetime_str}야.
            다음 메시지의 의도를 'add'(일정 추가), 'modify'(일정 변경), 'delete'(일정 삭제), 'search'(일정 검색), 'unknown'(알 수 없음) 중 하나로 분류해줘.
            메시지: '{cleaned_message}'

            \n\n응답 형식:
            {{"intent": "의도",
              "query": "검색 키워드 (제목)",
              "date": "검색 날짜 (YYYY-MM-DD)"}}\n
            
            예시:
            1. "내일 오후 3시 회의 추가해줘" → {{ "intent": "add", "query": "회의", "date": "2025-08-02" }}
            2. "다음주 목요일 회의 삭제해줘" → {{ "intent": "delete", "query": "회의", "date": "2025-08-08" }}
            3. "이번주 금요일 일정 알려줘" → {{ "intent": "search", "query": "일정", "date": "2025-08-02" }}
        """ + relative_date_guidelines

        # ChatOpenAI LLM 호출
        response = llm_model.invoke(prompt)
        intent_info_json_str = response.content.strip()

        # LLM 응답에서 Markdown 코드 블록 제거 후 JSON 파싱
        if intent_info_json_str.startswith('```json') and intent_info_json_str.endswith('```'):
            intent_info_json_str = intent_info_json_str[len('```json'):-len('```')].strip()
        elif intent_info_json_str.startswith('```') and intent_info_json_str.endswith('```'):
            intent_info_json_str = intent_info_json_str[len('```'):-len('```')].strip()

        print(f"LLM 의도 분류 응답: {intent_info_json_str}")
        intent_info = json.loads(intent_info_json_str)

        intent = intent_info.get('intent', 'unknown')
        query = intent_info.get('query', '')
        date = intent_info.get('date', '')

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_intent_classifier - 의도: {intent}, 쿼리: {query}, 날짜: {date}")
        return {"intent": intent, "search_query": query, "search_date": date}

    except Exception as e:
        print(f"LLM 의도 분류 중 오류 발생: {e}")
        bot_client.chat_postMessage(channel=channel_id, text=f"요청 의도를 파악하는 데 실패했습니다: {e}")
        return {"intent": "unknown", "llm_error": f"Intent classification error: {e}"}

# =============== 일정 추가 관련 노드 ================ #
def llm_calendar_extractor(state: AgentState):
    """
    LLM을 사용하여 '일정 추가' 의도에서 캘린더 정보를 추출하는 노드.
    """
    cleaned_message = state.get("cleaned_message")
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    # 정제된 메시지 내용물 확인
    if cleaned_message is None:
        return {"llm_error": "No cleaned message to process for extraction."}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_calendar_extractor - LLM 호출 시작")

    try:
        prompt = (
            f"다음 텍스트에서 '일정 제목', '시작 날짜 및 시간', '종료 날짜 및 시간'을 JSON 형식으로 추출해줘. "
            f"현재 시각은 '{current_datetime_str}'이야. 이 정보를 기준으로 날짜와 시간을 정확히 추론해줘. "
            f"만약 사용자가 일정의 **종료 기간, 또는 진행 시간(예: '1시간 30분', '2시간 반', '30분')**을 언급했다면, "
            f"그 정보를 사용해서 정확한 종료 시간을 계산해줘. "
            f"**진행 시간 언급이 없는 경우에만** 기본적으로 시작 시간에서 1시간을 더해서 종료 시간을 설정해. "
            f"예시: '다음주 화요일 오후 5시에 회의 1시간 반 예정', '수요일 오전 11시, 30분 미팅' 등도 고려해. "
            f"날짜 형식은 'YYYY-MM-DDTHH:MM:SS'로 맞춰줘. "
            f"추출할 수 없는 경우 빈 문자열로 반환해줘. "
            f"텍스트: '{cleaned_message}'"
            f"\n\n**상대적 날짜 표현 지침:**"
            f"\n- '오늘'은 '{datetime.now().strftime('%Y-%m-%d')}'"
            f"\n- '내일'은 '{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}'"
            f"\n- '모레'는 '{(datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')}'"
            f"\n- '다음주 화요일'처럼 '다음주'가 붙으면 현재 주의 다음 주 요일로 계산"
            f"\n- '이번주 금요일'이면 이번 주의 해당 요일로 계산"

            f"\n\n응답 형식:"
            f'\n{{"subject": "일정 제목", "start_datetime": "YYYY-MM-DDTHH:MM:SS", "end_datetime": "YYYY-MM-DDTHH:MM:SS"}}'
        )


        # ChatOpenAI LLM 호출
        response = llm_model.invoke(prompt)
        date_info_json_str = response.content.strip()

        print(f"LLM 응답: {date_info_json_str}")

        # LLM 응답에서 Markdown 코드 블록 제거 후 JSON 파싱
        if date_info_json_str.startswith('```json') and date_info_json_str.endswith('```'):
            date_info_json_str = date_info_json_str[len('```json'):-len('```')].strip()
        elif date_info_json_str.startswith('```') and date_info_json_str.endswith('```'):
            date_info_json_str = date_info_json_str[len('```'):-len('```')].strip()
        
        print(f"LLM 응답 (정제 후): {date_info_json_str}")

        try:
            date_info = json.loads(date_info_json_str)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}. LLM 응답이 유효한 JSON 형식이 아닐 수 있습니다. 원본 응답: '{response.text}'")
            bot_client.chat_postMessage(channel=channel_id, text="날짜 정보를 파싱하는 데 실패했습니다. 메시지 형식을 확인해주세요.")
            return {"llm_error": f"JSON parsing failed: {e}"}

        # 일정 정보 추출
        subject = date_info.get('subject')
        start_datetime_str = date_info.get('start_datetime')
        end_datetime_str = date_info.get('end_datetime')

        if subject and start_datetime_str and end_datetime_str:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_calendar_extractor - 일정 정보 추출 성공: {subject}")
            return {
                "calendar_subject": subject,
                "calendar_start_datetime": start_datetime_str,
                "calendar_end_datetime": end_datetime_str
            }
        else:
            bot_client.chat_postMessage(channel=channel_id, text="메시지에서 유효한 일정 정보를 찾을 수 없습니다. 예시: `@Agent이름 내일 10시 팀 회의`")
            return {"llm_error": "No valid calendar info extracted."}

    except Exception as e:
        print(f"LLM 처리 중 오류 발생: {e}")
        bot_client.chat_postMessage(channel=channel_id, text=f"일정을 처리하는 중 오류가 발생했습니다: {e}")
        return {"llm_error": f"LLM processing error: {e}"}


def add_google_calendar_event(state: AgentState):
    """
    추출된 캘린더 정보를 사용하여 Google Calendar에 일정을 추가하는 노드.
    """
    subject = state.get("calendar_subject")
    start_datetime_str = state.get("calendar_start_datetime")
    end_datetime_str = state.get("calendar_end_datetime")
    cleaned_message = state.get("cleaned_message") # 내용으로 재사용
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    # 필수 일정 정보 누락 여부 확인 
    if not (subject and start_datetime_str and end_datetime_str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: google_calendar_adder - 필수 일정 정보 누락. 캘린더 추가 건너뜀.")
        return {"calendar_add_success": False, "calendar_add_error": "Missing calendar details."}

    if not TARGET_CALENDAR_ID:
        error_msg = "Error: GOOGLE_CALENDAR_ID 환경 변수가 설정되지 않았습니다. 일정을 추가할 수 없습니다."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_add_success": False, "calendar_add_error": error_msg}

    # Google Calendar 서비스 객체 정의
    service = get_google_calendar_service()

    if not service:
        error_msg = "Google Calendar 인증 실패로 일정을 추가할 수 없습니다. 서버 로그를 확인해주세요."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_add_success": False, "calendar_add_error": "Google Calendar service not available."}

    try:
        event = {
            'summary': subject,
            'description': cleaned_message, # 원본 메시지 또는 정제된 메시지를 설명으로 활용
            'start': {
                'dateTime': start_datetime_str,
                'timeZone': 'Asia/Seoul',
            },
            'end': {
                'dateTime': end_datetime_str,
                'timeZone': 'Asia/Seoul',
            },
        }

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: add_google_calendar_event - Google Calendar에 이벤트 추가 시도")
        print(f"이벤트 데이터: {json.dumps(event, indent=2, ensure_ascii=False)}")

        event_response = service.events().insert(calendarId=TARGET_CALENDAR_ID, body=event).execute()
        
        success_msg = f"✅ '{subject}' 일정이 Google Calendar에 성공적으로 추가되었습니다."
        bot_client.chat_postMessage(channel=channel_id, text=success_msg)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: add_google_calendar_event - 일정 추가 성공!")
        return {"calendar_add_success": True, "calendar_action_success": True}

    except HttpError as http_err:
        error_msg = f"❌ Google Calendar 일정 추가 중 HTTP 오류가 발생했습니다: {http_err}"
        print(error_msg)
        try:
            error_details = json.loads(http_err.content.decode('utf-8'))
            print(f"응답 본문: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"응답 본문 (JSON 아님): {http_err.content.decode('utf-8')}")
        
        bot_client.chat_postMessage(channel=channel_id, text="Google Calendar에 일정을 추가하는 데 실패했습니다. 서버 로그를 확인해주세요.")
        return {"calendar_add_success": False, "calendar_add_error": str(http_err), "calendar_action_success": False}
    except Exception as e:
        error_msg = f"❌ Google Calendar 일정 추가 중 예상치 못한 오류가 발생했습니다: {e}"
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=f"일정을 처리하는 중 오류가 발생했습니다: {e}")
        return {"calendar_add_success": False, "calendar_add_error": str(e), "calendar_action_success": False}


# =============== 일정 검색 관련 노드 ================ #
def google_calendar_searcher(state: AgentState):
    """
    Google Calendar에서 일정을 검색하는 노드. 주로 변경/삭제 전에 사용.
    """
    search_query = state.get("search_query")
    search_date_str = state.get("search_date")
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]
    
    if not TARGET_CALENDAR_ID:
        error_msg = "Error: GOOGLE_CALENDAR_ID 환경 변수가 설정되지 않았습니다. 일정을 검색할 수 없습니다."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"found_events": [], "calendar_action_error": error_msg}

    service = get_google_calendar_service()
    if not service:
        error_msg = "Google Calendar 인증 실패로 일정을 검색할 수 없습니다. 서버 로그를 확인해주세요."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"found_events": [], "calendar_action_error": error_msg}
    
    events_result = []
    try:
        # 현재 시간을 시스템의 로컬 시간대 정보와 함께 가져옵니다.
        # 이 'now' 변수는 app.py 등에서 정의된 'now = datetime.now().astimezone()'와 동일한 방식으로 동작합니다.
        current_local_time = datetime.now().astimezone()
        
        time_min = None
        time_max = None

        if search_date_str:
            try:
                # search_date_str을 파싱하여 해당 날짜의 00:00:00부터 다음 날의 00:00:00까지 설정
                # 시간대 정보는 current_local_time의 시간대 정보를 사용합니다.
                search_date_obj = datetime.strptime(search_date_str, '%Y-%m-%d').replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                # 파싱된 날짜 객체에 현재 시스템의 시간대 정보 적용
                search_date_obj_with_tz = current_local_time.replace(
                    year=search_date_obj.year,
                    month=search_date_obj.month,
                    day=search_date_obj.day,
                    hour=0, minute=0, second=0, microsecond=0
                )
                
                time_min = search_date_obj_with_tz.isoformat()
                time_max = (search_date_obj_with_tz + timedelta(days=1)).isoformat()

                print(f"[{current_local_time.strftime('%H:%M:%S')}] 노드: google_calendar_searcher - 특정 날짜 검색 범위: timeMin={time_min}, timeMax={time_max}")

            except ValueError:
                print(f"[{current_local_time.strftime('%H:%M:%S')}] 노드: google_calendar_searcher - search_date_str 형식 오류: {search_date_str}. 현재 시간부터 검색 시도.")
                # 날짜 형식 오류 시, timeMin을 현재 시간으로 설정하고 timeMax를 비워둠 (향후 일정 검색)
                time_min = current_local_time.isoformat()
                time_max = None
        else:
            # search_date_str이 없는 경우 (예: "회의 찾아줘"와 같이 날짜 언급 없는 경우)
            # 현재 시간부터 이후의 일정을 검색하도록 하되, 넉넉한 기간을 설정하는 것이 좋습니다.
            # 예시: 현재 시간부터 30일 이내의 일정 검색
            time_min = current_local_time.isoformat()
            time_max = (current_local_time + timedelta(days=30)).isoformat()
            print(f"[{current_local_time.strftime('%H:%M:%S')}] 노드: google_calendar_searcher - 전체 기간 검색 범위: timeMin={time_min}, timeMax={time_max}")

        # Calendar API events().list 호출
        events_page = service.events().list(
            calendarId=TARGET_CALENDAR_ID,
            timeMin=time_min, # 수정된 timeMin 사용
            timeMax=time_max, # 수정된 timeMax 사용
            q=search_query, # 검색 키워드
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events_result = events_page.get('items', [])

        if not events_result:
            bot_client.chat_postMessage(channel=channel_id, text="검색된 일정이 없습니다. 검색어 또는 날짜를 확인해주세요.")
            return {"found_events": [], "calendar_action_error": "No events found."}
        
        response_text = "검색된 일정이 있습니다:\n"
        for i, event in enumerate(events_result):
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            try:
                # ISO 8601 형식 문자열을 파싱하여 로컬 시간대로 변환
                start_dt = datetime.fromisoformat(start).astimezone(current_local_time.tzinfo)
                formatted_start = start_dt.strftime('%m월 %d일 %H:%M')
            except ValueError:
                formatted_start = start # 파싱 실패 시 원본 문자열 사용

            try:
                end_dt = datetime.fromisoformat(end).astimezone(current_local_time.tzinfo)
                formatted_end = end_dt.strftime('%H:%M')
            except ValueError:
                formatted_end = end # 파싱 실패 시 원본 문자열 사용

            response_text += f"{i+1}. {event['summary']} (시작: {formatted_start}, 종료: {formatted_end})\n"
        
        # 검색된 일정이 있을 경우, 사용자가 어떤 일정을 변경/삭제할지 선택하도록 안내
        # TODO: 사용자 추가응답 상호작용
        if state["intent"] in ["modify", "delete"]:
            response_text += "어떤 일정을 변경/삭제하시겠습니까? (예: '1번 변경', '2번 삭제')"

        bot_client.chat_postMessage(channel=channel_id, text=response_text)
        
        return {"found_events": events_result}

    except HttpError as http_err:
        error_msg = f"❌ Google Calendar 일정 검색 중 HTTP 오류 발생: {http_err}"
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text="일정 검색 중 오류가 발생했습니다. 서버 로그를 확인해주세요.")
        return {"found_events": [], "calendar_action_error": str(http_err)}
    except Exception as e:
        error_msg = f"❌ Google Calendar 일정 검색 중 예상치 못한 오류 발생: {e}"
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=f"일정 검색 중 오류가 발생했습니다: {e}")
        return {"found_events": [], "calendar_action_error": str(e)}
    

# =============== 일정 변경 관련 노드 ================ #
def llm_calendar_modifier_extractor(state: AgentState):
    """
    LLM을 사용하여 '일정 변경' 의도에서 변경할 일정의 새로운 정보를 추출하는 노드.
    (예: "내일 회의를 11시로 변경해줘" -> 기존 일정 ID와 변경될 시간 정보 추출)
    """
    cleaned_message = state.get("cleaned_message")
    found_events = state.get("found_events", [])
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    if cleaned_message is None or not found_events:
        return {"llm_error": "No cleaned message or events for modification."}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_calendar_modifier_extractor - 변경 정보 추출 시작")
    
    # 사용자에게 제시된 일정 목록을 프롬프트에 포함
    event_list_str = ""
    for i, event in enumerate(found_events):
        event_list_str += f"{i+1}. {event.get('summary', '제목 없음')} (ID: {event.get('id')})\n"

    try:
        prompt = f"""
            현재 시각은 {current_datetime_str}야.
            다음 메시지와 일정 목록을 바탕으로, 사용자가 **어떤 일정을 변경하고자 하는지**와
            **어떻게 변경하고자 하는지**를 JSON 형식으로 추출해줘.

            1. 일정 변경 시 규칙 : 
            - 사용자는 기존 일정 시간과 변경할 시간을 모두 언급할 수 있어.
            - 이때 검색 대상 일정은 '기존 시간' 기준으로 판단하고, 새로 설정할 시간은 변경된 일정 시간이야.
            - 사용자가 진행 시간 또는 종료 시간을 명시하지 않았다면, 기존 일정의 진행시간을 유지해줘.
            - 사용자가 '1시간 반으로 바꿔줘', '30분만 할래', '2시간 예정; 등으로 **진행 시간 변경을 요청한 경우에만**, 새 시작 시간 기준으로 종료 시간을 계산해서 바꿔줘.
            - 만약 메시지에 ** 시작과 종료 시간이 모두 정확하게 포함되어 있다면**, 두 시간 모두 반영해줘.
            
            2. 날짜 형식 및 출력 관련 규칙 : 
            날짜 형식을 'YYYY-MM-DDTHH:MM:SS'로 맞춰줘.
            일정 번호와 변경될 정보가 없으면 빈 문자열로 반환해줘. \n

            {relative_date_guidelines}

            메시지: '{cleaned_message}'\n

            일정 목록:\n{event_list_str}
            
            \n\n응답 형식: 
            {{
                "event_index": "변경할 일정의 0부터 시작하는 인덱스", 
                "subject": "새 일정 제목", 
                "start_datetime": "YYYY-MM-DDTHH:MM:SS", 
                "end_datetime": "YYYY-MM-DDTHH:MM:SS"
            }}
        """

        # ChatOpenAI LLM 호출
        response = llm_model.invoke(prompt)
        modify_info_json_str = response.content.strip()

        # LLM 응답에서 Markdown 코드 블록 제거 후 JSON 파싱
        if modify_info_json_str.startswith('```json') and modify_info_json_str.endswith('```'):
            modify_info_json_str = modify_info_json_str[len('```json'):-len('```')].strip()
        elif modify_info_json_str.startswith('```') and modify_info_json_str.endswith('```'):
            modify_info_json_str = modify_info_json_str[len('```'):-len('```')].strip()

        print(f"LLM 변경 정보 추출 응답: {modify_info_json_str}")

        modify_info = json.loads(modify_info_json_str)
        event_index = modify_info.get('event_index')
        new_subject = modify_info.get('subject')
        new_start = modify_info.get('start_datetime')
        new_end = modify_info.get('end_datetime')

        if event_index is not None and 0 <= event_index < len(found_events):
            target_event = found_events[event_index]
            target_event_id = target_event.get('id')
            
            # 변경할 정보가 없으면 원래 정보를 사용 (제목은 변경될 가능성 높음)
            new_subject = new_subject if new_subject else target_event.get('summary')
            new_start = new_start if new_start else target_event.get('start', {}).get('dateTime')
            new_end = new_end if new_end else target_event.get('end', {}).get('dateTime')

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_calendar_modifier_extractor - 변경할 일정 ID: {target_event_id}")
            return {
                "target_event_id": target_event_id,
                "modified_subject": new_subject,
                "modified_start_datetime": new_start,
                "modified_end_datetime": new_end
            }
        else:
            bot_client.chat_postMessage(channel=channel_id, text="어떤 일정을 변경할지 또는 변경할 내용을 명확히 알려주세요. (예: '1번 일정을 오후 4시로 변경')")
            return {"llm_error": "Invalid event index or missing modification details."}

    except Exception as e:
        print(f"LLM 변경 정보 추출 중 오류 발생: {e}")
        bot_client.chat_postMessage(channel=channel_id, text=f"일정 변경 정보를 파악하는 데 실패했습니다: {e}")
        return {"llm_error": f"LLM modification extraction error: {e}"}


def google_calendar_updater(state: AgentState):
    """
    Google Calendar의 일정을 변경하는 노드.
    """
    target_event_id = state.get("target_event_id")
    modified_subject = state.get("modified_subject")
    modified_start_datetime = state.get("modified_start_datetime")
    modified_end_datetime = state.get("modified_end_datetime")
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    if not (target_event_id and modified_subject and modified_start_datetime and modified_end_datetime):
        error_msg = "일정 변경에 필요한 정보가 부족합니다."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_action_success": False, "calendar_action_error": error_msg}
    
    if not TARGET_CALENDAR_ID:
        error_msg = "Error: GOOGLE_CALENDAR_ID 환경 변수가 설정되지 않았습니다. 일정을 변경할 수 없습니다."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_action_success": False, "calendar_action_error": error_msg}

    service = get_google_calendar_service()
    if not service:
        error_msg = "Google Calendar 인증 실패로 일정을 변경할 수 없습니다. 서버 로그를 확인해주세요."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_action_success": False, "calendar_action_error": error_msg}

    try:
        updated_event = {
            'summary': modified_subject,
            'start': {
                'dateTime': modified_start_datetime,
                'timeZone': 'Asia/Seoul',
            },
            'end': {
                'dateTime': modified_end_datetime,
                'timeZone': 'Asia/Seoul',
            },
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: google_calendar_updater - 일정 변경 시도: {target_event_id}")
        print(f"변경될 이벤트 데이터: {json.dumps(updated_event, indent=2, ensure_ascii=False)}")

        event_response = service.events().update(
            calendarId=TARGET_CALENDAR_ID,
            eventId=target_event_id,
            body=updated_event
        ).execute()

        success_msg = f"✅ '{modified_subject}' 일정이 성공적으로 변경되었습니다.\n"
        bot_client.chat_postMessage(channel=channel_id, text=success_msg)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: google_calendar_updater - 일정 변경 성공!")
        return {"calendar_action_success": True}

    except HttpError as http_err:
        error_msg = f"❌ Google Calendar 일정 변경 중 HTTP 오류 발생: {http_err}"
        print(error_msg)
        try:
            error_details = json.loads(http_err.content.decode('utf-8'))
            print(f"응답 본문: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"응답 본문 (JSON 아님): {http_err.content.decode('utf-8')}")
        
        bot_client.chat_postMessage(channel=channel_id, text="일정 변경 중 오류가 발생했습니다. 서버 로그를 확인해주세요.")
        return {"calendar_action_success": False, "calendar_action_error": str(http_err)}
    except Exception as e:
        error_msg = f"❌ Google Calendar 일정 변경 중 예상치 못한 오류 발생: {e}"
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=f"일정 변경 중 오류가 발생했습니다: {e}")
        return {"calendar_action_success": False, "calendar_action_error": str(e)}
    
# =============== 일정 삭제 관련 노드 ================ #
def llm_calendar_deleter_extractor(state: AgentState):
    """
    LLM을 사용하여 '일정 삭제' 의도에서 삭제할 일정의 번호(인덱스)를 추출하는 노드.
    """
    cleaned_message = state.get("cleaned_message")
    found_events = state.get("found_events", [])
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    if cleaned_message is None or not found_events:
        return {"llm_error": "No cleaned message or events for deletion."}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_calendar_deleter_extractor - 삭제 정보 추출 시작")
    
    event_list_str = ""
    for i, event in enumerate(found_events):
        event_list_str += f"{i+1}. {event.get('summary', '제목 없음')} (ID: {event.get('id')})\n"

    try:
        prompt = (
            f"다음 메시지와 제공된 일정 목록을 바탕으로, 삭제하고자 하는 일정의 번호(0부터 시작하는 인덱스)를 JSON 형식으로 추출해줘. "
            f"일정 목록:\n{event_list_str}"
            f"메시지: '{cleaned_message}'"
            f"\n\n응답 형식: "
            f'{{"event_index": "삭제할 일정의 0부터 시작하는 인덱스"}}'
        )
        
        # ChatOpenAI LLM 호출
        response = llm_model.invoke(prompt)
        delete_info_json_str = response.content.strip()

        # LLM 응답에서 Markdown 코드 블록 제거 후 JSON 파싱
        if delete_info_json_str.startswith('```json') and delete_info_json_str.endswith('```'):
            delete_info_json_str = delete_info_json_str[len('```json'):-len('```')].strip()
        elif delete_info_json_str.startswith('```') and delete_info_json_str.endswith('```'):
            delete_info_json_str = delete_info_json_str[len('```'):-len('```')].strip()

        print(f"LLM 삭제 정보 추출 응답: {delete_info_json_str}")

        delete_info = json.loads(delete_info_json_str)
        event_index = delete_info.get('event_index')

        # event_index 타입 변환 (문쟈열 ->정수형)
        try:
            event_index = int(event_index)
        except (ValueError, TypeError):
            bot_client.chat_postMessage(channel=channel_id, text="일정 번호가 잘못 전달되었습니다. 다시 말해 주세요.")
            return {"llm_error": "Invalid event_index type (not an int)."}

        # 인덱스 유효성 검사
        if event_index is not None and 0 <= event_index < len(found_events):
            target_event_id = found_events[event_index].get('id')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: llm_calendar_deleter_extractor - 삭제할 일정 ID: {target_event_id}")
            return {"target_event_id": target_event_id}
        else:
            bot_client.chat_postMessage(channel=channel_id, text="선택한 일정 번호가 유효하지 않습니다. 다시 확인해 주세요.")
            return {"llm_error": "Invalid event index for deletion."}

    except Exception as e:
        print(f"LLM 삭제 정보 추출 중 오류 발생: {e}")
        bot_client.chat_postMessage(channel=channel_id, text=f"일정 삭제 정보를 파악하는 데 실패했습니다: {e}")
        return {"llm_error": f"LLM deletion extraction error: {e}"}

def google_calendar_deleter(state: AgentState):
    """
    Google Calendar의 일정을 삭제하는 노드.
    """
    target_event_id = state.get("target_event_id")
    channel_id = state["channel_id"]
    bot_client = state["bot_client"]

    if not target_event_id:
        error_msg = "일정 삭제에 필요한 정보(이벤트 ID)가 부족합니다."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_action_success": False, "calendar_action_error": error_msg}

    if not TARGET_CALENDAR_ID:
        error_msg = "Error: GOOGLE_CALENDAR_ID 환경 변수가 설정되지 않았습니다. 일정을 삭제할 수 없습니다."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_action_success": False, "calendar_action_error": error_msg}

    service = get_google_calendar_service()
    if not service:
        error_msg = "Google Calendar 인증 실패로 일정을 삭제할 수 없습니다. 서버 로그를 확인해주세요."
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=error_msg)
        return {"calendar_action_success": False, "calendar_action_error": error_msg}

    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: google_calendar_deleter - 일정 삭제 시도: {target_event_id}")
        
        service.events().delete(
            calendarId=TARGET_CALENDAR_ID,
            eventId=target_event_id
        ).execute()

        success_msg = f"✅ 일정이 성공적으로 삭제되었습니다."
        bot_client.chat_postMessage(channel=channel_id, text=success_msg)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 노드: google_calendar_deleter - 일정 삭제 성공!")
        return {"calendar_action_success": True}

    except HttpError as http_err:
        error_msg = f"❌ Google Calendar 일정 삭제 중 HTTP 오류 발생: {http_err}"
        print(error_msg)
        try:
            error_details = json.loads(http_err.content.decode('utf-8'))
            print(f"응답 본문: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"응답 본문 (JSON 아님): {http_err.content.decode('utf-8')}")
        
        bot_client.chat_postMessage(channel=channel_id, text="일정 삭제 중 오류가 발생했습니다. 서버 로그를 확인해주세요.")
        return {"calendar_action_success": False, "calendar_action_error": str(http_err)}
    except Exception as e:
        error_msg = f"❌ Google Calendar 일정 삭제 중 예상치 못한 오류 발생: {e}"
        print(error_msg)
        bot_client.chat_postMessage(channel=channel_id, text=f"일정 삭제 중 오류가 발생했습니다: {e}")
        return {"calendar_action_success": False, "calendar_action_error": str(e)}


# ============ 조건부 엣지를 위한 라우터 함수 ============ #
def route_by_intent(state: AgentState):
    """
    LLM이 분류한 의도(intent)에 따라 다음 노드를 결정하는 라우터.
    """
    intent = state.get("intent")
    llm_error = state.get("llm_error") # LLM에서 오류가 났는지 확인

    if llm_error: # LLM 처리 중 오류가 발생하면 종료
        return "__END__" # Langgraph의 END 노드로 직접 이동하는 특별한 키워드

    if intent == "add":
        return "add"
    elif intent in ["modify", "delete", "search"]:
        # 변경/삭제/검색 의도는 먼저 일정 검색이 필요
        return "search_calendar"
    else: # "unknown" 이거나 다른 의도일 경우
        return "__END__" # 적절한 응답 후 종료
    
def route_after_search(state: AgentState):
    """
    일정 검색 후 다음 노드를 결정하는 라우터.
    검색된 일정이 없거나, 검색 후 사용자가 어떤 작업을 할지 명확하지 않을 때.
    """
    intent = state.get("intent")
    found_events = state.get("found_events")
    calendar_action_error = state.get("calendar_action_error") # 검색 중 오류가 있었는지

    if calendar_action_error: # 검색 중 오류가 있으면 종료
        return "__END__"

    if not found_events: # 검색된 일정이 없으면 종료
        return "__END__"

    if intent == "modify":
        return "modify"
    elif intent == "delete":
        return "delete"
    elif intent == "search": # 검색만 원하는 경우, 이미 검색 결과를 Slack으로 보냈으므로 종료
        return "__END__"
    else:
        # 이 경우는 발생하지 않아야 하지만, 안전장치
        return "__END__"