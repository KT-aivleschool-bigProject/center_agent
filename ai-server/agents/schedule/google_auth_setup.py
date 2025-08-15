# =============== Google Calendar API 인증 코드, 최초 1회 실행 ================== #
import os
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Google Calendar API의 스코프. 일정 생성 및 수정 권한
# 필요한 최소 권한만 부여하는 것이 좋습니다.
# 'calendar.events'는 이벤트 생성/수정/삭제 권한.
# 'calendar'는 캘린더 설정까지 포함하는 전체 권한.
GOOGLE_CALENDAR_SCOPES = [os.getenv('GOOGLE_CALENDAR_SCOPES')]  # google_auth_setup.py에서 사용한 것과 동일한 스코프

# 스크립트 경로 정의
BASE_DIR = os.path.dirname(__file__)
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
CLIENT_SECRET_PATH = os.path.join(BASE_DIR, "client_secret.json")

def authenticate_google_calendar():
    creds = None

    # 토큰 파일 존재 여부 확인
    if os.path.exists(TOKEN_PATH):
        print("기존 token.json 파일을 사용합니다.")
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, GOOGLE_CALENDAR_SCOPES)
        except Exception as e:
            print(f"[ERROR] token.json 로딩 중 오류 발생: {e}")
            return False
    
    else:
        print("token.json 파일이 존재하지 않습니다. OAuth 인증을 시작합니다.")

        try:
            # 파일에 담긴 인증 정보로 구글 서버에 인증
            # 새 창이 열리면서 구글 로그인 및 정보 제공 동의 후 최종 인증 완료
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, GOOGLE_CALENDAR_SCOPES)
            creds = flow.run_local_server(
                port=8080,
                access_type='offline',
                prompt='consent'
            )

            with open(TOKEN_PATH, 'w') as token:
                token.write(creds.to_json())
            print("OAuth 인증 완료 및 token.json 저장 완료.")
            return True 
        except Exception as e:
            print(f"[ERROR] OAuth 인증 중 오류 발생: {e}")
            return False

if __name__ == '__main__':
    print("Google Calendar API 인증 설정 스크립트를 시작합니다.")
    print("이 스크립트는 token.json 파일을 생성하거나 갱신합니다.")
    print("Google Cloud Console에서 다운로드한 'client_secret.json' 파일이 같은 디렉토리에 있어야 합니다.")
    print("="*50)
    print()

    if authenticate_google_calendar():
        print()
        print("="*50)
        print("인증 설정이 완료되었습니다.")
        print("이제 main.py를 실행하여 Google Calendar에 일정을 추가할 수 있습니다.")