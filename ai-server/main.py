# main.py
"""
팀 에이전트 시스템의 AI 서버 메인 파일
이 파일은 FastAPI를 사용하여 AI 서버를 실행합니다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional
from threading import Thread
import asyncio
import os
from dotenv import load_dotenv
import openai
import json

# 필요한 에이전트 클래스 임포트
from agents import CodeAgent
from agents.rag_agent import RAGAgent
from agents.schedule.schedule_agent import ScheduleAgent
from agents.schedule.adapter.web_adapter import WebAdapter
from agents.schedule import slack_app

# 환경변수 로드
load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print(
        "⚠️  OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    )

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN') # Slack 봇 토큰 (xoxb- 로 시작)
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN') # Slack 앱 토큰 (xapp- 로 시작, Socket Mode에 필요)
SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET') # Slack 이벤트 서명 검증을 위한 시크릿

@asynccontextmanager
async def lifespan_slack_service(app: FastAPI):
    """FastAPI 서버 시작 시 Slack Agent를 백그라운드 스레드에서 실행"""   
    # 현재 파일 경로 기준으로 ai-server 디렉토리로 이동
    # ai_server_path = os.path.dirname(__file__)

    try:
        # Thread로 slack_app 실행
        if all([SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SIGNING_SECRET]):
            slack_thread = Thread(target=slack_app.run_slack_bot, daemon=True)
            slack_thread.start()
            print("✅ Slack Agent를 백그라운드 스레드에서 실행했습니다.")
        else:
            print("ℹ️ Slack 토큰 미설정: Slack Agent는 시작하지 않습니다.")
    except Exception as e:
        print(f"❌ Slack Agent 실행 중 오류 발생: {e}")

    yield   # FastAPI 서버가 실행되는 동안 이 부분이 유지됩니다.

app = FastAPI(
    title="팀 에이전트 시스템 AI 서버",
    description="멀티 에이전트 기반 AI 처리 서버",
    version="1.0.0",
    lifespan=lifespan_slack_service,  # Slack Agent를 백그라운드에서 실행
)

# CORS 설정 (React 앱과 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8002", "http://127.0.0.1:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
)


# Pydantic 모델들
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    channel_type: Optional[str] = "web" # 기본값은 'web'으로 설정
    channel_id: Optional[str] = "web"


class AgentResponse(BaseModel):
    message: str
    agent_type: str
    timestamp: str


class ChatResponse(BaseModel):
    response: str
    agents_used: List[str]
    processing_time: float


# 에이전트 클래스들
class ManagerAgent:
    def __init__(self):
        self.name = "Manager Agent"
        self.system_prompt = """당신은 팀 에이전트 시스템의 중앙 관리자입니다. 
사용자의 요청을 분석하고 적절한 전문 에이전트를 선택해야 합니다.

사용 가능한 에이전트:
1. Code Agent: 코드 리뷰, PR 요약 및 리뷰, 코드 품질 개선, github에서 관련 기능 코드 찾기
2. Document Agent: 문서 작성, 편집, 검색, API 문서 생성
3. Schedule Agent: 프로젝트 일정 관리, 마일스톤 추적, 팀원 작업량 분배
4. RAG Agent: 문서 검색 및 지식 기반 질문 답변

응답 형식:
{
    "selected_agent": "code|document|schedule|rag|general",
    "reason": "선택 이유",
    "confidence": 0.0-1.0
}"""

    async def analyze_prompt(self, message: str) -> dict:
        """사용자 프롬프트를 분석하고 적절한 에이전트를 선택"""
        if not openai.api_key:
            # API 키가 없을 때는 키워드 기반 분석
            return self._fallback_analysis(message)

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"다음 요청을 분석해주세요: {message}"},
                ],
                max_tokens=200,
                temperature=0.3,
            )

            result = response.choices[0].message.content
            try:
                return json.loads(result)
            except:
                return self._fallback_analysis(message)

        except Exception as e:
            print(f"Manager Agent 오류: {e}")
            return self._fallback_analysis(message)

    def _fallback_analysis(self, message: str) -> dict:
        """API 키가 없을 때 사용하는 키워드 기반 분석"""
        message_lower = message.lower()

        if any(
            word in message_lower
            for word in ["코드", "버그", "리뷰", "개발", "git", "repository"]
        ):
            return {
                "selected_agent": "code",
                "reason": "코드 관련 요청 감지",
                "confidence": 0.8,
            }
        elif any(
            word in message_lower
            for word in ["검색", "찾아", "알려", "질문", "답변", "문서에서", "자료에서"]
        ):
            return {
                "selected_agent": "rag",
                "reason": "문서 검색/질문 답변 요청 감지",
                "confidence": 0.8,
            }
        elif any(
            word in message_lower
            for word in ["문서", "작성", "편집", "api", "readme", "docs"]
        ):
            return {
                "selected_agent": "document",
                "reason": "문서 관련 요청 감지",
                "confidence": 0.8,
            }
        elif any(
            word in message_lower
            for word in ["일정", "스케줄", "마일스톤", "데드라인", "프로젝트"]
        ):
            return {
                "selected_agent": "schedule",
                "reason": "일정 관련 요청 감지",
                "confidence": 0.8,
            }
        else:
            return {
                "selected_agent": "general",
                "reason": "일반적인 대화",
                "confidence": 0.5,
            }


class DocumentAgent:
    def __init__(self):
        self.name = "Document Agent"
        self.system_prompt = """당신은 전문적인 문서 에이전트입니다. 
문서 작성, 편집, 검색, API 문서 생성 등을 담당합니다.

다음과 같은 작업을 수행할 수 있습니다:
- 기술 문서 작성 및 편집
- API 문서 생성 및 관리
- README 파일 작성 가이드
- 문서 검색 및 요약
- 마크다운 형식 문서 작성
- 사용자 매뉴얼 작성

항상 명확하고 구조화된 문서를 작성하도록 도와주세요."""

    async def process(self, message: str) -> str:
        """문서 관련 요청 처리"""
        if not openai.api_key:
            return f"문서 에이전트가 처리 중입니다: {message}\n\n문서 작성, 편집, 검색, API 문서 생성 등의 작업을 수행할 수 있습니다."

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ],
                max_tokens=800,
                temperature=0.3,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Document Agent 오류: {e}")
            return f"문서 에이전트 처리 중 오류가 발생했습니다: {str(e)}"


# 에이전트 인스턴스 생성
manager_agent = ManagerAgent()
code_agent = CodeAgent()
document_agent = DocumentAgent()
web_schedule_agent = ScheduleAgent(channel="web")  # 웹 채널용 ScheduleAgent
rag_agent = RAGAgent()


# ================ FastAPI 엔드포인트 설정 ================ #
@app.get("/", tags=["Root"])
async def root():
    """
    서버의 루트 엔드포인트입니다.
    """
    return {"message": "팀 에이전트 시스템 AI 서버가 실행 중입니다."}


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    서버의 상태를 확인합니다.
    """
    return {
        "status": "healthy",
        "agents": ["manager", "code", "document", "schedule", "rag"],
        "openai_configured": bool(openai.api_key),
    }


@app.post("/ai/process",
    response_model=ChatResponse,
    summary = "사용자 메시지를 처리하고 적절한 에이전트를 호출합니다.",
    description="""
    관리자 에이전트가 사용자 메시지를 분석하여 가장 적합한 전문 에이전트에게
    처리를 위임합니다.
    
    **요청 본문**:
    - `message`: 사용자의 채팅 메시지
    - `user_id`: 사용자 ID (선택사항)
    """,
    tags=["AI Agents"],
)
async def process_chat(chat_message: ChatMessage):
    """사용자 메시지를 처리하고 적절한 에이전트를 호출"""
    import time

    start_time = time.time()

    try:
        # 1. 관리자 에이전트가 프롬프트 분석
        analysis = await manager_agent.analyze_prompt(chat_message.message)

        # 2. 적절한 에이전트 호출
        agents_used = ["manager"]
        response = ""

        selected_agent = analysis.get("selected_agent", "general")

        if selected_agent == "code":
            response = await code_agent.process(chat_message.message)
            agents_used.append("code")
        elif selected_agent == "document":
            response = await document_agent.process(chat_message.message)
            agents_used.append("document")
        elif selected_agent == "schedule":
            # ScheduleAgent가 선택되면, 웹 어댑터를 생성하여 실행
            web_adapter = WebAdapter()
            user_id = chat_message.user_id if chat_message.user_id else "web_user"  # TODO: 실제 환경에서는 실제 사용자 ID를 사용해야 함
            channel_id = chat_message.channel_id if chat_message.channel_id else "web"
            
            # ScheduleAgent의 process 메서드 호출
            web_schedule_agent.process(
                message=chat_message.message,
                adapter=web_adapter, # WebAdapter 인스턴스 전달
                user_id=user_id,
                channel_id=channel_id
            )
            response = web_adapter.get_response() # WebAdapter에 저장된 응답 가져오기
            agents_used.append("schedule")
        elif selected_agent == "rag":
            response = await rag_agent.process(chat_message.message)
            agents_used.append("rag")
        else:
            # 일반적인 대화는 모든 에이전트의 도움을 받아 응답
            response = f"안녕하세요! '{chat_message.message}'에 대한 응답입니다.\n\n"
            response += "더 구체적인 요청을 해주시면 적절한 에이전트가 도움을 드릴 수 있습니다:\n"
            response += "• 코드 관련: '코드 리뷰를 해줘', '버그를 찾아줘'\n"
            response += "• 문서 검색: '프로젝트에 대해 알려줘', '사용법을 찾아줘'\n"
            response += "• 문서 작성: '문서를 작성해줘', 'API 문서를 만들어줘'\n"
            response += "• 일정 관리: '일정을 관리해줘', '마일스톤을 설정해줘'"

        processing_time = time.time() - start_time

        return ChatResponse(
            response=response,
            agents_used=agents_used,
            processing_time=round(processing_time, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"AI 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/ai/agents/{agent_type}",
    summary="특정 에이전트를 직접 호출합니다.",
    description="""
    관리자 에이전트의 분석 없이 특정 전문 에이전트를 직접 호출합니다.
    에이전트 타입은 URL 경로에 포함됩니다.
    """,
    tags=["AI Agents"],
)
async def call_specific_agent(agent_type: str, chat_message: ChatMessage):
    """특정 에이전트를 직접 호출"""
    try:
        if agent_type == "code":
            response = await code_agent.process(chat_message.message)
        elif agent_type == "document":
            response = await document_agent.process(chat_message.message)
        elif agent_type == "schedule":
            # ScheduleAgent가 선택되면, 웹 어댑터를 생성하여 실행
            web_adapter = WebAdapter()
            user_id = chat_message.user_id if chat_message.user_id else "web_user" # TODO: 실제 환경에서는 실제 사용자 ID를 사용해야 함
            channel_id = chat_message.channel_id if chat_message.channel_id else "web"
            
            # ScheduleAgent의 process 메서드 호출
            web_schedule_agent.process(
                message=chat_message.message,
                adapter=web_adapter, # WebAdapter 인스턴스 전달
                user_id=user_id,
                channel_id=channel_id
            )
            response = await web_adapter.get_response() # WebAdapter에 저장된 응답 가져오기
        elif agent_type == "rag":
            response = await rag_agent.process(chat_message.message)
        elif agent_type == "manager":
            analysis = await manager_agent.analyze_prompt(chat_message.message)
            response = (
                f"분석 결과: {json.dumps(analysis, ensure_ascii=False, indent=2)}"
            )
        else:
            raise HTTPException(
                status_code=404, detail=f"알 수 없는 에이전트 타입: {agent_type}"
            )

        return AgentResponse(
            message=response,
            agent_type=agent_type,
            timestamp=str(asyncio.get_event_loop().time()),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"에이전트 호출 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/ai/rag/add-documents",
    summary="RAG Agent에 새 문서를 추가합니다.",
    description="""
    RAG(Retrieval-Augmented Generation) 시스템에 문서 경로 목록을 제공하여
    새로운 지식 기반을 구축합니다.
    """,
    tags=["RAG Agent"],
    )
async def add_documents_to_rag(file_paths: List[str]):
    """RAG Agent에 새 문서 추가"""
    try:
        result = await rag_agent.add_documents(file_paths)
        return {"message": result}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"문서 추가 중 오류가 발생했습니다: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)