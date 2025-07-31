from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
from dotenv import load_dotenv
import openai
import json


from agents import CodeAgent

# 환경변수 로드
load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print(
        "⚠️  OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    )

app = FastAPI(
    title="팀 에이전트 시스템 AI 서버",
    description="멀티 에이전트 기반 AI 처리 서버",
    version="1.0.0",
)

# CORS 설정 (React 앱과 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic 모델들
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None


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
1. Code Agent: 코드 리뷰, 버그 탐지, 코드 품질 개선, Git 관리
2. Document Agent: 문서 작성, 편집, 검색, API 문서 생성
3. Schedule Agent: 프로젝트 일정 관리, 마일스톤 추적, 팀원 작업량 분배

응답 형식:
{
    "selected_agent": "code|document|schedule|general",
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


class ScheduleAgent:
    def __init__(self):
        self.name = "Schedule Agent"
        self.system_prompt = """당신은 전문적인 일정 관리 에이전트입니다. 
프로젝트 일정 관리, 마일스톤 추적, 팀원 작업량 분배 등을 담당합니다.

다음과 같은 작업을 수행할 수 있습니다:
- 프로젝트 일정 계획 및 관리
- 마일스톤 설정 및 추적
- 팀원 작업량 분배 및 조율
- 데드라인 관리 및 알림
- 스프린트 계획 및 리뷰
- 리소스 할당 및 최적화

항상 실용적이고 실행 가능한 일정 관리 방안을 제시하세요."""

    async def process(self, message: str) -> str:
        """일정 관련 요청 처리"""
        if not openai.api_key:
            return f"일정 에이전트가 처리 중입니다: {message}\n\n프로젝트 일정 관리, 마일스톤 추적, 팀원 작업량 분배 등의 작업을 수행할 수 있습니다."

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
            print(f"Schedule Agent 오류: {e}")
            return f"일정 에이전트 처리 중 오류가 발생했습니다: {str(e)}"


# 에이전트 인스턴스 생성
manager_agent = ManagerAgent()
code_agent = CodeAgent()
document_agent = DocumentAgent()
schedule_agent = ScheduleAgent()


@app.get("/")
async def root():
    return {"message": "팀 에이전트 시스템 AI 서버가 실행 중입니다."}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agents": ["manager", "code", "document", "schedule"],
        "openai_configured": bool(openai.api_key),
    }


@app.post("/ai/process", response_model=ChatResponse)
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
            response = await schedule_agent.process(chat_message.message)
            agents_used.append("schedule")
        else:
            # 일반적인 대화는 모든 에이전트의 도움을 받아 응답
            response = f"안녕하세요! '{chat_message.message}'에 대한 응답입니다.\n\n"
            response += "더 구체적인 요청을 해주시면 적절한 에이전트가 도움을 드릴 수 있습니다:\n"
            response += "• 코드 관련: '코드 리뷰를 해줘', '버그를 찾아줘'\n"
            response += "• 문서 관련: '문서를 작성해줘', 'API 문서를 만들어줘'\n"
            response += "• 일정 관련: '일정을 관리해줘', '마일스톤을 설정해줘'"

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


@app.post("/ai/agents/{agent_type}")
async def call_specific_agent(agent_type: str, chat_message: ChatMessage):
    """특정 에이전트를 직접 호출"""
    try:
        if agent_type == "code":
            response = await code_agent.process(chat_message.message)
        elif agent_type == "document":
            response = await document_agent.process(chat_message.message)
        elif agent_type == "schedule":
            response = await schedule_agent.process(chat_message.message)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
