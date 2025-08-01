from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import os
import json
import time
from dotenv import load_dotenv
import openai

# 사용자 정의 에이전트들
from agents import CodeAgent
from agents.rag_agent import RAGAgent

# 환경변수 로드
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("⚠️  OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요.")

app = FastAPI(
    title="팀 에이전트 시스템 AI 서버",
    description="멀티 에이전트 기반 AI 처리 서버",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델
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

# ManagerAgent
class ManagerAgent:
    def __init__(self):
        self.system_prompt = """당신은 팀 에이전트 시스템의 중앙 관리자입니다. 
사용자의 요청을 분석하고 적절한 전문 에이전트를 선택해야 합니다.

사용 가능한 에이전트:
1. Code Agent
2. Document Agent
3. Schedule Agent
4. RAG Agent

응답 형식:
{
    "selected_agent": "code|document|schedule|rag|general",
    "reason": "선택 이유",
    "confidence": 0.0-1.0
}"""

    async def analyze_prompt(self, message: str) -> dict:
        if not openai.api_key:
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
            return json.loads(result)
        except Exception as e:
            print(f"[Manager 오류] {e}")
            return self._fallback_analysis(message)

    def _fallback_analysis(self, message: str) -> dict:
        text = message.lower()
        if any(w in text for w in ["코드", "버그", "리뷰", "git"]):
            return {"selected_agent": "code", "reason": "코드 관련 키워드", "confidence": 0.8}
        if any(w in text for w in ["문서", "작성", "readme", "편집", "요약"]):
            return {"selected_agent": "document", "reason": "문서 관련 키워드", "confidence": 0.8}
        if any(w in text for w in ["일정", "프로젝트", "마일스톤", "데드라인"]):
            return {"selected_agent": "schedule", "reason": "일정 관련 키워드", "confidence": 0.8}
        if any(w in text for w in ["질문", "자료", "pdf", "문서 질문", "내용 요약"]):
            return {"selected_agent": "rag", "reason": "RAG 관련 키워드", "confidence": 0.8}
        return {"selected_agent": "general", "reason": "일반적 요청", "confidence": 0.5}

# 다른 에이전트들 (요약)
class DocumentAgent:
    def __init__(self):
        self.system_prompt = "당신은 전문적인 문서 에이전트입니다."

    async def process(self, message: str) -> str:
        if not openai.api_key:
            return "문서 에이전트가 처리 중입니다."
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
            return f"[문서 에이전트 오류] {str(e)}"

class ScheduleAgent:
    def __init__(self):
        self.system_prompt = "당신은 일정 관리 전문가입니다."

    async def process(self, message: str) -> str:
        if not openai.api_key:
            return "일정 에이전트가 처리 중입니다."
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
            return f"[일정 에이전트 오류] {str(e)}"

# 에이전트 인스턴스 생성
manager_agent = ManagerAgent()
code_agent = CodeAgent()
document_agent = DocumentAgent()
schedule_agent = ScheduleAgent()
rag_agent = RAGAgent()

# 엔드포인트
@app.get("/")
async def root():
    return {"message": "AI 서버 실행 중"}

@app.get("/health")
async def health():
    return {"status": "ok", "agents": ["manager", "code", "document", "schedule", "rag"]}

@app.post("/ai/process", response_model=ChatResponse)
async def process_chat(chat_message: ChatMessage):
    start = time.time()
    analysis = await manager_agent.analyze_prompt(chat_message.message)
    selected = analysis.get("selected_agent", "general")

    agents_used = ["manager"]
    if selected == "code":
        response = await code_agent.process(chat_message.message)
        agents_used.append("code")
    elif selected == "document":
        response = await document_agent.process(chat_message.message)
        agents_used.append("document")
    elif selected == "schedule":
        response = await schedule_agent.process(chat_message.message)
        agents_used.append("schedule")
    elif selected == "rag":
        response = await rag_agent.process(chat_message.message)
        agents_used.append("rag")
    else:
        response = "일반적인 요청입니다. 좀 더 구체적으로 말씀해주세요."
    
    return ChatResponse(
        response=response,
        agents_used=agents_used,
        processing_time=round(time.time() - start, 2),
    )

@app.post("/ai/agents/{agent_type}", response_model=AgentResponse)
async def call_agent(agent_type: str, chat_message: ChatMessage):
    try:
        if agent_type == "code":
            res = await code_agent.process(chat_message.message)
        elif agent_type == "document":
            res = await document_agent.process(chat_message.message)
        elif agent_type == "schedule":
            res = await schedule_agent.process(chat_message.message)
        elif agent_type == "manager":
            res = json.dumps(await manager_agent.analyze_prompt(chat_message.message), ensure_ascii=False)
        elif agent_type == "rag":
            res = await rag_agent.process(chat_message.message)
        else:
            raise HTTPException(status_code=404, detail="알 수 없는 에이전트")
        return AgentResponse(message=res, agent_type=agent_type, timestamp=str(asyncio.get_event_loop().time()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
