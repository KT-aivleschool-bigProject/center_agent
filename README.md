ai-server 안에 env.example 을 .env로 바꾸고 본인 오픈API 키를 넣어서 사용하시면 됩니다.

# 팀 에이전트 시스템

멀티 에이전트 기반의 팀 관리 시스템입니다.

## 시스템 구조

```
관리자 Agent (Manager Agent)
├── 코드관리 Agent (Code Agent)
├── 문서관리 Agent (Document Agent)
├── 일정관리 Agent (Schedule Agent)
└── RAG Agent (문서 검색 및 지식 기반 질문답변)
```

## 에이전트 역할

### 🎯 관리자 Agent (Manager Agent)
- 사용자 프롬프트 분석 및 분류
- 적절한 에이전트 선택 및 호출
- 에이전트 간 협업 조율
- 최종 결과 통합 및 응답

### 💻 코드관리 Agent (Code Agent)
- 코드 리뷰 및 분석
- 버그 탐지 및 수정 제안
- 코드 품질 개선
- Git 저장소 관리

### 📄 문서관리 Agent (Document Agent)
- 문서 작성 및 편집 - 미정
- 문서 검색 및 요약 - 시현
- API 문서 생성 - 승훈
- 기술 문서 관리 - 미정

### 📅 일정관리 Agent (Schedule Agent)
- 프로젝트 일정 관리
- 마일스톤 추적
- 팀원 작업량 분배
- 데드라인 관리

### 🔍 RAG Agent (문서 검색)
- 문서 기반 질문 답변 - 완료
- 벡터 데이터베이스 검색 - 완료 
- LangGraph 워크플로우 - 완료
- 다중 파일 형식 지원 (txt, pdf, docx) - 완료

## 기술 스택

### 🚀 Backend
- **Web Server**: Spring Boot (향후 사용자 관리, 게시판용)
- **AI Server**: FastAPI + OpenAI GPT-4o-mini
- **Database**: PostgreSQL
- **Container**: Docker

### 🎨 Frontend
- **Framework**: React
- **State Management**: Redux Toolkit
- **UI Library**: Material-UI
- **Build Tool**: Vite

### 🤖 AI/ML
- **AI Framework**: OpenAI GPT-4o-mini, GPT-4o
- **Vector Database**: ChromaDB
- **RAG Framework**: LangChain, LangGraph
- **Embeddings**: OpenAI Embeddings
- **API**: FastAPI (AI 서비스 전용)
- **Agent System**: 멀티 에이전트 아키텍처

## 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React App     │    │  FastAPI        │    │   OpenAI        │
│   (Frontend)    │◄──►│   (AI Server)   │◄──►│   GPT-4o-mini   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   (Database)    │
                       └─────────────────┘
```

## 에이전트 시스템

### 🎯 중앙 관리자 Agent (Manager Agent)
- **역할**: 사용자 프롬프트 분석 및 적절한 에이전트 선택
- **기능**: 
  - OpenAI GPT-4o-mini를 사용한 지능적 분석
  - 키워드 기반 fallback 분석
  - 에이전트 선택 신뢰도 평가

### 💻 코드관리 Agent (Code Agent)
- **역할**: 코드 리뷰, 버그 탐지, 코드 품질 개선
- **기능**:
  - 코드 리뷰 및 개선 제안
  - 버그 탐지 및 수정 방법 제시
  - Git 명령어 및 워크플로우 안내
  - 프로그래밍 언어별 모범 사례 제시

### 📄 문서관리 Agent (Document Agent)
- **역할**: 문서 작성, 편집, API 문서 생성
- **기능**:
  - 기술 문서 작성 및 편집
  - API 문서 생성 및 관리
  - README 파일 작성 가이드
  - 마크다운 형식 문서 작성

### 📅 일정관리 Agent (Schedule Agent)
- **역할**: 프로젝트 일정 관리, 마일스톤 추적
- **기능**:
  - 프로젝트 일정 계획 및 관리
  - 마일스톤 설정 및 추적
  - 팀원 작업량 분배 및 조율
  - 스프린트 계획 및 리뷰

### 🔍 RAG Agent (문서 검색)
- **역할**: 문서 기반 질문 답변 및 지식 검색
- **기능**:
  - LangGraph 기반 워크플로우 (entry → retrieve → generate → output)
  - ChromaDB 벡터 데이터베이스 활용
  - 다중 파일 형식 지원 (txt, pdf, docx)
  - 유사도 기반 문서 검색
  - OpenAI GPT-4o 모델로 RAG 방식 답변 생성
  - 코드 관련 질문 분기점 (향후 code_agent 연동)

## 실행 방법

### 📋 사전 준비

1. **OpenAI API 키 설정**
```bash
# ai-server 디렉토리에서
cp env.example .env
# .env 파일에 OpenAI API 키 설정
OPENAI_API_KEY=your_openai_api_key_here
```

2. **Python 가상환경 설정** (선택사항)
```bash
# 프로젝트 루트에서
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux  
source .venv/bin/activate
```

### 🚀 수동 실행 방법

#### 1단계: AI 서버 (Backend) 실행

```bash
# 1. ai-server 디렉토리로 이동
cd center_agent/ai-server

# 2. Python 패키지 설치
pip install -r requirements.txt

# 3. FastAPI 서버 실행
python main.py
```

**서버 실행 확인:**
- 브라우저에서 `http://localhost:8003` 접속
- 또는 `http://localhost:8003/health` 에서 상태 확인

#### 2단계: Frontend 실행

**새 터미널 창에서:**

```bash
# 1. frontend 디렉토리로 이동
cd center_agent/frontend

# 2. Node.js 패키지 설치
npm install

# 3. React 개발 서버 실행
npm run dev
```

**Frontend 접속:**
- 브라우저에서 `http://localhost:5173` 접속

### 🧪 테스트 방법

웹 인터페이스에서 다음 명령어들을 시도해보세요:

#### RAG Agent 테스트
```
"프로젝트에 대해 알려줘"
"API 엔드포인트가 뭐가 있나요?"
"사용 가능한 에이전트는?"
"기술 스택은 무엇인가요?"
```

#### 다른 Agent 테스트
```
"코드 리뷰를 해줘" (Code Agent)
"문서를 작성해줘" (Document Agent)  
"일정을 관리해줘" (Schedule Agent)
```

### 🔧 포트 설정

- **AI 서버**: http://localhost:8003
- **Frontend**: http://localhost:5173
- **포트 충돌시**: `main.py`에서 포트 번호 변경 가능

### 📁 문서 추가 방법

RAG Agent에 새 문서를 추가하려면:

1. **파일 업로드**:
```bash
# ai-server/data/docs/ 폴더에 파일 복사
cp your_document.txt ai-server/data/docs/
```

2. **API 호출**:
```bash
curl -X POST "http://localhost:8003/ai/rag/add-documents" \
     -H "Content-Type: application/json" \
     -d '{"file_paths": ["data/docs/your_document.txt"]}'
```

### ⚠️ 문제 해결

#### OpenAI API 키 없이 테스트
API 키가 없어도 키워드 기반 분석으로 작동합니다.

#### 포트 충돌 오류
다른 포트를 사용 중일 경우 `main.py`의 포트 번호를 변경하세요:
```python
uvicorn.run(app, host="0.0.0.0", port=8004)  # 포트 변경
```

#### CORS 오류
브라우저에서 CORS 오류 발생시 `main.py`의 CORS 설정에 포트를 추가하세요.

### Docker 환경 실행

```bash
# 전체 시스템 실행
docker-compose up -d
```

## 환경 설정

### OpenAI API 설정
1. `ai-server/env.example`를 `ai-server/.env`로 복사
2. OpenAI API 키 설정:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### API 키 없이 테스트
OpenAI API 키가 설정되지 않아도 키워드 기반 분석으로 작동합니다.

## API 엔드포인트

### FastAPI AI 서버 (http://localhost:8003)
- `GET /`: 서버 상태 확인
- `GET /health`: 서버 상태 및 OpenAI 설정 확인
- `POST /ai/process`: 사용자 프롬프트 처리 (중앙 관리자 Agent 호출)
- `POST /ai/agents/{agent_type}`: 특정 에이전트 직접 호출
  - agent_type: `manager`, `code`, `document`, `schedule`, `rag`
- `POST /ai/rag/add-documents`: RAG Agent에 새 문서 추가

### 요청/응답 예시

**프롬프트 처리:**
```bash
curl -X POST "http://localhost:8003/ai/process" \
     -H "Content-Type: application/json" \
     -d '{"message": "프로젝트에 대해 알려줘"}'
```

**특정 에이전트 호출:**
```bash
curl -X POST "http://localhost:8003/ai/agents/rag" \
     -H "Content-Type: application/json" \
     -d '{"message": "API 문서가 어디에 있나요?"}'
```

## 시스템 플로우

```
1. 사용자가 React 앱에서 메시지 입력
   ↓
2. FastAPI AI 서버로 메시지 전송
   ↓
3. 중앙 관리자 Agent가 OpenAI GPT-3.5-turbo로 프롬프트 분석
   ↓
4. 적절한 전문 에이전트 호출 (코드/문서/일정 관리 Agent)
   ↓
5. 각 에이전트가 OpenAI API로 전문적인 응답 생성
   ↓
6. 결과를 React 앱으로 반환하여 사용자에게 표시
```

## 테스트 예시

### 🔍 RAG Agent (문서 검색)
```
"프로젝트에 대해 알려줘"
"API 엔드포인트가 뭐가 있나요?"
"사용 가능한 에이전트는?"
"기술 스택은 무엇인가요?"
"문서에서 FastAPI 정보 찾아줘"
```

### 💻 코드 관련
```
"코드 리뷰를 해줘"
"버그를 찾아줘"
"Git 명령어 알려줘"
```

### 📄 문서 관련
```
"API 문서를 만들어줘"
"README 파일 작성해줘"
"기술 문서 작성해줘"
```

### 📅 일정 관련
```
"프로젝트 일정을 관리해줘"
"마일스톤을 설정해줘"
"스프린트 계획을 세워줘"
```

### 🎯 자동 에이전트 선택 테스트
Manager Agent가 자동으로 적절한 에이전트를 선택합니다:
- "찾아줘", "알려줘" → RAG Agent
- "코드", "개발" → Code Agent  
- "문서 작성" → Document Agent
- "일정", "스케줄" → Schedule Agent 
