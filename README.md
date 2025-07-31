ai-server 안에 env.example 을 .env로 바꾸고 본인 오픈API 키를 넣어서 사용하시면 됩니다.

# 팀 에이전트 시스템

멀티 에이전트 기반의 팀 관리 시스템입니다.

## 시스템 구조

```
관리자 Agent (Manager Agent)
├── 코드관리 Agent (Code Agent)
├── 문서관리 Agent (Document Agent)
└── 일정관리 Agent (Schedule Agent)
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
- **AI Framework**: OpenAI GPT-4o-mini
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

## 실행 방법

### 개발 환경 설정

```bash
# 1. FastAPI AI 서버 실행
cd ai-server
cp env.example .env
# .env 파일에 OpenAI API 키 설정
pip install -r requirements.txt
python main.py

# 2. React 앱 실행
cd frontend
npm install
npm run dev
```

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

### FastAPI AI 서버
- `POST /ai/process`: 사용자 프롬프트 처리 (중앙 관리자 Agent 호출)
- `POST /ai/agents/{agent_type}`: 특정 에이전트 직접 호출
- `GET /health`: 서버 상태 및 OpenAI 설정 확인

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

### 코드 관련
- "코드 리뷰를 해줘"
- "버그를 찾아줘"
- "Git 명령어 알려줘"

### 문서 관련
- "API 문서를 만들어줘"
- "README 파일 작성해줘"
- "기술 문서 작성해줘"

### 일정 관련
- "프로젝트 일정을 관리해줘"
- "마일스톤을 설정해줘"
- "스프린트 계획을 세워줘" 
