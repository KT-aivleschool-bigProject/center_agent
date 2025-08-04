import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class RAGState(TypedDict):
    """RAG Agent의 상태 정의"""
    question: str
    retrieved_docs: List[Document]
    answer: str
    error: Optional[str]
    metadata: Dict[str, Any]


class RAGAgent:
    """RAG 기반 LangGraph Agent"""
    
    def __init__(self):
        self.name = "RAG Agent"
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retrieval_qa = None
        self.graph = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # 문서 저장 경로
        self.docs_path = Path("data/docs")
        self.vector_db_path = Path("data/vector_db")
        
        # 초기화
        self._initialize_components()
        self._setup_graph()
        
        # 문서는 첫 번째 요청시 로딩 (비동기 함수이므로 별도 호출 필요)
        self._documents_loaded = False
    
    def _initialize_components(self):
        """LangChain 구성 요소 초기화"""
        try:
            # OpenAI API 키 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("  RAG Agent: OpenAI API 키가 설정되지 않았습니다.")
                return
            
            # LLM 초기화
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=api_key
            )
            
            # 임베딩 모델 초기화
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
            
            # 벡터 저장소 디렉토리 생성
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            
            print(" RAG Agent 구성 요소가 초기화되었습니다.")
            
        except Exception as e:
            print(f" RAG Agent 초기화 오류: {e}")
    
    def _setup_graph(self):
        """LangGraph 워크플로우 설정"""
        # 상태 그래프 생성
        workflow = StateGraph(RAGState)
        
        # 노드 추가
        workflow.add_node("entry", self._entry_node)
        workflow.add_node("retrieve_docs", self._retrieve_docs_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("output", self._output_node)
        
        # 엣지 설정
        workflow.set_entry_point("entry")
        workflow.add_edge("entry", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "generate_answer")
        workflow.add_edge("generate_answer", "output")
        workflow.add_edge("output", END)
        
        # 그래프 컴파일
        self.graph = workflow.compile()
    
    async def _load_documents(self):
        """문서 디렉토리에서 실제 문서들을 로딩"""
        try:
            # 문서 디렉토리 생성 (존재하지 않을 경우)
            self.docs_path.mkdir(parents=True, exist_ok=True)
            
            # 문서 로딩 및 벡터화
            await self._load_and_vectorize_documents()
            
        except Exception as e:
            print(f"❌ 문서 로딩 오류: {e}")
    
    async def _load_and_vectorize_documents(self):
        """문서를 로딩하고 벡터화하여 저장"""
        try:
            if not self.embeddings:
                print("  임베딩 모델이 초기화되지 않았습니다.")
                return
            
            documents = []
            
            # 텍스트 파일 직접 로딩
            for txt_file in self.docs_path.glob("*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        doc = Document(
                            page_content=content,
                            metadata={"source": str(txt_file)}
                        )
                        documents.append(doc)
                        print(f" 로딩 완료: {txt_file.name}")
                except Exception as e:
                    print(f" {txt_file.name} 로딩 실패: {e}")
            
            # PDF 파일 로딩
            for pdf_file in self.docs_path.glob("*.pdf"):
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    documents.extend(docs)
                    print(f" PDF 로딩 완료: {pdf_file.name}")
                except Exception as e:
                    print(f" {pdf_file.name} 로딩 실패: {e}")
            
            # DOCX 파일 로딩
            for docx_file in self.docs_path.glob("*.docx"):
                try:
                    loader = UnstructuredWordDocumentLoader(str(docx_file))
                    docs = loader.load()
                    documents.extend(docs)
                    print(f" DOCX 로딩 완료: {docx_file.name}")
                except Exception as e:
                    print(f" {docx_file.name} 로딩 실패: {e}")
            
            if documents:
                # 문서 분할
                texts = self.text_splitter.split_documents(documents)
                print(f"📝 {len(documents)}개 문서를 {len(texts)}개 청크로 분할했습니다.")
                
                # 벡터 저장소 생성/업데이트
                if self.vectorstore is None:
                    self.vectorstore = Chroma.from_documents(
                        documents=texts,
                        embedding=self.embeddings,
                        persist_directory=str(self.vector_db_path)
                    )
                    print("✅ 새로운 벡터 저장소를 생성했습니다.")
                else:
                    self.vectorstore.add_documents(texts)
                    print("✅ 기존 벡터 저장소에 문서를 추가했습니다.")
                
                # RetrievalQA 체인 설정
                self._setup_retrieval_qa()
                
                print(f"🎯 벡터화 완료: {len(documents)}개 문서, {len(texts)}개 텍스트 청크")
            else:
                print("⚠️  data/docs 폴더에 문서 파일이 없습니다.")
                print("💡 문서 파일(.txt, .pdf, .docx)을 data/docs 폴더에 추가해주세요.")
                
        except Exception as e:
            print(f" 문서 벡터화 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_retrieval_qa(self):
        """RetrievalQA 체인 설정"""
        if not self.vectorstore or not self.llm:
            return
        
        # 프롬프트 템플릿 설정
        prompt_template = """
당신은 문서 기반 질문 답변 AI입니다. 제공된 문서 컨텍스트를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 해주세요.

컨텍스트:
{context}

질문: {question}

답변 가이드라인:
1. 제공된 컨텍스트를 주로 활용하되, 필요시 일반적인 지식도 함께 활용하세요.
2. 답변은 명확하고 구체적으로 작성하세요.
3. 컨텍스트에서 정확한 정보를 찾을 수 없다면 그 사실을 명시하세요.
4. 한국어로 답변하세요.

답변:
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA 체인 생성
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    async def _entry_node(self, state: RAGState) -> RAGState:
        """진입점: 질문 입력 처리"""
        print(f"🔍 RAG Agent: 질문 처리 시작 - {state['question']}")
        
        # 코드 관련 질문 분기점 (향후 code_agent 포워딩용)
        question_lower = state["question"].lower()
        code_keywords = ["코드", "구현", "개발", "프로그래밍", "함수", "클래스", "버그"]
        
        if any(keyword in question_lower for keyword in code_keywords):
            state["metadata"] = {
                "potential_code_question": True,
                "forward_to_code_agent": False  # 현재는 비활성화, 향후 확장시 True로 변경
            }
            print("💡 코드 관련 질문으로 감지됨 (향후 code_agent 포워딩 고려)")
        else:
            state["metadata"] = {"potential_code_question": False}
        
        return state
    
    async def _retrieve_docs_node(self, state: RAGState) -> RAGState:
        """문서 검색 노드"""
        try:
            if not self.vectorstore:
                state["error"] = "벡터 저장소가 초기화되지 않았습니다. 문서를 먼저 로딩해주세요."
                print("❌ 벡터 저장소가 없습니다. 문서 로딩을 다시 시도합니다.")
                await self._load_documents()  # 문서 재로딩 시도
                if not self.vectorstore:
                    return state
            
            # 관련 문서 검색
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            docs = retriever.get_relevant_documents(state["question"])
            state["retrieved_docs"] = docs
            
            print(f"📚 {len(docs)}개의 관련 문서를 검색했습니다.")
            
            # 검색된 문서가 없을 경우 대체 응답 준비
            if not docs:
                state["error"] = "관련 문서를 찾을 수 없습니다."
                print("  검색된 문서가 없습니다.")
            
        except Exception as e:
            state["error"] = f"문서 검색 중 오류: {str(e)}"
            print(f" 문서 검색 오류: {e}")
        
        return state
    
    async def _generate_answer_node(self, state: RAGState) -> RAGState:
        """답변 생성 노드"""
        try:
            if state.get("error"):
                # 오류가 있지만 문서가 없는 경우 대체 응답 생성
                if "벡터 저장소" in state["error"] or "관련 문서를 찾을 수 없습니다" in state["error"]:
                    state["answer"] = f"""죄송합니다. 현재 문서 저장소에 관련 정보가 없어 정확한 답변을 드릴 수 없습니다.

질문: {state["question"]}

대신 일반적인 도움을 드릴 수 있습니다:
• 프로젝트 문서를 업로드해주시면 더 정확한 답변을 제공할 수 있습니다.
• 구체적인 질문을 해주시면 다른 전문 에이전트가 도움을 드릴 수 있습니다.
• 코드 관련 질문은 'code' 에이전트에게, 문서 작성은 'document' 에이전트에게 문의해보세요."""
                    state["error"] = None  # 에러를 클리어하여 정상 응답으로 처리
                    print("📝 대체 응답을 생성했습니다.")
                    return state
                else:
                    return state
            
            if not self.retrieval_qa:
                state["error"] = "RetrievalQA 체인이 초기화되지 않았습니다."
                return state
            
            # RAG 방식으로 답변 생성
            result = self.retrieval_qa({"query": state["question"]})
            state["answer"] = result["result"]
            
            # 소스 문서 정보 추가
            if result.get("source_documents"):
                state["metadata"]["source_count"] = len(result["source_documents"])
                state["answer"] += f"\n\n📋 참조한 문서: {len(result['source_documents'])}개"
            
            print(" RAG 방식으로 답변이 생성되었습니다.")
            
        except Exception as e:
            state["error"] = f"답변 생성 중 오류: {str(e)}"
            print(f" 답변 생성 오류: {e}")
        
        return state
    
    async def _output_node(self, state: RAGState) -> RAGState:
        """출력 노드"""
        if state.get("error"):
            state["answer"] = f"처리 중 오류가 발생했습니다: {state['error']}"
        
        print(" RAG Agent 처리 완료")
        return state
    
    async def process(self, message: str) -> str:
        """메시지 처리 (FastAPI 통합용)"""
        try:
            # 첫 번째 요청시 문서 로딩
            if not self._documents_loaded:
                await self._load_documents()
                self._documents_loaded = True
            
            # 초기 상태 설정
            initial_state = RAGState(
                question=message,
                retrieved_docs=[],
                answer="",
                error=None,
                metadata={}
            )
            
            # 그래프 실행
            if self.graph:
                result = await self.graph.ainvoke(initial_state)
                return result["answer"]
            else:
                return "RAG Agent가 초기화되지 않았습니다. OpenAI API 키를 확인해주세요."
        
        except Exception as e:
            return f"RAG Agent 처리 중 오류가 발생했습니다: {str(e)}"
    
    async def add_documents(self, file_paths: List[str]) -> str:
        """새 문서 추가"""
        try:
            documents = []
            for file_path in file_paths:
                path = Path(file_path)
                if path.suffix == ".txt":
                    loader = TextLoader(str(path))
                elif path.suffix == ".pdf":
                    loader = PyPDFLoader(str(path))
                elif path.suffix == ".docx":
                    loader = UnstructuredWordDocumentLoader(str(path))
                else:
                    continue
                
                docs = loader.load()
                documents.extend(docs)
            
            if documents and self.vectorstore:
                texts = self.text_splitter.split_documents(documents)
                self.vectorstore.add_documents(texts)
                return f"{len(documents)}개 문서가 추가되었습니다."
            else:
                return "추가할 수 있는 문서가 없습니다."
        
        except Exception as e:
            return f"문서 추가 중 오류: {str(e)}"
