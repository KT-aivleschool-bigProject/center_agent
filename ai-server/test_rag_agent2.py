"""
RAG Agent 2 문서 로딩 및 벡터 DB 저장 테스트
"""
import asyncio
import os
from pathlib import Path
from agents.rag_agent2 import RAGAgent

async def test_document_loading():
    """문서 로딩 및 벡터 DB 저장 테스트"""
    print("🔄 RAG Agent 2 문서 로딩 테스트 시작...")
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("환경변수에 OPENAI_API_KEY를 설정해주세요.")
        return
    
    try:
        # RAG Agent 인스턴스 생성
        rag_agent = RAGAgent()
        
        # 문서 경로 확인
        docs_path = Path("data/docs")
        print(f"📁 문서 폴더 경로: {docs_path.absolute()}")
        
        if not docs_path.exists():
            print("❌ 문서 폴더가 존재하지 않습니다.")
            return
        
        # 문서 파일 목록 확인
        txt_files = list(docs_path.glob("*.txt"))
        pdf_files = list(docs_path.glob("*.pdf"))
        docx_files = list(docs_path.glob("*.docx"))
        
        print(f"📄 발견된 문서 파일:")
        print(f"  - TXT 파일: {len(txt_files)}개")
        for f in txt_files:
            print(f"    • {f.name}")
        print(f"  - PDF 파일: {len(pdf_files)}개")
        for f in pdf_files:
            print(f"    • {f.name}")
        print(f"  - DOCX 파일: {len(docx_files)}개")
        for f in docx_files:
            print(f"    • {f.name}")
        
        total_files = len(txt_files) + len(pdf_files) + len(docx_files)
        if total_files == 0:
            print("❌ 로딩할 문서가 없습니다.")
            return
        
        # 문서 로딩 및 벡터화 테스트
        print("\n🔄 문서 로딩 및 벡터화 시작...")
        await rag_agent._load_documents()
        
        # 벡터 저장소 확인
        if rag_agent.vectorstore:
            print("✅ 벡터 저장소 생성 성공!")
            
            # 벡터 DB 파일 확인
            vector_db_path = Path("data/vector_db")
            if (vector_db_path / "index.faiss").exists():
                print(f"💾 FAISS 인덱스 파일 저장됨: {vector_db_path / 'index.faiss'}")
            
            if (vector_db_path / "index.pkl").exists():
                print(f"💾 FAISS 메타데이터 파일 저장됨: {vector_db_path / 'index.pkl'}")
            
            # 간단한 검색 테스트
            print("\n🔍 벡터 검색 테스트...")
            test_query = "프로젝트에 대해 알려줘"
            retriever = rag_agent.vectorstore.as_retriever(search_kwargs={"k": 2})
            docs = retriever.get_relevant_documents(test_query)
            
            print(f"검색 쿼리: '{test_query}'")
            print(f"검색 결과: {len(docs)}개 문서")
            
            for i, doc in enumerate(docs):
                print(f"  📝 문서 {i+1}:")
                print(f"    - 내용 미리보기: {doc.page_content[:100]}...")
                print(f"    - 메타데이터: {doc.metadata}")
            
            # RetrievalQA 체인 테스트
            if rag_agent.retrieval_qa:
                print("\n💬 RetrievalQA 체인 테스트...")
                result = rag_agent.retrieval_qa({"query": test_query})
                print(f"질문: {test_query}")
                print(f"답변: {result['result']}")
                if result.get('source_documents'):
                    print(f"참조 문서: {len(result['source_documents'])}개")
            
        else:
            print("❌ 벡터 저장소 생성 실패")
        
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 작업 디렉토리를 ai-server로 변경
    os.chdir(Path(__file__).parent)
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # 비동기 테스트 실행
    asyncio.run(test_document_loading())
