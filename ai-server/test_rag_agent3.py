"""
RAG Agent 3 실험/테스트 코드
- 로컬 파일 추가 테스트
- URL 다운로드 테스트
- 벡터 검색 테스트
- 질문답변 테스트
"""
import asyncio
import os
from pathlib import Path
from agents.rag_agent3 import RAGAgent

async def test_rag_agent3():
    """RAG Agent 3 종합 실험"""
    print("🚀 RAG Agent 3 실험 시작...")
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("환경변수에 OPENAI_API_KEY를 설정해주세요.")
        return
    
    try:
        # RAG Agent 3 인스턴스 생성
        print("\n📦 RAG Agent 3 초기화...")
        rag_agent = RAGAgent()
        
        # 1. 기존 로컬 문서 확인
        print("\n📁 기존 로컬 문서 확인...")
        docs_path = Path("data/docs")
        if docs_path.exists():
            txt_files = list(docs_path.glob("*.txt"))
            pdf_files = list(docs_path.glob("*.pdf"))
            docx_files = list(docs_path.glob("*.docx"))
            
            print(f"  - TXT 파일: {len(txt_files)}개")
            for f in txt_files[:3]:  # 최대 3개만 표시
                print(f"    • {f.name}")
            print(f"  - PDF 파일: {len(pdf_files)}개")
            for f in pdf_files[:3]:
                print(f"    • {f.name}")
            print(f"  - DOCX 파일: {len(docx_files)}개")
            for f in docx_files[:3]:
                print(f"    • {f.name}")
        
        # 2. 테스트 문서 추가 (로컬 파일 + URL)
        print("\n📝 테스트 문서 추가...")
        
        # 로컬 파일 경로들 (실제 존재하는 파일들)
        local_sources = []
        if docs_path.exists():
            local_sources.extend([str(f) for f in docs_path.glob("*.txt")][:2])
            local_sources.extend([str(f) for f in docs_path.glob("*.pdf")][:1])
        
        # 테스트용 URL들 (실제 다운로드 가능한 URL)
        test_urls = [
            # GitHub README 예시 (Raw URL)
            "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
            # 간단한 텍스트 파일 예시
            "https://www.w3.org/TR/PNG/iso_8859-1.txt",
        ]
        
        # 테스트할 소스들 결합
        test_sources = local_sources + test_urls
        
        if test_sources:
            print(f"추가할 문서 소스: {len(test_sources)}개")
            for i, src in enumerate(test_sources):
                if src.startswith("http"):
                    print(f"  {i+1}. [URL] {src}")
                else:
                    print(f"  {i+1}. [로컬] {Path(src).name}")
            
            # 문서 추가 실행
            print("\n⏳ 문서 추가 중...")
            result = await rag_agent.add_documents(test_sources)
            print(f"✅ 결과: {result}")
        else:
            print("⚠️  테스트할 문서가 없습니다. 기존 문서로 진행...")
        
        # 3. 벡터 저장소 상태 확인
        print("\n🔍 벡터 저장소 상태 확인...")
        if rag_agent.vectorstore:
            print("✅ 벡터 저장소가 정상적으로 로드되었습니다.")
            
            # FAISS 파일 확인
            vector_db_path = Path("data/vector_db")
            if (vector_db_path / "index.faiss").exists():
                file_size = (vector_db_path / "index.faiss").stat().st_size
                print(f"💾 FAISS 인덱스 파일 크기: {file_size:,} bytes")
        else:
            print("❌ 벡터 저장소가 없습니다.")
            return
        
        # 4. 벡터 검색 테스트
        print("\n🔍 벡터 검색 테스트...")
        test_queries = [
            "프로젝트에 대해 알려줘",
            "API 문서는 어디에 있나요?",
            "시스템 아키텍처 설명해줘",
            "설치 방법을 알려줘"
        ]
        
        retriever = rag_agent.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        for query in test_queries:
            print(f"\n📋 검색 쿼리: '{query}'")
            try:
                docs = retriever.get_relevant_documents(query)
                print(f"  검색 결과: {len(docs)}개 문서")
                
                for i, doc in enumerate(docs):
                    preview = doc.page_content[:100].replace('\n', ' ')
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"    {i+1}. {preview}... [출처: {Path(source).name}]")
                    
            except Exception as e:
                print(f"  ❌ 검색 오류: {e}")
        
        # 5. 질문답변 테스트
        print("\n💬 질문답변 테스트...")
        qa_queries = [
            "프로젝트의 주요 기능은 무엇인가요?",
            "설치하는 방법을 단계별로 알려주세요",
            "이 시스템의 아키텍처는 어떻게 구성되어 있나요?"
        ]
        
        for query in qa_queries:
            print(f"\n❓ 질문: {query}")
            try:
                answer = await rag_agent.process(query)
                print(f"💡 답변: {answer}")
                print("-" * 50)
            except Exception as e:
                print(f"❌ 답변 생성 오류: {e}")
        
        # 6. 성능 테스트
        print("\n⚡ 성능 테스트...")
        import time
        
        perf_query = "프로젝트에 대해 간단히 설명해주세요"
        start_time = time.time()
        
        try:
            answer = await rag_agent.process(perf_query)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"처리 시간: {processing_time:.2f}초")
            print(f"답변 길이: {len(answer)}자")
        except Exception as e:
            print(f"❌ 성능 테스트 오류: {e}")
        
        print("\n🎉 RAG Agent 3 실험 완료!")
        
    except Exception as e:
        print(f"❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def test_url_download_only():
    """URL 다운로드 기능만 단독 테스트"""
    print("🌐 URL 다운로드 단독 테스트...")
    
    from agents.rag_agent3 import download_file_from_url
    
    test_urls = [
        "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
        "https://www.w3.org/TR/PNG/iso_8859-1.txt",
    ]
    
    for url in test_urls:
        print(f"\n📥 다운로드 테스트: {url}")
        try:
            content = await download_file_from_url(url)
            if content:
                print(f"✅ 성공: {len(content):,} bytes 다운로드")
                # 텍스트 미리보기 (처음 200자)
                try:
                    preview = content.decode('utf-8')[:200]
                    print(f"미리보기: {preview}...")
                except:
                    print("바이너리 파일입니다.")
            else:
                print("❌ 다운로드 실패")
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    # 작업 디렉토리를 ai-server로 변경
    os.chdir(Path(__file__).parent)
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # 메뉴 선택
    print("\n🔬 RAG Agent 3 실험 메뉴:")
    print("1. 전체 종합 실험")
    print("2. URL 다운로드만 테스트")
    
    choice = input("선택하세요 (1-2): ").strip()
    
    if choice == "1":
        asyncio.run(test_rag_agent3())
    elif choice == "2":
        asyncio.run(test_url_download_only())
    else:
        print("잘못된 선택입니다. 전체 실험을 실행합니다.")
        asyncio.run(test_rag_agent3())
