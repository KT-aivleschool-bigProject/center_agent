"""
간단한 API 테스트 스크립트
"""
import asyncio
import httpx
import json


async def simple_test():
    """간단한 API 테스트"""
    base_url = "http://localhost:8005"
    
    print("🔗 서버 연결 테스트...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1. Health check (docs 엔드포인트)
            print("1. API 문서 페이지 확인...")
            response = await client.get(f"{base_url}/docs")
            print(f"   응답 코드: {response.status_code}")
            
            # 2. Agent 실행 테스트
            print("2. 멀티에이전트 실행 테스트...")
            request_data = {
                "message": "안녕하세요, 프로젝트에 대해 알려주세요",
                "user_id": "test_user"
            }
            
            response = await client.post(
                f"{base_url}/agent/run",
                json=request_data
            )
            
            print(f"   응답 코드: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   성공: {result['message']}")
                print(f"   실행된 에이전트 수: {len(result['agent_results'])}")
                for agent_result in result['agent_results']:
                    print(f"     - {agent_result['agent_name']}: {agent_result['message']}")
            else:
                print(f"   실패: {response.text}")
            
    except Exception as e:
        print(f"❌ 오류: {str(e)}")


if __name__ == "__main__":
    asyncio.run(simple_test())
