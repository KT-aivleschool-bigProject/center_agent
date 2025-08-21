"""
Multi-Agent Orchestration Test Script
"""
import asyncio
import json
import httpx
from datetime import datetime


async def test_multi_agent_orchestration():
    """멀티에이전트 오케스트레이션 테스트"""
    
    base_url = "http://localhost:8000"
    
    print("🔄 멀티에이전트 오케스트레이션 테스트 시작...")
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            "name": "PR 리뷰 → 보안 점검",
            "request": {
                "message": "GitHub PR을 리뷰하고 보안 점검을 해주세요. 코드에 password='123456' 부분이 있습니다.",
                "user_id": "test_user_1",
                "data": {
                    "code": "def login(username, password='123456'): return authenticate(username, password)"
                }
            }
        },
        {
            "name": "문서 Q&A",
            "request": {
                "message": "프로젝트 API 문서에서 인증 관련 내용을 찾아주세요",
                "user_id": "test_user_2"
            }
        },
        {
            "name": "일정 제안",
            "request": {
                "message": "다음 주에 팀 회의 일정을 잡아주세요",
                "user_id": "test_user_3"
            }
        }
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🧪 테스트 {i}: {scenario['name']}")
            print("-" * 50)
            
            try:
                # 에이전트 워크플로우 실행
                response = await client.post(
                    f"{base_url}/agent/run",
                    json=scenario["request"]
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 성공: {result['message']}")
                    print(f"📊 처리 시간: {result['total_processing_time']:.2f}초")
                    print(f"🤖 실행된 에이전트 수: {len(result['agent_results'])}")
                    
                    # 에이전트 결과 출력
                    for j, agent_result in enumerate(result['agent_results'], 1):
                        print(f"  {j}. {agent_result['agent_name']}: {agent_result['message'][:100]}...")
                    
                    # 승인이 필요한 경우
                    if result['approval_required']:
                        print("⚠️  승인이 필요한 액션이 있습니다:")
                        approval_req = result['approval_request']
                        if approval_req:
                            print(f"   Agent: {approval_req['agent_name']}")
                            print(f"   Action: {approval_req['action_type']}")
                            print(f"   Message: {approval_req['message']}")
                            
                            # 자동 승인 테스트
                            print("🔄 자동 승인 테스트 중...")
                            approval_response = await client.post(
                                f"{base_url}/agent/approve",
                                json={
                                    "session_id": result['session_id'],
                                    "action": "approve",
                                    "comment": "테스트용 자동 승인"
                                }
                            )
                            
                            if approval_response.status_code == 200:
                                print("✅ 승인 완료")
                            else:
                                print(f"❌ 승인 실패: {approval_response.text}")
                    
                else:
                    print(f"❌ 실패 (HTTP {response.status_code}): {response.text}")
                    
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
    
    # 활성 세션 확인
    print(f"\n📋 활성 세션 확인...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/agent/sessions")
            if response.status_code == 200:
                sessions = response.json()
                print(f"활성 세션 수: {sessions['total_count']}")
            else:
                print(f"세션 조회 실패: {response.status_code}")
    except Exception as e:
        print(f"세션 조회 오류: {str(e)}")
    
    print("\n🎉 멀티에이전트 오케스트레이션 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_multi_agent_orchestration())
