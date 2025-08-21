"""
멀티에이전트 오케스트레이션 테스트 스크립트
"""
import asyncio
import json
import httpx
from datetime import datetime


class MultiAgentTester:
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        
    async def test_code_review_workflow(self):
        """코드 리뷰 → 보안 검사 워크플로우 테스트"""
        print("🔍 테스트 1: 코드 리뷰 + 보안 검사 워크플로우")
        
        request_data = {
            "message": "다음 코드를 리뷰해주세요: def login(username, password): return username == 'admin' and password == 'password123'",
            "user_id": "test_user",
            "data": {
                "code": "def login(username, password): return username == 'admin' and password == 'password123'",
                "language": "python"
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/agent/run",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 성공: {result['message']}")
                    print(f"📊 실행된 에이전트: {len(result['agent_results'])}개")
                    
                    for i, agent_result in enumerate(result['agent_results']):
                        print(f"  {i+1}. {agent_result['agent_name']}: {agent_result['message']}")
                    
                    if result['approval_required']:
                        approval_msg = result.get('approval_request', {}).get('message', 'Approval required')
                        print(f"⚠️ 승인 필요: {approval_msg}")
                        return result['session_id']
                    
                    return None
                else:
                    print(f"❌ 실패: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ 연결 오류: {str(e)}")
            return None

    async def test_document_search_workflow(self):
        """문서 검색 워크플로우 테스트"""
        print("\n📚 테스트 2: 문서 검색 워크플로우")
        
        request_data = {
            "message": "프로젝트 개요에 대해 알려주세요",
            "user_id": "test_user",
            "data": {
                "search_type": "document"
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/agent/run",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 성공: {result['message']}")
                    print(f"📊 실행된 에이전트: {len(result['agent_results'])}개")
                    
                    for i, agent_result in enumerate(result['agent_results']):
                        print(f"  {i+1}. {agent_result['agent_name']}: {agent_result['message']}")
                    
                    return result.get('session_id')
                else:
                    print(f"❌ 실패: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ 연결 오류: {str(e)}")
            return None

    async def test_schedule_workflow(self):
        """일정 관리 워크플로우 테스트"""
        print("\n📅 테스트 3: 일정 관리 워크플로우")
        
        request_data = {
            "message": "내일 오후 2시에 팀 회의를 잡아주세요",
            "user_id": "test_user",
            "data": {
                "meeting_type": "team_meeting",
                "participants": ["team@company.com"]
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/agent/run",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 성공: {result['message']}")
                    print(f"📊 실행된 에이전트: {len(result['agent_results'])}개")
                    
                    for i, agent_result in enumerate(result['agent_results']):
                        print(f"  {i+1}. {agent_result['agent_name']}: {agent_result['message']}")
                    
                    if result['approval_required']:
                        approval_msg = result.get('approval_request', {}).get('message', 'Approval required')
                        print(f"⚠️ 승인 필요: {approval_msg}")
                        return result['session_id']
                    
                    return None
                else:
                    print(f"❌ 실패: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ 연결 오류: {str(e)}")
            return None

    async def test_approval_workflow(self, session_id: str):
        """승인 워크플로우 테스트"""
        print(f"\n✅ 테스트 4: 승인 워크플로우 (세션: {session_id})")
        
        # 승인 처리
        approval_data = {
            "session_id": session_id,
            "action": "approve",
            "comment": "테스트 승인입니다"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/agent/approve",
                    json=approval_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 승인 완료: {result['message']}")
                    return True
                else:
                    print(f"❌ 승인 실패: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ 승인 오류: {str(e)}")
            return False

    async def test_active_sessions(self):
        """활성 세션 조회 테스트"""
        print("\n📋 테스트 5: 활성 세션 조회")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/agent/sessions")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 활성 세션 수: {result['total_count']}")
                    
                    for session in result['active_sessions']:
                        print(f"  📝 세션 {session['session_id']}: {session['user_request']['message'][:50]}...")
                    
                    return True
                else:
                    print(f"❌ 조회 실패: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ 조회 오류: {str(e)}")
            return False

    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 멀티에이전트 오케스트레이션 테스트 시작\n")
        
        # 1. 코드 리뷰 워크플로우 테스트
        approval_session = await self.test_code_review_workflow()
        
        # 2. 문서 검색 워크플로우 테스트
        await self.test_document_search_workflow()
        
        # 3. 일정 관리 워크플로우 테스트
        schedule_session = await self.test_schedule_workflow()
        
        # 4. 활성 세션 조회
        await self.test_active_sessions()
        
        # 5. 승인 워크플로우 테스트 (승인 필요한 세션이 있다면)
        if approval_session:
            await self.test_approval_workflow(approval_session)
        if schedule_session:
            await self.test_approval_workflow(schedule_session)
        
        print("\n🏁 테스트 완료!")


async def main():
    tester = MultiAgentTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
