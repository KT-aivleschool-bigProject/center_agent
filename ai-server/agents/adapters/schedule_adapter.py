"""
Schedule Agent Adapter - Wraps ScheduleAgent to return standardized AgentResult
"""
import time
from agents.schedule.schedule_agent import ScheduleAgent
from agents.adapters.base_adapter import BaseAgentAdapter
from shared.types import AgentResult, AgentName, ActionType
from shared.state import GraphState


class ScheduleAgentAdapter(BaseAgentAdapter):
    def __init__(self):
        self.agent = ScheduleAgent(channel="web")  # 웹 채널로 초기화

    async def execute(self, state: GraphState) -> GraphState:
        """Execute schedule agent and return standardized result"""
        start_time = time.time()
        
        try:
            user_request = state["user_request"]
            
            # Execute the schedule agent
            response = await self._call_schedule_agent(user_request.message, user_request.context or {})
            
            # Determine action type based on response
            action_type = ActionType.SCHEDULE_MEETING
            if "일정" in user_request.message or "meeting" in user_request.message.lower():
                action_type = ActionType.SCHEDULE_MEETING
            
            # Create standardized result
            result = AgentResult(
                agent_name=AgentName.SCHEDULE,
                success=True,
                message=response.get("message", "Schedule operation completed"),
                data=response,
                action_type=action_type,
                processing_time=time.time() - start_time
            )
            
            # Add result to state
            return self._add_result_to_state(state, result)
            
        except Exception as e:
            error_result = AgentResult(
                agent_name=AgentName.SCHEDULE,
                success=False,
                message=f"Schedule agent execution failed: {str(e)}",
                processing_time=time.time() - start_time
            )
            
            return self._add_result_to_state(state, error_result)

    async def _call_schedule_agent(self, message: str, data: dict = None) -> dict:
        """Call the actual schedule agent with proper interface"""
        
        # For now, create a mock response
        # This should be replaced with actual ScheduleAgent integration
        return {
            "message": f"Schedule analysis completed for: {message[:100]}...",
            "meeting_suggestions": [
                {
                    "title": "Project Review Meeting",
                    "date": "2024-12-20",
                    "time": "14:00",
                    "duration": "1 hour",
                    "attendees": ["team@company.com"]
                }
            ],
            "available_slots": [
                "2024-12-20 14:00-15:00",
                "2024-12-20 16:00-17:00",
                "2024-12-21 10:00-11:00"
            ],
            "calendar_conflicts": []
        }
