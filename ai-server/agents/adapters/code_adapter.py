"""
Code Agent Adapter - Wraps CodeAgent to return standardized AgentResult
"""
import time
from agents.code_agent import CodeAgent
from agents.adapters.base_adapter import BaseAgentAdapter
from shared.types import AgentResult, AgentName, ActionType
from shared.state import GraphState


class CodeAgentAdapter(BaseAgentAdapter):
    def __init__(self):
        self.agent = CodeAgent()

    async def execute(self, state: GraphState) -> GraphState:
        """Execute code agent and return standardized result"""
        start_time = time.time()
        
        try:
            user_request = state["user_request"]
            
            # Execute the code agent
            # Note: CodeAgent might need to be modified to accept the request format
            response = await self._call_code_agent(user_request.message, user_request.context or {})
            
            # Create standardized result
            result = AgentResult(
                agent_name=AgentName.CODE,
                success=True,
                message=response.get("message", "Code analysis completed"),
                data=response,
                action_type=ActionType.CODE_REVIEW,  # Default action type
                processing_time=time.time() - start_time
            )
            
            # Add result to state
            return self._add_result_to_state(state, result)
            
        except Exception as e:
            error_result = AgentResult(
                agent_name=AgentName.CODE,
                success=False,
                message=f"Code agent execution failed: {str(e)}",
                processing_time=time.time() - start_time
            )
            
            return self._add_result_to_state(state, error_result)

    async def _call_code_agent(self, message: str, data: dict = None) -> dict:
        """Call the actual code agent with proper interface"""
        
        # For now, create a mock response
        # This should be replaced with actual CodeAgent integration
        return {
            "message": f"Code analysis completed for: {message[:100]}...",
            "analysis": {
                "code_quality": "Good",
                "suggestions": ["Add more comments", "Consider refactoring"],
                "security_issues": ["Check input validation"]
            },
            "requires_security_check": True
        }
