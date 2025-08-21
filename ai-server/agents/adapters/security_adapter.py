"""
Security Agent Adapter - Wraps SecurityAgent to return standardized AgentResult
"""
import time
from agents.security_agent import SecurityAgent
from agents.adapters.base_adapter import BaseAgentAdapter
from shared.types import AgentResult, AgentName, ActionType
from shared.state import GraphState


class SecurityAgentAdapter(BaseAgentAdapter):
    def __init__(self):
        self.agent = SecurityAgent()

    async def execute(self, state: GraphState) -> GraphState:
        """Execute security agent and return standardized result"""
        start_time = time.time()
        
        try:
            user_request = state["user_request"]
            
            # Get code from previous agent result if available
            code_to_analyze = user_request.message
            previous_results = state.get("agent_results", [])
            
            if previous_results:
                last_result = previous_results[-1]
                if last_result.agent_name == AgentName.CODE and last_result.data:
                    code_to_analyze = last_result.data.get("code", user_request.message)
            
            # Execute the security agent
            response = await self._call_security_agent(code_to_analyze, user_request.context or {})
            
            # Determine action type based on response
            action_type = ActionType.SECURITY_ANALYSIS
            if response.get("vulnerabilities_found"):
                action_type = ActionType.SECURITY_FIX
            
            # Create standardized result
            result = AgentResult(
                agent_name=AgentName.SECURITY,
                success=True,
                message=response.get("message", "Security analysis completed"),
                data=response,
                action_type=action_type,
                processing_time=time.time() - start_time
            )
            
            # Add result to state
            return self._add_result_to_state(state, result)
            
        except Exception as e:
            error_result = AgentResult(
                agent_name=AgentName.SECURITY,
                success=False,
                message=f"Security agent execution failed: {str(e)}",
                processing_time=time.time() - start_time
            )
            
            return self._add_result_to_state(state, error_result)

    async def _call_security_agent(self, code: str, data: dict = None) -> dict:
        """Call the actual security agent with proper interface"""
        
        # For now, create a mock response
        # This should be replaced with actual SecurityAgent integration
        vulnerabilities_found = "password" in code.lower() or "secret" in code.lower()
        
        return {
            "message": f"Security analysis completed for code ({len(code)} characters)",
            "vulnerabilities_found": vulnerabilities_found,
            "security_score": 85 if not vulnerabilities_found else 45,
            "issues": [
                {
                    "severity": "high" if vulnerabilities_found else "low",
                    "type": "hardcoded_credentials" if vulnerabilities_found else "info",
                    "description": "Potential hardcoded credentials detected" if vulnerabilities_found else "No major issues found"
                }
            ],
            "recommendations": [
                "Use environment variables for secrets",
                "Implement input validation",
                "Add security headers"
            ]
        }
