"""
Base adapter class for standardizing agent interfaces
"""
from abc import ABC, abstractmethod
from shared.types import AgentResult
from shared.state import GraphState


class BaseAgentAdapter(ABC):
    """Base class for agent adapters"""
    
    @abstractmethod
    async def execute(self, state: GraphState) -> GraphState:
        """Execute the agent and return standardized results"""
        pass
    
    def _add_result_to_state(self, state: GraphState, result: AgentResult) -> GraphState:
        """Helper method to add agent result to state"""
        state["agent_results"].append(result)
        state["current_agent"] = result.agent_name
        
        if not result.success:
            state["error"] = result.message
            
        return state
