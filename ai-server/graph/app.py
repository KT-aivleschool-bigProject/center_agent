"""
Multi-Agent Orchestration Graph using LangGraph
"""
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from shared.state import GraphState
from shared.types import AgentName, ActionType
from agents.manager import ManagerAgent
from agents.adapters.code_adapter import CodeAgentAdapter
from agents.adapters.security_adapter import SecurityAgentAdapter
from agents.adapters.rag_adapter import RagAgentAdapter
from agents.adapters.schedule_adapter import ScheduleAgentAdapter


class MultiAgentOrchestrator:
    def __init__(self):
        self.manager = ManagerAgent()
        self.agents = {
            AgentName.CODE: CodeAgentAdapter(),
            AgentName.SECURITY: SecurityAgentAdapter(),
            AgentName.RAG: RagAgentAdapter(),
            AgentName.SCHEDULE: ScheduleAgentAdapter(),
        }
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for agent orchestration"""
        
        # Create the state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("manager", self._manager_node)
        workflow.add_node("code_agent", self._code_agent_node)
        workflow.add_node("security_agent", self._security_agent_node)
        workflow.add_node("rag_agent", self._rag_agent_node)
        workflow.add_node("schedule_agent", self._schedule_agent_node)
        workflow.add_node("approval_check", self._approval_check_node)
        
        # Set entry point
        workflow.set_entry_point("manager")
        
        # Add conditional edges from manager
        workflow.add_conditional_edges(
            "manager",
            self._route_to_agent,
            {
                "code_agent": "code_agent",
                "security_agent": "security_agent",
                "rag_agent": "rag_agent",
                "schedule_agent": "schedule_agent",
                "end": END
            }
        )
        
        # Add edges from agents to approval check or end
        workflow.add_conditional_edges(
            "code_agent",
            self._check_need_approval,
            {
                "approval": "approval_check",
                "security": "security_agent",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "security_agent",
            self._check_need_approval,
            {
                "approval": "approval_check",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "rag_agent",
            self._check_need_approval,
            {
                "approval": "approval_check",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "schedule_agent",
            self._check_need_approval,
            {
                "approval": "approval_check",
                "end": END
            }
        )
        
        # Approval check edges
        workflow.add_conditional_edges(
            "approval_check",
            self._handle_approval,
            {
                "approved": END,
                "rejected": END,
                "pending": END
            }
        )
        
        return workflow.compile()

    async def _manager_node(self, state: GraphState) -> GraphState:
        """Manager node - analyze request and route"""
        return await self.manager.analyze_request(state)

    async def _code_agent_node(self, state: GraphState) -> GraphState:
        """Code agent node"""
        return await self.agents[AgentName.CODE].execute(state)

    async def _security_agent_node(self, state: GraphState) -> GraphState:
        """Security agent node"""
        return await self.agents[AgentName.SECURITY].execute(state)

    async def _rag_agent_node(self, state: GraphState) -> GraphState:
        """RAG agent node"""
        return await self.agents[AgentName.RAG].execute(state)

    async def _schedule_agent_node(self, state: GraphState) -> GraphState:
        """Schedule agent node"""
        return await self.agents[AgentName.SCHEDULE].execute(state)

    async def _approval_check_node(self, state: GraphState) -> GraphState:
        """Approval check node"""
        # Get the last agent result
        if not state["agent_results"]:
            return state
            
        last_result = state["agent_results"][-1]
        
        # Check if approval is needed based on agent type and action
        if self._requires_approval(last_result):
            # Set approval status to pending
            state["approval_status"] = "pending"
            state["approval_pending"] = True
            
            # Create approval request
            from shared.types import ApprovalRequest
            approval_request = ApprovalRequest(
                agent_name=last_result.agent_name,
                action_type=last_result.action_type,
                data=last_result.data,
                message=f"Approval needed for {last_result.agent_name.value} action: {last_result.message}",
                session_id=state.get("session_id")
            )
            state["approval_request"] = approval_request
        else:
            state["approval_status"] = "not_required"
            
        return state

    def _route_to_agent(self, state: GraphState) -> Literal["code_agent", "security_agent", "rag_agent", "schedule_agent", "end"]:
        """Route to the appropriate agent based on manager analysis"""
        if "error" in state:
            return "end"
            
        next_agent = state.get("next_agent")
        if not next_agent:
            return "end"
            
        agent_routing = {
            AgentName.CODE: "code_agent",
            AgentName.SECURITY: "security_agent",
            AgentName.RAG: "rag_agent",
            AgentName.SCHEDULE: "schedule_agent"
        }
        
        return agent_routing.get(next_agent, "end")

    def _check_need_approval(self, state: GraphState) -> Literal["approval", "security", "end"]:
        """Check if approval is needed or if there's a next agent in the chain"""
        if "error" in state:
            return "end"
            
        # Check if there are more agents in the chain
        current_agent = state.get("current_agent")
        if current_agent == AgentName.CODE:
            # Code review should go to security next
            return "security"
            
        # Check if approval is needed
        if state["agent_results"] and self._requires_approval(state["agent_results"][-1]):
            return "approval"
            
        return "end"

    def _handle_approval(self, state: GraphState) -> Literal["approved", "rejected", "pending"]:
        """Handle approval status"""
        approval_status = state.get("approval_status", "pending")
        return approval_status

    def _requires_approval(self, result) -> bool:
        """Check if an agent result requires approval"""
        # Actions that require approval
        approval_actions = [
            ActionType.CODE_DEPLOYMENT,
            ActionType.SECURITY_FIX,
            ActionType.SCHEDULE_MEETING,
            ActionType.DATA_MODIFICATION
        ]
        
        return result.action_type in approval_actions

    async def run(self, state: GraphState) -> GraphState:
        """Run the multi-agent workflow"""
        return await self.graph.ainvoke(state)


# Global orchestrator instance
orchestrator = MultiAgentOrchestrator()
