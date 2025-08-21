"""
Graph state for multi-agent orchestration
"""
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict
from shared.types import AgentName, AgentResult, UserRequest


class GraphState(TypedDict):
    """State shared across all agents in the graph"""
    # User input
    user_request: UserRequest
    
    # Current processing
    current_agent: Optional[AgentName]
    session_id: str
    
    # Agent results
    agent_results: List[AgentResult]
    final_result: Optional[AgentResult]
    
    # Flow control
    next_agent: Optional[AgentName]
    requires_approval: bool
    approval_pending: bool
    
    # Context and metadata
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    
    # Error handling
    error: Optional[str]
    retry_count: int
