"""
Common types for multi-agent system
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class AgentName(str, Enum):
    """Available agent names"""
    MANAGER = "manager"
    CODE = "code"
    SECURITY = "security"
    RAG = "rag"
    SCHEDULE = "schedule"


class ActionType(str, Enum):
    """Types of actions an agent can perform"""
    ANALYZE = "analyze"
    REVIEW = "review"
    SEARCH = "search"
    CREATE = "create"
    UPDATE = "update"
    APPROVE = "approve"
    REJECT = "reject"
    # Agent-specific actions
    CODE_REVIEW = "code_review"
    CODE_DEPLOYMENT = "code_deployment"
    SECURITY_ANALYSIS = "security_analysis"
    SECURITY_FIX = "security_fix"
    DATA_RETRIEVAL = "data_retrieval"
    DATA_MODIFICATION = "data_modification"
    SCHEDULE_MEETING = "schedule_meeting"


class AgentResult(BaseModel):
    """Standard result format for all agents"""
    agent_name: AgentName
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    action_type: Optional[ActionType] = None
    next_agent: Optional[AgentName] = None
    requires_approval: bool = False
    approval_data: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class UserRequest(BaseModel):
    """User request structure"""
    message: str
    user_id: Optional[str] = None
    channel_type: Optional[str] = "web"
    channel_id: Optional[str] = "web"
    context: Optional[Dict[str, Any]] = None


class ApprovalRequest(BaseModel):
    """Approval request structure"""
    agent_name: AgentName
    action_type: Optional[ActionType] = None
    data: Optional[Dict[str, Any]] = None
    message: str
    session_id: Optional[str] = None
    approved: Optional[bool] = None
    feedback: Optional[str] = None
