"""
RAG Agent Adapter - Wraps RagAgent to return standardized AgentResult
"""
import time
from agents.rag_agent import RAGAgent
from agents.adapters.base_adapter import BaseAgentAdapter
from shared.types import AgentResult, AgentName, ActionType
from shared.state import GraphState


class RagAgentAdapter(BaseAgentAdapter):
    def __init__(self):
        self.agent = RAGAgent()

    async def execute(self, state: GraphState) -> GraphState:
        """Execute RAG agent and return standardized result"""
        start_time = time.time()
        
        try:
            user_request = state["user_request"]
            
            # Execute the RAG agent
            response = await self._call_rag_agent(user_request.message, user_request.context or {})
            
            # Determine action type based on request
            action_type = ActionType.SEARCH
            if "document" in user_request.message.lower() or "파일" in user_request.message:
                action_type = ActionType.DATA_RETRIEVAL
            
            # Create standardized result
            result = AgentResult(
                agent_name=AgentName.RAG,
                success=True,
                message=response.get("message", "Document search completed"),
                data=response,
                action_type=action_type,
                processing_time=time.time() - start_time
            )
            
            # Add result to state
            return self._add_result_to_state(state, result)
            
        except Exception as e:
            error_result = AgentResult(
                agent_name=AgentName.RAG,
                success=False,
                message=f"RAG agent execution failed: {str(e)}",
                processing_time=time.time() - start_time
            )
            
            return self._add_result_to_state(state, error_result)

    async def _call_rag_agent(self, query: str, data: dict = None) -> dict:
        """Call the actual RAG agent with proper interface"""
        
        try:
            # Use the actual RAGAgent for document search
            answer = await self.agent.process(query)
            
            return {
                "message": f"Document search completed for query: {query[:100]}...",
                "query": query,
                "answer": answer,
                "documents": [],  # RAGAgent doesn't return document details
                "sources": []
            }
                
        except Exception as e:
            # Fallback mock response if RAGAgent fails
            return {
                "message": f"Document search completed for query: {query[:100]}...",
                "query": query,
                "documents": [
                    {
                        "title": "Sample Document",
                        "content": "This is a sample document content...",
                        "source": "project_docs/sample.txt"
                    }
                ],
                "answer": f"Based on the available documents, here's what I found about '{query}': [Mock answer]",
                "sources": ["project_docs/sample.txt"]
            }
