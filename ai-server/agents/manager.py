"""
Manager Agent - Routes user requests to appropriate agents
"""
import json
import time
from typing import Dict, Any, Optional
from shared.types import AgentName, AgentResult, ActionType
from shared.state import GraphState
import openai
import os


class ManagerAgent:
    def __init__(self):
        self.name = "Manager Agent"
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        self.routing_rules = {
            # Code-related keywords
            "code": [AgentName.CODE, AgentName.SECURITY],  # Code review → Security check
            "pr": [AgentName.CODE, AgentName.SECURITY],
            "pull request": [AgentName.CODE, AgentName.SECURITY],
            "review": [AgentName.CODE],
            "코드": [AgentName.CODE, AgentName.SECURITY],
            "리뷰": [AgentName.CODE],
            
            # Security-related keywords
            "security": [AgentName.SECURITY],
            "vulnerability": [AgentName.SECURITY],
            "보안": [AgentName.SECURITY],
            "취약점": [AgentName.SECURITY],
            
            # Document/RAG-related keywords
            "document": [AgentName.RAG],
            "search": [AgentName.RAG],
            "find": [AgentName.RAG],
            "프로젝트": [AgentName.RAG],
            "문서": [AgentName.RAG],
            "검색": [AgentName.RAG],
            "알려줘": [AgentName.RAG],
            
            # Schedule-related keywords
            "schedule": [AgentName.SCHEDULE],
            "meeting": [AgentName.SCHEDULE],
            "calendar": [AgentName.SCHEDULE],
            "일정": [AgentName.SCHEDULE],
            "미팅": [AgentName.SCHEDULE],
            "회의": [AgentName.SCHEDULE],
        }

    async def analyze_request(self, state: GraphState) -> GraphState:
        """Analyze user request and determine routing"""
        start_time = time.time()
        
        try:
            user_message = state["user_request"].message.lower()
            
            # Try LLM-based analysis if API key is available
            if self.api_key:
                next_agent = await self._llm_analysis(user_message)
            else:
                next_agent = self._keyword_analysis(user_message)
            
            # Create manager result
            result = AgentResult(
                agent_name=AgentName.MANAGER,
                success=True,
                message=f"Request analyzed. Routing to {next_agent.value} agent.",
                data={
                    "analyzed_intent": next_agent.value,
                    "confidence": 0.8
                },
                next_agent=next_agent,
                processing_time=time.time() - start_time
            )
            
            # Update state
            state["current_agent"] = AgentName.MANAGER
            state["next_agent"] = next_agent
            state["agent_results"].append(result)
            
            return state
            
        except Exception as e:
            error_result = AgentResult(
                agent_name=AgentName.MANAGER,
                success=False,
                message=f"Manager analysis failed: {str(e)}",
                processing_time=time.time() - start_time
            )
            
            state["agent_results"].append(error_result)
            state["error"] = str(e)
            return state

    async def _llm_analysis(self, message: str) -> AgentName:
        """Use LLM to analyze user intent"""
        system_prompt = f"""
You are a request router for a multi-agent system. Analyze the user's request and determine which agent should handle it.

Available agents:
- {AgentName.CODE.value}: Code review, PR analysis, development tasks
- {AgentName.SECURITY.value}: Security vulnerability analysis, code security
- {AgentName.RAG.value}: Document search, project information, Q&A
- {AgentName.SCHEDULE.value}: Meeting scheduling, calendar management, task planning

Respond with only the agent name: {', '.join([agent.value for agent in AgentName if agent != AgentName.MANAGER])}

User request: {message}
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            agent_name = response.choices[0].message.content.strip().lower()
            
            # Validate and return
            for agent in AgentName:
                if agent.value == agent_name and agent != AgentName.MANAGER:
                    return agent
                    
            # Fallback to keyword analysis
            return self._keyword_analysis(message)
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._keyword_analysis(message)

    def _keyword_analysis(self, message: str) -> AgentName:
        """Fallback keyword-based routing"""
        message = message.lower()
        
        # Check for keyword matches
        for keyword, agents in self.routing_rules.items():
            if keyword in message:
                return agents[0]  # Return first agent in the chain
        
        # Default to RAG for general questions
        return AgentName.RAG

    def get_agent_chain(self, initial_agent: AgentName) -> list[AgentName]:
        """Get the chain of agents for a given initial agent"""
        chains = {
            AgentName.CODE: [AgentName.CODE, AgentName.SECURITY],  # Code → Security
            AgentName.SECURITY: [AgentName.SECURITY],
            AgentName.RAG: [AgentName.RAG],
            AgentName.SCHEDULE: [AgentName.SCHEDULE]
        }
        
        return chains.get(initial_agent, [initial_agent])
