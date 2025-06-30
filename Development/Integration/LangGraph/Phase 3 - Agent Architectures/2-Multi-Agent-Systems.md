# Multi-Agent Systems

## Agent Communication

### Message Passing Between Agents

```python
from typing import TypedDict, List, Dict, Optional, Any, Literal
from langgraph.graph import StateGraph
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    message_type: Literal["request", "response", "notification", "data"]
    timestamp: str
    metadata: Dict[str, Any]

class MultiAgentState(TypedDict):
    current_agent: str
    messages: List[Message]
    agent_data: Dict[str, Dict]
    task_assignments: Dict[str, List[str]]
    coordination_status: str
    final_result: str

def agent_communicator(agent_name: str):
    """Factory for creating agent communication nodes."""
    def communicate(state: MultiAgentState) -> MultiAgentState:
        # Process incoming messages for this agent
        incoming_messages = [msg for msg in state["messages"] 
                           if msg.receiver == agent_name and msg.sender != agent_name]
        
        # Generate response based on agent role
        if agent_name == "coordinator":
            response = coordinate_agents(state, incoming_messages)
        elif agent_name == "researcher":
            response = research_task(state, incoming_messages)
        elif agent_name == "analyzer":
            response = analyze_data(state, incoming_messages)
        else:
            response = f"{agent_name} processed {len(incoming_messages)} messages"
        
        # Create outgoing message
        outgoing_message = Message(
            sender=agent_name,
            receiver="all" if agent_name == "coordinator" else "coordinator",
            content=response,
            message_type="response",
            timestamp=datetime.now().isoformat(),
            metadata={"processed_messages": len(incoming_messages)}
        )
        
        return {
            **state,
            "current_agent": agent_name,
            "messages": state["messages"] + [outgoing_message],
            "agent_data": {
                **state["agent_data"],
                agent_name: {"last_action": response, "status": "active"}
            }
        }
    
    return communicate

def coordinate_agents(state: MultiAgentState, messages: List[Message]) -> str:
    """Coordinator agent logic."""
    if not messages:
        return "Initiating task coordination. Assigning work to specialized agents."
    
    # Analyze responses from other agents
    responses = [msg.content for msg in messages]
    
    if all("completed" in response.lower() for response in responses):
        return "All agents have completed their tasks. Preparing final synthesis."
    else:
        return "Monitoring agent progress. Providing additional guidance as needed."

def research_task(state: MultiAgentState, messages: List[Message]) -> str:
    """Research agent logic."""
    if messages:
        coordinator_msg = messages[-1].content
        if "research" in coordinator_msg.lower():
            return "Research completed: Found relevant data on the assigned topic."
    
    return "Standing by for research assignments."

def analyze_data(state: MultiAgentState, messages: List[Message]) -> str:
    """Analyzer agent logic."""
    if messages:
        # Look for data from researcher
        data_messages = [msg for msg in messages if "data" in msg.content.lower()]
        if data_messages:
            return "Analysis completed: Data patterns identified and insights generated."
    
    return "Waiting for data to analyze."
```

### Shared State Management

```python
class SharedAgentState(TypedDict):
    global_context: Dict[str, Any]
    agent_states: Dict[str, Dict[str, Any]]
    shared_memory: List[Dict[str, Any]]
    coordination_locks: Dict[str, bool]
    resource_pool: Dict[str, Any]

def shared_state_manager(state: SharedAgentState, agent_id: str, updates: Dict) -> SharedAgentState:
    """Manage shared state across multiple agents."""
    
    # Update agent-specific state
    current_agent_state = state["agent_states"].get(agent_id, {})
    updated_agent_state = {**current_agent_state, **updates}
    
    # Update global context if agent provides global updates
    global_updates = updates.get("global_updates", {})
    updated_global_context = {**state["global_context"], **global_updates}
    
    # Add to shared memory if specified
    shared_memory = state["shared_memory"]
    if "memory_entry" in updates:
        shared_memory = shared_memory + [updates["memory_entry"]]
    
    return {
        **state,
        "global_context": updated_global_context,
        "agent_states": {
            **state["agent_states"],
            agent_id: updated_agent_state
        },
        "shared_memory": shared_memory
    }

def resource_coordinator_node(state: SharedAgentState) -> SharedAgentState:
    """Coordinate shared resources among agents."""
    
    # Check resource requests from agents
    resource_requests = {}
    for agent_id, agent_state in state["agent_states"].items():
        if "resource_request" in agent_state:
            resource_requests[agent_id] = agent_state["resource_request"]
    
    # Allocate resources based on priority and availability
    resource_allocations = {}
    available_resources = state["resource_pool"].copy()
    
    for agent_id, request in resource_requests.items():
        if request in available_resources and available_resources[request] > 0:
            resource_allocations[agent_id] = request
            available_resources[request] -= 1
    
    return {
        **state,
        "resource_pool": available_resources,
        "global_context": {
            **state["global_context"],
            "resource_allocations": resource_allocations
        }
    }
```

## Role-Based Architectures

### Specialized Agent Roles

```python
class SpecializedAgentState(TypedDict):
    task_description: str
    agent_assignments: Dict[str, str]
    research_data: List[Dict]
    analysis_results: Dict[str, Any]
    written_content: str
    review_feedback: List[str]
    current_phase: str

def research_agent_node(state: SpecializedAgentState) -> SpecializedAgentState:
    """Specialized research agent."""
    task = state["task_description"]
    
    # Simulate research process
    research_results = [
        {"source": "academic_db", "data": f"Academic research on: {task}"},
        {"source": "web_search", "data": f"Current information about: {task}"},
        {"source": "expert_interviews", "data": f"Expert opinions on: {task}"}
    ]
    
    return {
        **state,
        "research_data": state["research_data"] + research_results,
        "current_phase": "research_completed",
        "agent_assignments": {
            **state["agent_assignments"],
            "researcher": "completed"
        }
    }

def analysis_agent_node(state: SpecializedAgentState) -> SpecializedAgentState:
    """Specialized analysis agent."""
    research_data = state["research_data"]
    
    if not research_data:
        return {**state, "current_phase": "waiting_for_research"}
    
    # Analyze collected research
    analysis = {
        "data_sources": len(research_data),
        "key_themes": ["theme1", "theme2", "theme3"],
        "reliability_score": 0.85,
        "recommendations": ["rec1", "rec2", "rec3"]
    }
    
    return {
        **state,
        "analysis_results": analysis,
        "current_phase": "analysis_completed",
        "agent_assignments": {
            **state["agent_assignments"],
            "analyzer": "completed"
        }
    }

def writing_agent_node(state: SpecializedAgentState) -> SpecializedAgentState:
    """Specialized writing agent."""
    analysis = state["analysis_results"]
    research = state["research_data"]
    
    if not analysis or not research:
        return {**state, "current_phase": "waiting_for_input"}
    
    # Generate written content
    content = f"""
# Report on {state['task_description']}

## Research Summary
Based on {len(research)} sources, including {analysis.get('data_sources', 0)} data points.

## Key Findings
{', '.join(analysis.get('key_themes', []))}

## Analysis
Reliability Score: {analysis.get('reliability_score', 0)}

## Recommendations
{', '.join(analysis.get('recommendations', []))}
"""
    
    return {
        **state,
        "written_content": content,
        "current_phase": "writing_completed",
        "agent_assignments": {
            **state["agent_assignments"],
            "writer": "completed"
        }
    }

def review_agent_node(state: SpecializedAgentState) -> SpecializedAgentState:
    """Specialized review agent."""
    content = state["written_content"]
    
    if not content:
        return {**state, "current_phase": "waiting_for_content"}
    
    # Review and provide feedback
    feedback = [
        "Content structure is clear and logical",
        "Research citations could be more specific",
        "Analysis section needs more detail",
        "Overall quality is good, minor revisions needed"
    ]
    
    return {
        **state,
        "review_feedback": feedback,
        "current_phase": "review_completed",
        "agent_assignments": {
            **state["agent_assignments"],
            "reviewer": "completed"
        }
    }
```

### Hierarchical Agent Structures

```python
class HierarchicalState(TypedDict):
    project_goal: str
    supervisor_decisions: List[str]
    team_assignments: Dict[str, List[str]]
    team_reports: Dict[str, str]
    escalated_issues: List[str]
    project_status: str

def supervisor_agent_node(state: HierarchicalState) -> HierarchicalState:
    """High-level supervisor agent."""
    
    # Make strategic decisions
    decisions = [
        f"Project goal: {state['project_goal']}",
        "Assign Team A to handle research phase",
        "Assign Team B to handle implementation phase",
        "Set milestone checkpoints for progress tracking"
    ]
    
    # Create team assignments
    assignments = {
        "team_a": ["research", "data_collection", "initial_analysis"],
        "team_b": ["implementation", "testing", "deployment_prep"]
    }
    
    return {
        **state,
        "supervisor_decisions": decisions,
        "team_assignments": assignments,
        "project_status": "teams_assigned"
    }

def team_lead_agent_node(team_name: str):
    """Factory for team lead agents."""
    def team_lead_logic(state: HierarchicalState) -> HierarchicalState:
        assignments = state["team_assignments"].get(team_name, [])
        
        if not assignments:
            return state
        
        # Team lead processes assignments and reports back
        report = f"Team {team_name} Status:\n"
        for task in assignments:
            report += f"- {task}: In Progress\n"
        
        # Check for issues that need escalation
        escalated = []
        if len(assignments) > 3:
            escalated.append(f"Team {team_name}: Resource constraints identified")
        
        current_reports = state["team_reports"]
        current_reports[team_name] = report
        
        current_escalations = state["escalated_issues"] + escalated
        
        return {
            **state,
            "team_reports": current_reports,
            "escalated_issues": current_escalations
        }
    
    return team_lead_logic

def task_delegation_controller(state: HierarchicalState) -> str:
    """Control task delegation flow."""
    if not state["team_assignments"]:
        return "assign_teams"
    
    if len(state["team_reports"]) < len(state["team_assignments"]):
        return "execute_teams"
    
    if state["escalated_issues"]:
        return "handle_escalations"
    
    return "finalize_project"

# Build hierarchical workflow
def create_hierarchical_system():
    workflow = StateGraph(HierarchicalState)
    
    workflow.add_node("supervisor", supervisor_agent_node)
    workflow.add_node("team_a_lead", team_lead_agent_node("team_a"))
    workflow.add_node("team_b_lead", team_lead_agent_node("team_b"))
    workflow.add_node("escalation_handler", lambda s: {**s, "project_status": "escalations_resolved"})
    workflow.add_node("project_finalizer", lambda s: {**s, "project_status": "completed"})
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        task_delegation_controller,
        {
            "assign_teams": "team_a_lead",
            "execute_teams": "team_a_lead"
        }
    )
    
    workflow.add_edge("team_a_lead", "team_b_lead")
    
    workflow.add_conditional_edges(
        "team_b_lead",
        task_delegation_controller,
        {
            "handle_escalations": "escalation_handler",
            "finalize_project": "project_finalizer"
        }
    )
    
    workflow.add_edge("escalation_handler", "project_finalizer")
    workflow.add_edge("project_finalizer", "__end__")
    
    return workflow.compile()

# Usage example
def test_multi_agent_systems():
    # Test specialized agent system
    specialized_workflow = StateGraph(SpecializedAgentState)
    specialized_workflow.add_node("research", research_agent_node)
    specialized_workflow.add_node("analyze", analysis_agent_node)
    specialized_workflow.add_node("write", writing_agent_node)
    specialized_workflow.add_node("review", review_agent_node)
    
    specialized_workflow.set_entry_point("research")
    specialized_workflow.add_edge("research", "analyze")
    specialized_workflow.add_edge("analyze", "write")
    specialized_workflow.add_edge("write", "review")
    specialized_workflow.add_edge("review", "__end__")
    
    specialized_app = specialized_workflow.compile()
    
    result = specialized_app.invoke({
        "task_description": "AI impact on healthcare",
        "agent_assignments": {},
        "research_data": [],
        "analysis_results": {},
        "written_content": "",
        "review_feedback": [],
        "current_phase": "starting"
    })
    
    print("Multi-Agent System Result:")
    print(f"Final Phase: {result['current_phase']}")
    print(f"Content Length: {len(result['written_content'])}")
    print(f"Review Feedback: {len(result['review_feedback'])} items")

if __name__ == "__main__":
    test_multi_agent_systems()
```

These multi-agent patterns enable sophisticated coordination between specialized agents, allowing for complex task decomposition, parallel processing, and hierarchical management structures essential for enterprise-scale AI applications.
