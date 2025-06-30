# Phase 2 Projects

## Project 1: Multi-Step Reasoning Agent

### Overview
Build an agent that breaks down complex problems into manageable steps, implements reflection and self-correction loops, and iteratively improves its reasoning.

### Implementation

```python
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph

class ReasoningState(TypedDict):
    original_problem: str
    problem_breakdown: List[str]
    current_step: int
    step_solutions: List[Dict[str, str]]
    reflection_notes: List[str]
    final_solution: str
    confidence_score: float
    iteration_count: int
    max_iterations: int

def problem_decomposer(state: ReasoningState) -> ReasoningState:
    """Break down complex problem into steps."""
    problem = state["original_problem"]
    
    # Simple decomposition logic (enhance with LLM)
    steps = []
    if "calculate" in problem.lower():
        steps = ["Identify variables", "Apply formula", "Compute result", "Verify answer"]
    elif "analyze" in problem.lower():
        steps = ["Gather data", "Identify patterns", "Draw conclusions", "Validate findings"]
    else:
        steps = ["Understand problem", "Research context", "Generate solution", "Review result"]
    
    return {
        **state,
        "problem_breakdown": steps,
        "current_step": 0,
        "step_solutions": [],
        "confidence_score": 0.3
    }

def step_solver(state: ReasoningState) -> ReasoningState:
    """Solve current step."""
    if state["current_step"] >= len(state["problem_breakdown"]):
        return state
    
    current_step_desc = state["problem_breakdown"][state["current_step"]]
    
    # Simulate step solving (replace with actual LLM reasoning)
    solution = {
        "step_description": current_step_desc,
        "solution": f"Solved: {current_step_desc}",
        "reasoning": f"Applied methodology for {current_step_desc}",
        "confidence": 0.7 + (state["current_step"] * 0.1)
    }
    
    return {
        **state,
        "step_solutions": state["step_solutions"] + [solution],
        "current_step": state["current_step"] + 1
    }

def reflector(state: ReasoningState) -> ReasoningState:
    """Reflect on solution quality and identify improvements."""
    if not state["step_solutions"]:
        return state
    
    # Analyze solution quality
    avg_confidence = sum(s["confidence"] for s in state["step_solutions"]) / len(state["step_solutions"])
    
    reflection_notes = []
    if avg_confidence < 0.6:
        reflection_notes.append("Low confidence detected - need more thorough analysis")
    if len(state["step_solutions"]) < len(state["problem_breakdown"]):
        reflection_notes.append("Incomplete solution - missing steps")
    
    # Check for contradictions or gaps
    solution_texts = [s["solution"] for s in state["step_solutions"]]
    if len(set(solution_texts)) != len(solution_texts):
        reflection_notes.append("Potential contradictions detected")
    
    return {
        **state,
        "reflection_notes": state["reflection_notes"] + reflection_notes,
        "confidence_score": avg_confidence,
        "iteration_count": state["iteration_count"] + 1
    }

def solution_integrator(state: ReasoningState) -> ReasoningState:
    """Integrate step solutions into final answer."""
    if not state["step_solutions"]:
        return state
    
    integrated_solution = "Final Solution:\n"
    for i, step_sol in enumerate(state["step_solutions"]):
        integrated_solution += f"{i+1}. {step_sol['step_description']}: {step_sol['solution']}\n"
    
    integrated_solution += f"\nConfidence: {state['confidence_score']:.2f}"
    integrated_solution += f"\nIterations: {state['iteration_count']}"
    
    return {
        **state,
        "final_solution": integrated_solution
    }

def reasoning_controller(state: ReasoningState) -> str:
    """Control reasoning loop flow."""
    # Check if all steps completed
    if state["current_step"] >= len(state["problem_breakdown"]):
        # Check if reflection suggests improvement needed
        if (state["confidence_score"] < 0.7 and 
            state["iteration_count"] < state["max_iterations"]):
            return "improve"
        else:
            return "integrate"
    else:
        return "solve_step"

# Build reasoning workflow
def create_reasoning_agent():
    workflow = StateGraph(ReasoningState)
    
    workflow.add_node("decompose", problem_decomposer)
    workflow.add_node("solve_step", step_solver)
    workflow.add_node("reflect", reflector)
    workflow.add_node("integrate", solution_integrator)
    workflow.add_node("improve", lambda s: {**s, "current_step": 0, "step_solutions": []})
    
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "solve_step")
    
    workflow.add_conditional_edges(
        "solve_step",
        reasoning_controller,
        {
            "solve_step": "solve_step",
            "improve": "improve", 
            "integrate": "integrate"
        }
    )
    
    workflow.add_edge("solve_step", "reflect")
    workflow.add_edge("improve", "solve_step")
    workflow.add_edge("integrate", "__end__")
    
    return workflow.compile()
```

## Project 2: Parallel Research Assistant

### Overview
Create concurrent information gathering from multiple sources and synthesize results into comprehensive reports.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ResearchState(TypedDict):
    research_query: str
    search_tasks: List[str]
    web_results: List[Dict]
    academic_results: List[Dict]
    news_results: List[Dict]
    social_results: List[Dict]
    synthesis_report: str
    task_completion: Dict[str, bool]

def research_planner(state: ResearchState) -> ResearchState:
    """Plan parallel research tasks."""
    query = state["research_query"]
    
    # Define parallel search strategies
    tasks = [
        f"Web search: {query}",
        f"Academic papers: {query}",
        f"Recent news: {query}",
        f"Social media: {query}"
    ]
    
    return {
        **state,
        "search_tasks": tasks,
        "task_completion": {task: False for task in tasks}
    }

def web_researcher(state: ResearchState) -> ResearchState:
    """Simulate web research (parallel execution)."""
    import time
    time.sleep(1)  # Simulate API delay
    
    results = [
        {"source": "website1", "content": f"Web info about {state['research_query']}"},
        {"source": "website2", "content": f"Additional web data on {state['research_query']}"}
    ]
    
    completion = state["task_completion"].copy()
    completion["Web search: " + state["research_query"]] = True
    
    return {
        **state,
        "web_results": results,
        "task_completion": completion
    }

def academic_researcher(state: ResearchState) -> ResearchState:
    """Simulate academic research (parallel execution)."""
    import time
    time.sleep(0.8)  # Simulate database query
    
    results = [
        {"source": "journal1", "content": f"Academic study on {state['research_query']}"},
        {"source": "conference1", "content": f"Research findings: {state['research_query']}"}
    ]
    
    completion = state["task_completion"].copy()
    completion["Academic papers: " + state["research_query"]] = True
    
    return {
        **state,
        "academic_results": results,
        "task_completion": completion
    }

def news_researcher(state: ResearchState) -> ResearchState:
    """Simulate news research (parallel execution)."""
    import time
    time.sleep(0.6)
    
    results = [
        {"source": "news1", "content": f"Recent news about {state['research_query']}"}
    ]
    
    completion = state["task_completion"].copy()
    completion["Recent news: " + state["research_query"]] = True
    
    return {
        **state,
        "news_results": results,
        "task_completion": completion
    }

def social_researcher(state: ResearchState) -> ResearchState:
    """Simulate social media research (parallel execution)."""
    import time
    time.sleep(0.5)
    
    results = [
        {"source": "twitter", "content": f"Social discussion: {state['research_query']}"}
    ]
    
    completion = state["task_completion"].copy()
    completion["Social media: " + state["research_query"]] = True
    
    return {
        **state,
        "social_results": results,
        "task_completion": completion
    }

def result_synthesizer(state: ResearchState) -> ResearchState:
    """Synthesize all parallel research results."""
    all_results = []
    
    # Collect all research findings
    for results_list, source_type in [
        (state.get("web_results", []), "Web"),
        (state.get("academic_results", []), "Academic"),
        (state.get("news_results", []), "News"),
        (state.get("social_results", []), "Social")
    ]:
        for result in results_list:
            all_results.append({
                **result,
                "type": source_type
            })
    
    # Create synthesis report
    synthesis = f"Research Report: {state['research_query']}\n\n"
    synthesis += f"Total sources analyzed: {len(all_results)}\n\n"
    
    # Organize by source type
    by_type = {}
    for result in all_results:
        result_type = result["type"]
        if result_type not in by_type:
            by_type[result_type] = []
        by_type[result_type].append(result)
    
    for source_type, results in by_type.items():
        synthesis += f"{source_type} Sources ({len(results)}):\n"
        for result in results:
            synthesis += f"  - {result['source']}: {result['content']}\n"
        synthesis += "\n"
    
    # Summary insights
    synthesis += "Key Insights:\n"
    synthesis += f"- Found information from {len(by_type)} different source types\n"
    synthesis += f"- {len(all_results)} total data points collected\n"
    synthesis += "- Research completed successfully\n"
    
    return {
        **state,
        "synthesis_report": synthesis
    }

def check_research_completion(state: ResearchState) -> str:
    """Check if all parallel research tasks completed."""
    completion_status = state.get("task_completion", {})
    
    if all(completion_status.values()):
        return "synthesize"
    else:
        return "wait"  # In practice, this would be handled by parallel execution

# Build parallel research workflow
def create_research_assistant():
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("plan", research_planner)
    workflow.add_node("web_search", web_researcher)
    workflow.add_node("academic_search", academic_researcher) 
    workflow.add_node("news_search", news_researcher)
    workflow.add_node("social_search", social_researcher)
    workflow.add_node("synthesize", result_synthesizer)
    
    workflow.set_entry_point("plan")
    
    # Parallel execution after planning
    workflow.add_edge("plan", "web_search")
    workflow.add_edge("plan", "academic_search")
    workflow.add_edge("plan", "news_search")
    workflow.add_edge("plan", "social_search")
    
    # All converge to synthesis
    workflow.add_edge("web_search", "synthesize")
    workflow.add_edge("academic_search", "synthesize")
    workflow.add_edge("news_search", "synthesize")
    workflow.add_edge("social_search", "synthesize")
    
    workflow.add_edge("synthesize", "__end__")
    
    return workflow.compile()

# Test both projects
def test_projects():
    # Test reasoning agent
    reasoning_app = create_reasoning_agent()
    reasoning_result = reasoning_app.invoke({
        "original_problem": "Calculate the compound interest on $1000 at 5% for 3 years",
        "problem_breakdown": [],
        "current_step": 0,
        "step_solutions": [],
        "reflection_notes": [],
        "final_solution": "",
        "confidence_score": 0.0,
        "iteration_count": 0,
        "max_iterations": 3
    })
    
    print("Reasoning Agent Result:")
    print(reasoning_result["final_solution"])
    
    # Test research assistant
    research_app = create_research_assistant()
    research_result = research_app.invoke({
        "research_query": "artificial intelligence in healthcare",
        "search_tasks": [],
        "web_results": [],
        "academic_results": [],
        "news_results": [],
        "social_results": [],
        "synthesis_report": "",
        "task_completion": {}
    })
    
    print("\n" + "="*50)
    print("Research Assistant Result:")
    print(research_result["synthesis_report"])

if __name__ == "__main__":
    test_projects()
```

These projects demonstrate advanced LangGraph concepts including iterative reasoning loops, reflection mechanisms, parallel processing, and result synthesis - essential patterns for building sophisticated AI agents.
