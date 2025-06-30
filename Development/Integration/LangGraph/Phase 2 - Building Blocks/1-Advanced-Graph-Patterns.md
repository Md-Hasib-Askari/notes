# Advanced Graph Patterns

## Cycles and Loops

### Implementing Iterative Processes

Cycles in LangGraph enable iterative workflows where nodes can loop back to previous steps, allowing for refinement and self-correction.

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph

class IterativeState(TypedDict):
    input: str
    attempts: List[str]
    current_result: str
    iteration_count: int
    max_iterations: int
    goal_achieved: bool

def processing_node(state: IterativeState) -> IterativeState:
    """Main processing logic that may need refinement."""
    attempt = f"Attempt {state['iteration_count']}: Processing {state['input']}"
    
    # Simulate processing quality improving with iterations
    quality_score = min(0.3 + (state['iteration_count'] * 0.2), 1.0)
    
    return {
        **state,
        "current_result": attempt,
        "attempts": state["attempts"] + [attempt],
        "iteration_count": state["iteration_count"] + 1,
        "goal_achieved": quality_score >= 0.8
    }

def evaluation_node(state: IterativeState) -> IterativeState:
    """Evaluate if we should continue iterating."""
    return state  # Pass through for routing decision

def should_continue_loop(state: IterativeState) -> str:
    """Loop termination logic."""
    if state["goal_achieved"]:
        return "finalize"
    elif state["iteration_count"] >= state["max_iterations"]:
        return "max_attempts"
    else:
        return "continue"

# Build iterative workflow
workflow = StateGraph(IterativeState)
workflow.add_node("process", processing_node)
workflow.add_node("evaluate", evaluation_node)
workflow.add_node("finalize", lambda s: {**s, "status": "completed"})

workflow.set_entry_point("process")
workflow.add_edge("process", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    should_continue_loop,
    {
        "continue": "process",  # Loop back
        "finalize": "finalize",
        "max_attempts": "finalize"
    }
)
workflow.add_edge("finalize", "__end__")
```

### Loop Termination Conditions

Critical patterns for preventing infinite loops:

```python
def safe_loop_controller(state: IterativeState) -> str:
    """Multiple termination conditions for safety."""
    # Primary success condition
    if state["goal_achieved"]:
        return "success"
    
    # Safety: maximum iterations
    if state["iteration_count"] >= state["max_iterations"]:
        return "timeout"
    
    # Safety: detect no progress
    if len(state["attempts"]) >= 3:
        recent_attempts = state["attempts"][-3:]
        if len(set(recent_attempts)) == 1:  # All identical
            return "stuck"
    
    # Safety: error accumulation
    error_count = sum(1 for attempt in state["attempts"] if "error" in attempt.lower())
    if error_count >= 2:
        return "too_many_errors"
    
    return "continue"
```

### Managing State in Cyclic Graphs

```python
class CyclicState(TypedDict):
    data: str
    history: List[dict]  # Track state changes
    cycle_count: int
    convergence_threshold: float
    
def track_convergence(state: CyclicState) -> CyclicState:
    """Monitor state changes for convergence detection."""
    current_snapshot = {"data": state["data"], "cycle": state["cycle_count"]}
    
    # Calculate change from previous iteration
    if state["history"]:
        prev_data = state["history"][-1]["data"]
        change_ratio = len(set(state["data"]) - set(prev_data)) / max(len(state["data"]), 1)
    else:
        change_ratio = 1.0
    
    return {
        **state,
        "history": state["history"] + [current_snapshot],
        "cycle_count": state["cycle_count"] + 1,
        "convergence_score": 1.0 - change_ratio
    }
```

## Parallel Processing

### Concurrent Node Execution

LangGraph supports parallel execution for independent operations:

```python
class ParallelState(TypedDict):
    input: str
    search_results: List[dict]
    analysis_results: List[dict]
    summary_results: List[dict]
    parallel_complete: bool

def web_search_node(state: ParallelState) -> ParallelState:
    """Simulated web search - can run in parallel."""
    import time
    time.sleep(1)  # Simulate API call
    
    results = [
        {"source": "web", "content": f"Web result for: {state['input']}"},
        {"source": "web", "content": f"Additional web data: {state['input']}"}
    ]
    
    return {**state, "search_results": results}

def database_query_node(state: ParallelState) -> ParallelState:
    """Simulated database query - can run in parallel."""
    import time
    time.sleep(0.5)  # Simulate DB query
    
    results = [
        {"source": "db", "content": f"Database entry: {state['input']}"},
        {"source": "db", "content": f"Related DB record: {state['input']}"}
    ]
    
    return {**state, "analysis_results": results}

def document_analysis_node(state: ParallelState) -> ParallelState:
    """Simulated document processing - can run in parallel."""
    import time
    time.sleep(0.8)  # Simulate processing
    
    results = [
        {"source": "docs", "content": f"Document analysis: {state['input']}"}
    ]
    
    return {**state, "summary_results": results}

# Parallel workflow setup
parallel_workflow = StateGraph(ParallelState)
parallel_workflow.add_node("web_search", web_search_node)
parallel_workflow.add_node("db_query", database_query_node)
parallel_workflow.add_node("doc_analysis", document_analysis_node)
parallel_workflow.add_node("merge_results", merge_parallel_results)

# All three nodes can execute in parallel
parallel_workflow.set_entry_point("web_search")
parallel_workflow.set_entry_point("db_query")
parallel_workflow.set_entry_point("doc_analysis")

# Converge to merge node
parallel_workflow.add_edge("web_search", "merge_results")
parallel_workflow.add_edge("db_query", "merge_results")
parallel_workflow.add_edge("doc_analysis", "merge_results")
```

### Merging Parallel Results

```python
def merge_parallel_results(state: ParallelState) -> ParallelState:
    """Combine results from parallel operations."""
    all_results = []
    
    # Combine all parallel results
    if state.get("search_results"):
        all_results.extend(state["search_results"])
    if state.get("analysis_results"):
        all_results.extend(state["analysis_results"])
    if state.get("summary_results"):
        all_results.extend(state["summary_results"])
    
    # Create unified summary
    sources = set(result["source"] for result in all_results)
    content_summary = f"Merged {len(all_results)} results from {len(sources)} sources"
    
    return {
        **state,
        "merged_results": all_results,
        "summary": content_summary,
        "parallel_complete": True
    }
```

### Error Handling in Parallel Flows

```python
class RobustParallelState(TypedDict):
    input: str
    results: dict
    errors: dict
    success_count: int
    total_tasks: int

def safe_parallel_node(node_name: str, operation_func):
    """Wrapper for safe parallel execution."""
    def wrapped_node(state: RobustParallelState) -> RobustParallelState:
        try:
            result = operation_func(state["input"])
            
            # Update success results
            new_results = state.get("results", {})
            new_results[node_name] = result
            
            return {
                **state,
                "results": new_results,
                "success_count": state.get("success_count", 0) + 1
            }
            
        except Exception as e:
            # Track errors without failing entire workflow
            new_errors = state.get("errors", {})
            new_errors[node_name] = str(e)
            
            return {
                **state,
                "errors": new_errors
            }
    
    return wrapped_node

# Create safe parallel nodes
safe_search = safe_parallel_node("search", lambda x: f"Search: {x}")
safe_analysis = safe_parallel_node("analysis", lambda x: f"Analysis: {x}")

def parallel_completion_check(state: RobustParallelState) -> str:
    """Check if enough parallel tasks completed successfully."""
    total_completed = state.get("success_count", 0) + len(state.get("errors", {}))
    
    if total_completed >= state["total_tasks"]:
        # At least 50% success required
        if state.get("success_count", 0) >= state["total_tasks"] * 0.5:
            return "merge_results"
        else:
            return "handle_failures"
    
    return "wait_for_completion"
```

## Best Practices for Advanced Patterns

### Cycle Management
- Always include maximum iteration limits
- Implement convergence detection
- Track state history for debugging
- Use timeout mechanisms for safety

### Parallel Processing
- Ensure nodes are truly independent
- Handle partial failures gracefully
- Use appropriate merge strategies
- Monitor resource usage

### Performance Considerations
```python
# Efficient state updates for cycles
def efficient_cycle_update(state: dict, new_data: dict) -> dict:
    """Minimize state copying in cycles."""
    # Only update changed fields
    updated_fields = {k: v for k, v in new_data.items() if k not in state or state[k] != v}
    
    if not updated_fields:
        return state  # No changes needed
    
    return {**state, **updated_fields}

# Memory management for long cycles
def cleanup_history(state: dict, max_history: int = 10) -> dict:
    """Prevent unbounded memory growth in cycles."""
    if "history" in state and len(state["history"]) > max_history:
        # Keep only recent history
        trimmed_history = state["history"][-max_history:]
        return {**state, "history": trimmed_history}
    return state
```

This foundation in advanced graph patterns prepares you for building sophisticated multi-agent systems and complex workflow orchestration in the next phases.
