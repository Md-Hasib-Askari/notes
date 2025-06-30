# StateGraph Fundamentals

## Introduction to StateGraph

StateGraph is the core building block of LangGraph applications. It provides a structured way to define workflows using nodes, edges, and shared state. Understanding StateGraph is essential for building any LangGraph application.

## Core Components Overview

### 1. Nodes (Functions that Process State)
### 2. Edges (Connections Between Nodes)
### 3. State (Shared Data Structure)
### 4. Conditional Edges and Routing

---

## State: The Foundation

### What is State?

State is the shared data structure that flows through your graph. It's the "memory" of your workflow, containing:
- Input data and user requests
- Intermediate processing results
- Agent outputs and decisions
- Metadata and context information

### Defining State with TypedDict

```python
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph

class BasicState(TypedDict):
    input: str
    output: str
    step_count: int

# More complex state example
class AdvancedState(TypedDict):
    # User input and context
    user_query: str
    conversation_history: List[Dict[str, str]]
    
    # Processing data
    extracted_entities: List[str]
    search_results: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    
    # Control flow
    current_step: str
    retry_count: int
    error_messages: List[str]
    
    # Final output
    final_response: str
    confidence_score: float
```

### State Management Best Practices

#### 1. Immutable Updates
```python
def process_node(state: BasicState) -> BasicState:
    # ✅ Good: Create new state with updates
    return {
        **state,
        "output": f"Processed: {state['input']}",
        "step_count": state["step_count"] + 1
    }

def bad_process_node(state: BasicState) -> BasicState:
    # ❌ Bad: Direct mutation (avoid this)
    state["output"] = f"Processed: {state['input']}"
    return state
```

#### 2. State Validation
```python
from pydantic import BaseModel, Field
from typing import Optional

class ValidatedState(BaseModel):
    input: str = Field(..., min_length=1, description="User input")
    output: Optional[str] = Field(None, description="Processing output")
    step_count: int = Field(default=0, ge=0, description="Number of steps completed")
    
    class Config:
        extra = "forbid"  # Prevent unexpected fields
```

---

## Nodes: Processing Units

### Basic Node Structure

A node is a Python function that:
1. Takes state as input
2. Performs some processing
3. Returns updated state

```python
def simple_node(state: BasicState) -> BasicState:
    """A basic node that processes input."""
    processed_input = state["input"].upper()
    
    return {
        **state,
        "output": processed_input,
        "step_count": state["step_count"] + 1
    }
```

### Advanced Node Patterns

#### 1. LLM Integration Node
```python
from langchain_openai import ChatOpenAI

def llm_node(state: AdvancedState) -> AdvancedState:
    """Node that calls an LLM."""
    llm = ChatOpenAI(model="gpt-4")
    
    # Prepare prompt with state context
    prompt = f"""
    User Query: {state['user_query']}
    Previous Analysis: {state.get('analysis_results', 'None')}
    
    Please provide a response:
    """
    
    response = llm.invoke(prompt)
    
    return {
        **state,
        "final_response": response.content,
        "current_step": "llm_completed"
    }
```

#### 2. Tool Integration Node
```python
from langchain_community.tools import DuckDuckGoSearchRun

def search_node(state: AdvancedState) -> AdvancedState:
    """Node that performs web search."""
    search_tool = DuckDuckGoSearchRun()
    
    try:
        search_results = search_tool.run(state["user_query"])
        
        return {
            **state,
            "search_results": [{"content": search_results, "source": "DuckDuckGo"}],
            "current_step": "search_completed"
        }
    except Exception as e:
        return {
            **state,
            "error_messages": state.get("error_messages", []) + [str(e)],
            "current_step": "search_failed"
        }
```

#### 3. Conditional Processing Node
```python
def analysis_node(state: AdvancedState) -> AdvancedState:
    """Node that performs conditional analysis."""
    user_query = state["user_query"].lower()
    
    if "weather" in user_query:
        analysis_type = "weather_query"
        confidence = 0.9
    elif "news" in user_query:
        analysis_type = "news_query"
        confidence = 0.8
    else:
        analysis_type = "general_query"
        confidence = 0.6
    
    analysis_results = {
        "query_type": analysis_type,
        "confidence": confidence,
        "requires_search": confidence < 0.7
    }
    
    return {
        **state,
        "analysis_results": analysis_results,
        "current_step": "analysis_completed"
    }
```

### Error Handling in Nodes

```python
def robust_node(state: BasicState) -> BasicState:
    """Node with comprehensive error handling."""
    try:
        # Main processing logic
        result = complex_processing(state["input"])
        
        return {
            **state,
            "output": result,
            "current_step": "success"
        }
    
    except ValueError as e:
        # Handle specific error types
        return {
            **state,
            "error_messages": [f"Value error: {str(e)}"],
            "current_step": "validation_error"
        }
    
    except Exception as e:
        # Handle unexpected errors
        return {
            **state,
            "error_messages": [f"Unexpected error: {str(e)}"],
            "current_step": "system_error"
        }

def complex_processing(input_text: str) -> str:
    if not input_text:
        raise ValueError("Input cannot be empty")
    return f"Processed: {input_text}"
```

---

## Edges: Connecting the Flow

### Direct Edges

Direct edges create unconditional connections between nodes.

```python
from langgraph.graph import StateGraph

# Create graph
workflow = StateGraph(BasicState)

# Add nodes
workflow.add_node("process", simple_node)
workflow.add_node("finalize", finalize_node)

# Add direct edge (unconditional flow)
workflow.add_edge("process", "finalize")
```

### Entry and Exit Points

```python
# Set entry point
workflow.set_entry_point("process")

# Set finish point
workflow.set_finish_point("finalize")

# Compile the graph
app = workflow.compile()
```

---

## Conditional Edges and Routing

### Basic Conditional Routing

Conditional edges allow dynamic routing based on state content.

```python
def route_based_on_step(state: AdvancedState) -> str:
    """Routing function that determines next step."""
    current_step = state.get("current_step", "start")
    
    if current_step == "analysis_completed":
        # Route based on analysis results
        if state["analysis_results"]["requires_search"]:
            return "search"
        else:
            return "llm_direct"
    
    elif current_step == "search_completed":
        return "llm_with_search"
    
    elif current_step in ["llm_completed", "llm_direct_completed"]:
        return "END"
    
    else:
        return "analysis"  # Default route

# Add conditional edges
workflow.add_conditional_edges(
    "analysis",
    route_based_on_step,
    {
        "search": "search_node",
        "llm_direct": "llm_direct_node",
        "END": "__end__"
    }
)
```

### Advanced Routing Patterns

#### 1. Multi-Condition Routing
```python
def complex_router(state: AdvancedState) -> str:
    """Router with multiple conditions."""
    # Check error conditions first
    if state.get("error_messages"):
        return "error_handler"
    
    # Check retry logic
    if state.get("retry_count", 0) > 3:
        return "max_retries_exceeded"
    
    # Check confidence scores
    confidence = state.get("confidence_score", 0.0)
    if confidence < 0.5:
        return "low_confidence_handler"
    elif confidence > 0.9:
        return "high_confidence_handler"
    else:
        return "standard_handler"

workflow.add_conditional_edges(
    "evaluation_node",
    complex_router,
    {
        "error_handler": "error_node",
        "max_retries_exceeded": "failure_node",
        "low_confidence_handler": "clarification_node",
        "high_confidence_handler": "quick_response_node",
        "standard_handler": "standard_processing_node"
    }
)
```

#### 2. State-Based Loops
```python
def loop_controller(state: AdvancedState) -> str:
    """Controls iterative processing loops."""
    max_iterations = 5
    current_iteration = state.get("iteration_count", 0)
    
    # Check termination conditions
    if current_iteration >= max_iterations:
        return "max_iterations"
    
    if state.get("goal_achieved", False):
        return "success"
    
    if state.get("unrecoverable_error", False):
        return "failure"
    
    # Continue iteration
    return "continue_loop"

workflow.add_conditional_edges(
    "iteration_node",
    loop_controller,
    {
        "continue_loop": "processing_node",
        "max_iterations": "timeout_handler",
        "success": "success_node",
        "failure": "failure_node"
    }
)
```

---

## Complete Example: Simple Workflow

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class WorkflowState(TypedDict):
    input: str
    processed_data: str
    analysis: str
    final_output: str
    step: str

def input_processor(state: WorkflowState) -> WorkflowState:
    """Process and clean input data."""
    cleaned_input = state["input"].strip().lower()
    
    return {
        **state,
        "processed_data": cleaned_input,
        "step": "processed"
    }

def analyzer(state: WorkflowState) -> WorkflowState:
    """Analyze the processed data."""
    data = state["processed_data"]
    
    if "question" in data:
        analysis = "user_question"
    elif "help" in data:
        analysis = "help_request"
    else:
        analysis = "general_statement"
    
    return {
        **state,
        "analysis": analysis,
        "step": "analyzed"
    }

def response_generator(state: WorkflowState) -> WorkflowState:
    """Generate appropriate response."""
    analysis = state["analysis"]
    
    if analysis == "user_question":
        response = f"I understand you're asking about: {state['processed_data']}"
    elif analysis == "help_request":
        response = "I'm here to help! What do you need assistance with?"
    else:
        response = f"I received your message: {state['processed_data']}"
    
    return {
        **state,
        "final_output": response,
        "step": "completed"
    }

def route_after_analysis(state: WorkflowState) -> str:
    """Route based on analysis results."""
    if state["step"] == "analyzed":
        return "generate_response"
    else:
        return "END"

# Build the workflow
def create_simple_workflow():
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("process_input", input_processor)
    workflow.add_node("analyze", analyzer)
    workflow.add_node("generate_response", response_generator)
    
    # Add edges
    workflow.add_edge("process_input", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "generate_response": "generate_response",
            "END": "__end__"
        }
    )
    workflow.add_edge("generate_response", "__end__")
    
    # Set entry point
    workflow.set_entry_point("process_input")
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    app = create_simple_workflow()
    
    # Run the workflow
    initial_state = {
        "input": "How does machine learning work?",
        "processed_data": "",
        "analysis": "",
        "final_output": "",
        "step": "start"
    }
    
    final_state = app.invoke(initial_state)
    print(f"Final output: {final_state['final_output']}")
```

---

## Best Practices for StateGraph Design

### 1. State Schema Design
- Keep state structure flat when possible
- Use descriptive field names
- Include control flow fields (`current_step`, `iteration_count`)
- Add error handling fields (`error_messages`, `retry_count`)

### 2. Node Design Principles
- **Single Responsibility**: Each node should have one clear purpose
- **Stateless Logic**: Business logic shouldn't depend on external state
- **Error Handling**: Always handle potential errors gracefully
- **Immutable Updates**: Return new state objects, don't mutate input

### 3. Edge and Routing Design
- **Clear Conditions**: Make routing logic explicit and testable
- **Default Routes**: Always have fallback routes for unexpected states
- **Loop Prevention**: Include safeguards against infinite loops
- **Error Routes**: Plan for error conditions and recovery paths

### 4. Testing Strategies
```python
def test_node_behavior():
    """Test individual nodes in isolation."""
    test_state = {
        "input": "test input",
        "processed_data": "",
        "analysis": "",
        "final_output": "",
        "step": "start"
    }
    
    result = input_processor(test_state)
    
    assert result["processed_data"] == "test input"
    assert result["step"] == "processed"
    assert result["input"] == "test input"  # Original data preserved

def test_routing_logic():
    """Test routing functions."""
    test_state = {
        "analysis": "user_question",
        "step": "analyzed"
    }
    
    route = route_after_analysis(test_state)
    assert route == "generate_response"
```

## Next Steps

After mastering StateGraph fundamentals:

1. **Practice building simple graphs** with 2-3 nodes
2. **Experiment with different routing patterns**
3. **Learn about parallel processing** and advanced patterns
4. **Study error handling** and resilience patterns
5. **Explore state persistence** and memory integration

The StateGraph is your foundation for building sophisticated LangGraph applications. Master these concepts before moving to more advanced patterns.
