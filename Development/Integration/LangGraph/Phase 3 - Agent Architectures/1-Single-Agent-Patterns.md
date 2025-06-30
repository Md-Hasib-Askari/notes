# Single Agent Patterns

## ReAct (Reason + Act) Implementation

### Core ReAct Pattern
ReAct combines reasoning and acting in an iterative loop, allowing agents to think through problems and take actions based on observations.

```python
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import StateGraph
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

class ReActState(TypedDict):
    task: str
    thought: str
    action: str
    action_input: str
    observation: str
    answer: str
    scratchpad: List[str]
    iteration: int
    max_iterations: int
    tools_used: List[str]

def reasoning_node(state: ReActState) -> ReActState:
    """Generate reasoning about next action."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    context = "\n".join(state["scratchpad"]) if state["scratchpad"] else ""
    
    prompt = f"""
    Task: {state['task']}
    Previous steps: {context}
    
    Think step by step about what to do next.
    Available tools: search, calculator, text_processor
    
    Thought:"""
    
    response = llm.invoke(prompt)
    thought = response.content.strip()
    
    return {
        **state,
        "thought": thought,
        "scratchpad": state["scratchpad"] + [f"Thought: {thought}"]
    }

def action_planning_node(state: ReActState) -> ReActState:
    """Plan specific action based on reasoning."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"""
    Based on this thought: {state['thought']}
    
    Choose an action:
    - search[query]: Search for information
    - calculator[expression]: Calculate mathematical expressions
    - text_processor[text]: Process and analyze text
    - finish[answer]: Complete the task with final answer
    
    Action:"""
    
    response = llm.invoke(prompt)
    action_text = response.content.strip()
    
    # Parse action and input
    if "[" in action_text and "]" in action_text:
        action = action_text.split("[")[0]
        action_input = action_text.split("[")[1].split("]")[0]
    else:
        action = "finish"
        action_input = action_text
    
    return {
        **state,
        "action": action,
        "action_input": action_input,
        "scratchpad": state["scratchpad"] + [f"Action: {action}[{action_input}]"]
    }
```

### Tool Integration Within Graphs

```python
def tool_execution_node(state: ReActState) -> ReActState:
    """Execute the planned action using appropriate tools."""
    action = state["action"]
    action_input = state["action_input"]
    
    tools = {
        "search": lambda x: f"Search results for '{x}': Mock search data...",
        "calculator": lambda x: f"Calculation result: {eval(x) if x.replace('.','').replace('-','').isdigit() else 'Error'}",
        "text_processor": lambda x: f"Text analysis of '{x}': Contains {len(x.split())} words"
    }
    
    if action in tools:
        try:
            observation = tools[action](action_input)
            tools_used = state["tools_used"] + [action]
        except Exception as e:
            observation = f"Tool error: {str(e)}"
            tools_used = state["tools_used"]
    elif action == "finish":
        observation = "Task completed"
        tools_used = state["tools_used"]
    else:
        observation = f"Unknown action: {action}"
        tools_used = state["tools_used"]
    
    return {
        **state,
        "observation": observation,
        "tools_used": tools_used,
        "iteration": state["iteration"] + 1,
        "scratchpad": state["scratchpad"] + [f"Observation: {observation}"]
    }

def react_controller(state: ReActState) -> str:
    """Control ReAct loop flow."""
    if state["action"] == "finish":
        return "finalize"
    elif state["iteration"] >= state["max_iterations"]:
        return "timeout"
    elif "error" in state["observation"].lower():
        return "error_recovery"
    else:
        return "continue_reasoning"

def error_recovery_node(state: ReActState) -> ReActState:
    """Handle errors and adjust strategy."""
    recovery_thought = f"Error encountered: {state['observation']}. Need to try different approach."
    
    return {
        **state,
        "thought": recovery_thought,
        "scratchpad": state["scratchpad"] + [f"Recovery: {recovery_thought}"]
    }

# Build ReAct workflow
def create_react_agent():
    workflow = StateGraph(ReActState)
    
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("plan_action", action_planning_node)
    workflow.add_node("execute_tool", tool_execution_node)
    workflow.add_node("error_recovery", error_recovery_node)
    workflow.add_node("finalize", lambda s: {**s, "answer": s["observation"]})
    
    workflow.set_entry_point("reason")
    workflow.add_edge("reason", "plan_action")
    workflow.add_edge("plan_action", "execute_tool")
    
    workflow.add_conditional_edges(
        "execute_tool",
        react_controller,
        {
            "continue_reasoning": "reason",
            "error_recovery": "error_recovery",
            "finalize": "finalize",
            "timeout": "finalize"
        }
    )
    
    workflow.add_edge("error_recovery", "reason")
    workflow.add_edge("finalize", "__end__")
    
    return workflow.compile()
```

## Chain of Thought Integration

### Multi-Step Reasoning Graphs

```python
class ChainOfThoughtState(TypedDict):
    problem: str
    reasoning_steps: List[Dict[str, str]]
    current_step: int
    step_confidence: List[float]
    final_answer: str
    validation_results: List[str]
    needs_revision: bool

def step_reasoner(state: ChainOfThoughtState) -> ChainOfThoughtState:
    """Generate detailed reasoning for current step."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    problem = state["problem"]
    prev_steps = state["reasoning_steps"]
    
    context = ""
    if prev_steps:
        context = "\n".join([f"Step {i+1}: {step['reasoning']}" for i, step in enumerate(prev_steps)])
    
    prompt = f"""
    Problem: {problem}
    Previous reasoning: {context}
    
    Continue the reasoning process. Explain your next logical step clearly.
    Focus on one specific aspect or sub-problem.
    
    Step {len(prev_steps) + 1} reasoning:"""
    
    response = llm.invoke(prompt)
    reasoning = response.content.strip()
    
    # Assess confidence in this step
    confidence_prompt = f"Rate confidence in this reasoning (0.0-1.0): {reasoning}"
    confidence_response = llm.invoke(confidence_prompt)
    
    try:
        confidence = float(confidence_response.content.strip())
    except:
        confidence = 0.7  # Default confidence
    
    new_step = {
        "step_number": len(prev_steps) + 1,
        "reasoning": reasoning,
        "confidence": confidence
    }
    
    return {
        **state,
        "reasoning_steps": prev_steps + [new_step],
        "current_step": len(prev_steps) + 1,
        "step_confidence": state["step_confidence"] + [confidence]
    }

def intermediate_validator(state: ChainOfThoughtState) -> ChainOfThoughtState:
    """Validate intermediate reasoning steps."""
    if not state["reasoning_steps"]:
        return state
    
    latest_step = state["reasoning_steps"][-1]
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    validation_prompt = f"""
    Original problem: {state['problem']}
    Current reasoning step: {latest_step['reasoning']}
    
    Validate this reasoning step:
    1. Is it logically sound?
    2. Does it follow from previous steps?
    3. Does it move toward solving the problem?
    
    Validation result:"""
    
    validation = llm.invoke(validation_prompt).content.strip()
    
    # Determine if revision needed
    needs_revision = any(word in validation.lower() for word in ['incorrect', 'flawed', 'error', 'wrong'])
    
    return {
        **state,
        "validation_results": state["validation_results"] + [validation],
        "needs_revision": needs_revision
    }

def dynamic_path_adjuster(state: ChainOfThoughtState) -> ChainOfThoughtState:
    """Adjust reasoning path based on validation."""
    if state["needs_revision"] and state["reasoning_steps"]:
        # Remove problematic step and add correction note
        corrected_steps = state["reasoning_steps"][:-1]
        correction_note = {
            "step_number": len(corrected_steps) + 1,
            "reasoning": f"Revision needed for previous step. Reconsidering approach...",
            "confidence": 0.5
        }
        
        return {
            **state,
            "reasoning_steps": corrected_steps + [correction_note],
            "current_step": len(corrected_steps) + 1,
            "needs_revision": False
        }
    
    return state

def reasoning_controller(state: ChainOfThoughtState) -> str:
    """Control chain of thought flow."""
    # Check if we have enough reasoning steps
    if len(state["reasoning_steps"]) >= 5:
        # Check average confidence
        avg_confidence = sum(state["step_confidence"]) / len(state["step_confidence"])
        if avg_confidence >= 0.7:
            return "conclude"
        else:
            return "revise"
    
    # Continue reasoning if validation passed
    if not state["needs_revision"]:
        return "continue"
    else:
        return "adjust"

def conclusion_generator(state: ChainOfThoughtState) -> ChainOfThoughtState:
    """Generate final answer from reasoning chain."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    reasoning_summary = "\n".join([
        f"{step['step_number']}. {step['reasoning']}" 
        for step in state["reasoning_steps"]
    ])
    
    conclusion_prompt = f"""
    Problem: {state['problem']}
    
    Reasoning chain:
    {reasoning_summary}
    
    Based on this step-by-step reasoning, provide the final answer:"""
    
    final_answer = llm.invoke(conclusion_prompt).content.strip()
    
    return {
        **state,
        "final_answer": final_answer
    }

# Build Chain of Thought workflow
def create_cot_agent():
    workflow = StateGraph(ChainOfThoughtState)
    
    workflow.add_node("reason_step", step_reasoner)
    workflow.add_node("validate", intermediate_validator)
    workflow.add_node("adjust_path", dynamic_path_adjuster)
    workflow.add_node("conclude", conclusion_generator)
    
    workflow.set_entry_point("reason_step")
    workflow.add_edge("reason_step", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        reasoning_controller,
        {
            "continue": "reason_step",
            "adjust": "adjust_path",
            "conclude": "conclude",
            "revise": "adjust_path"
        }
    )
    
    workflow.add_edge("adjust_path", "reason_step")
    workflow.add_edge("conclude", "__end__")
    
    return workflow.compile()

# Usage examples
def test_single_agent_patterns():
    # Test ReAct agent
    react_app = create_react_agent()
    react_result = react_app.invoke({
        "task": "Find the population of Tokyo and calculate what 5% of that would be",
        "thought": "",
        "action": "",
        "action_input": "",
        "observation": "",
        "answer": "",
        "scratchpad": [],
        "iteration": 0,
        "max_iterations": 5,
        "tools_used": []
    })
    
    print("ReAct Agent Result:")
    print(f"Answer: {react_result['answer']}")
    print(f"Tools used: {react_result['tools_used']}")
    
    # Test Chain of Thought agent
    cot_app = create_cot_agent()
    cot_result = cot_app.invoke({
        "problem": "A train travels 120 km in 2 hours, then stops for 30 minutes, then travels another 180 km in 2.5 hours. What is the average speed for the entire journey?",
        "reasoning_steps": [],
        "current_step": 0,
        "step_confidence": [],
        "final_answer": "",
        "validation_results": [],
        "needs_revision": False
    })
    
    print("\nChain of Thought Agent Result:")
    print(f"Final Answer: {cot_result['final_answer']}")
    print(f"Reasoning Steps: {len(cot_result['reasoning_steps'])}")

if __name__ == "__main__":
    test_single_agent_patterns()
```

These single agent patterns provide the foundation for building sophisticated reasoning agents that can integrate tools, validate their thinking, and adapt their approach based on feedback and results.
