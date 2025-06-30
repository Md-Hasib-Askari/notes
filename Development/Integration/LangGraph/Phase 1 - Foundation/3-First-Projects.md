# First Projects: Building Your Foundation

This section provides hands-on projects to solidify your understanding of LangGraph fundamentals. These projects progress from simple to more complex, giving you practical experience with state management, node creation, and workflow design.

## Project 1: Simple Linear Chain

### Project Overview
Create a basic graph with 2-3 sequential nodes that demonstrates fundamental LangGraph concepts:
- State passing between nodes
- Sequential workflow execution
- Basic data transformation

### Learning Objectives
- Understand how state flows through nodes
- Practice creating and connecting nodes
- Learn basic graph compilation and execution

### Project Implementation

#### Step 1: Define the State Schema
```python
from typing import TypedDict, List
from langgraph.graph import StateGraph

class LinearChainState(TypedDict):
    original_input: str
    step1_output: str
    step2_output: str
    final_result: str
    processing_log: List[str]
```

#### Step 2: Create Processing Nodes
```python
def input_preparation_node(state: LinearChainState) -> LinearChainState:
    """First node: Clean and prepare input data."""
    original = state["original_input"]
    
    # Simple preprocessing
    cleaned = original.strip().lower()
    step1_result = f"Cleaned: {cleaned}"
    
    # Update processing log
    log = state.get("processing_log", [])
    log.append(f"Step 1: Input prepared - '{original}' -> '{cleaned}'")
    
    return {
        **state,
        "step1_output": step1_result,
        "processing_log": log
    }

def text_analysis_node(state: LinearChainState) -> LinearChainState:
    """Second node: Analyze the processed text."""
    input_text = state["step1_output"]
    
    # Extract information
    word_count = len(input_text.split())
    char_count = len(input_text)
    has_numbers = any(char.isdigit() for char in input_text)
    
    analysis = {
        "word_count": word_count,
        "character_count": char_count,
        "contains_numbers": has_numbers,
        "length_category": "short" if word_count < 5 else "medium" if word_count < 15 else "long"
    }
    
    step2_result = f"Analysis: {analysis}"
    
    # Update log
    log = state["processing_log"]
    log.append(f"Step 2: Text analyzed - {word_count} words, {char_count} characters")
    
    return {
        **state,
        "step2_output": step2_result,
        "processing_log": log
    }

def result_formatter_node(state: LinearChainState) -> LinearChainState:
    """Third node: Format and finalize results."""
    
    # Create comprehensive summary
    summary_parts = [
        f"Original Input: {state['original_input']}",
        f"Processed: {state['step1_output']}",
        f"Analysis: {state['step2_output']}",
        f"Processing Steps: {len(state['processing_log'])}"
    ]
    
    final_result = "\n".join(summary_parts)
    
    # Final log entry
    log = state["processing_log"]
    log.append("Step 3: Results formatted and finalized")
    
    return {
        **state,
        "final_result": final_result,
        "processing_log": log
    }
```

#### Step 3: Build the Workflow
```python
def create_linear_chain_workflow():
    """Create and configure the linear chain workflow."""
    
    # Initialize the graph
    workflow = StateGraph(LinearChainState)
    
    # Add nodes in sequence
    workflow.add_node("input_preparation", input_preparation_node)
    workflow.add_node("text_analysis", text_analysis_node)
    workflow.add_node("result_formatter", result_formatter_node)
    
    # Connect nodes sequentially
    workflow.add_edge("input_preparation", "text_analysis")
    workflow.add_edge("text_analysis", "result_formatter")
    workflow.add_edge("result_formatter", "__end__")
    
    # Set entry point
    workflow.set_entry_point("input_preparation")
    
    return workflow.compile()
```

#### Step 4: Test and Run
```python
def test_linear_chain():
    """Test the linear chain workflow."""
    
    # Create the workflow
    app = create_linear_chain_workflow()
    
    # Test cases
    test_inputs = [
        "Hello World! This is a test.",
        "AI and Machine Learning are fascinating topics with 123 applications.",
        "Short text",
        "This is a much longer piece of text that contains multiple sentences and should be categorized as long text for testing purposes."
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}: {test_input}")
        print('='*50)
        
        # Initialize state
        initial_state = {
            "original_input": test_input,
            "step1_output": "",
            "step2_output": "",
            "final_result": "",
            "processing_log": []
        }
        
        # Run workflow
        result = app.invoke(initial_state)
        
        # Display results
        print("\nFinal Result:")
        print(result["final_result"])
        
        print("\nProcessing Log:")
        for step in result["processing_log"]:
            print(f"  - {step}")

if __name__ == "__main__":
    test_linear_chain()
```

### Enhancement Ideas
1. **Add error handling** for invalid inputs
2. **Include timing information** in the processing log
3. **Add data validation** between steps
4. **Implement different analysis types** based on input characteristics

---

## Project 2: Conditional Workflow

### Project Overview
Build a graph with branching logic that demonstrates:
- Conditional routing based on state
- Multiple processing paths
- State-based decision making

### Learning Objectives
- Implement conditional edges
- Create routing functions
- Handle different workflow paths
- Merge results from different branches

### Project Implementation

#### Step 1: Define Enhanced State Schema
```python
class ConditionalWorkflowState(TypedDict):
    user_input: str
    input_type: str
    processing_path: str
    question_analysis: str
    command_execution: str
    general_response: str
    final_output: str
    metadata: dict
```

#### Step 2: Create Analysis Node
```python
def input_analyzer_node(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """Analyze input to determine processing path."""
    user_input = state["user_input"].lower().strip()
    
    # Determine input type
    if user_input.endswith('?') or any(word in user_input for word in ['what', 'how', 'why', 'when', 'where', 'who']):
        input_type = "question"
    elif any(word in user_input for word in ['run', 'execute', 'start', 'stop', 'create', 'delete']):
        input_type = "command"
    else:
        input_type = "general"
    
    # Create metadata
    metadata = {
        "word_count": len(user_input.split()),
        "has_keywords": input_type != "general",
        "analysis_confidence": 0.9 if input_type != "general" else 0.6,
        "timestamp": "2024-01-01T00:00:00Z"  # In real app, use actual timestamp
    }
    
    return {
        **state,
        "input_type": input_type,
        "metadata": metadata,
        "processing_path": f"Detected as {input_type}"
    }
```

#### Step 3: Create Specialized Processing Nodes
```python
def question_processor_node(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """Process question-type inputs."""
    user_input = state["user_input"]
    
    # Analyze question type
    question_words = ['what', 'how', 'why', 'when', 'where', 'who']
    question_type = next((word for word in question_words if word in user_input.lower()), 'general')
    
    # Generate appropriate response
    responses = {
        'what': f"You're asking about the nature or definition of something in: '{user_input}'",
        'how': f"You want to know the process or method for: '{user_input}'",
        'why': f"You're seeking reasons or explanations for: '{user_input}'",
        'when': f"You're asking about timing related to: '{user_input}'",
        'where': f"You want to know the location or place for: '{user_input}'",
        'who': f"You're asking about people involved in: '{user_input}'",
        'general': f"I understand this is a question: '{user_input}'"
    }
    
    analysis = f"Question type: {question_type.upper()} - {responses[question_type]}"
    
    return {
        **state,
        "question_analysis": analysis,
        "processing_path": state["processing_path"] + " -> Question Processed"
    }

def command_processor_node(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """Process command-type inputs."""
    user_input = state["user_input"].lower()
    
    # Identify command type
    if 'run' in user_input or 'execute' in user_input:
        command_type = "execution"
        action = "I would execute the requested operation"
    elif 'create' in user_input:
        command_type = "creation"
        action = "I would create the requested resource"
    elif 'delete' in user_input or 'remove' in user_input:
        command_type = "deletion"
        action = "I would safely remove the specified item"
    elif 'start' in user_input:
        command_type = "startup"
        action = "I would initiate the requested service"
    elif 'stop' in user_input:
        command_type = "shutdown"
        action = "I would safely stop the requested service"
    else:
        command_type = "general"
        action = "I would process this command"
    
    execution_result = f"Command type: {command_type} - {action} for: '{state['user_input']}'"
    
    return {
        **state,
        "command_execution": execution_result,
        "processing_path": state["processing_path"] + " -> Command Processed"
    }

def general_processor_node(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """Process general statements or unclear inputs."""
    user_input = state["user_input"]
    
    # Analyze sentiment and provide appropriate response
    positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
    
    has_positive = any(word in user_input.lower() for word in positive_words)
    has_negative = any(word in user_input.lower() for word in negative_words)
    
    if has_positive:
        sentiment = "positive"
        response = f"I sense positivity in your message: '{user_input}'. That's great!"
    elif has_negative:
        sentiment = "negative"
        response = f"I notice some concerns in: '{user_input}'. I'm here to help."
    else:
        sentiment = "neutral"
        response = f"I received your message: '{user_input}'. Could you provide more details?"
    
    general_response = f"Sentiment: {sentiment} - {response}"
    
    return {
        **state,
        "general_response": general_response,
        "processing_path": state["processing_path"] + " -> General Processed"
    }
```

#### Step 4: Create Routing Logic
```python
def route_by_input_type(state: ConditionalWorkflowState) -> str:
    """Route to appropriate processor based on input type."""
    input_type = state["input_type"]
    
    routing_map = {
        "question": "question_processor",
        "command": "command_processor",
        "general": "general_processor"
    }
    
    return routing_map.get(input_type, "general_processor")

def merge_results_node(state: ConditionalWorkflowState) -> ConditionalWorkflowState:
    """Merge results from different processing paths."""
    
    # Collect all results
    results = []
    
    if state.get("question_analysis"):
        results.append(f"Question Analysis: {state['question_analysis']}")
    
    if state.get("command_execution"):
        results.append(f"Command Processing: {state['command_execution']}")
    
    if state.get("general_response"):
        results.append(f"General Response: {state['general_response']}")
    
    # Create comprehensive final output
    final_output = f"""
Input: {state['user_input']}
Type: {state['input_type']}
Processing Path: {state['processing_path']}

Results:
{chr(10).join(f'  - {result}' for result in results)}

Metadata:
  - Confidence: {state['metadata']['analysis_confidence']}
  - Word Count: {state['metadata']['word_count']}
  - Processing Complete: True
"""
    
    return {
        **state,
        "final_output": final_output.strip(),
        "processing_path": state["processing_path"] + " -> Results Merged"
    }
```

#### Step 5: Build the Conditional Workflow
```python
def create_conditional_workflow():
    """Create the conditional branching workflow."""
    
    workflow = StateGraph(ConditionalWorkflowState)
    
    # Add all nodes
    workflow.add_node("analyzer", input_analyzer_node)
    workflow.add_node("question_processor", question_processor_node)
    workflow.add_node("command_processor", command_processor_node)
    workflow.add_node("general_processor", general_processor_node)
    workflow.add_node("merge_results", merge_results_node)
    
    # Set entry point
    workflow.set_entry_point("analyzer")
    
    # Add conditional routing from analyzer
    workflow.add_conditional_edges(
        "analyzer",
        route_by_input_type,
        {
            "question_processor": "question_processor",
            "command_processor": "command_processor",
            "general_processor": "general_processor"
        }
    )
    
    # All paths converge to merge_results
    workflow.add_edge("question_processor", "merge_results")
    workflow.add_edge("command_processor", "merge_results")
    workflow.add_edge("general_processor", "merge_results")
    workflow.add_edge("merge_results", "__end__")
    
    return workflow.compile()
```

#### Step 6: Test the Conditional Workflow
```python
def test_conditional_workflow():
    """Test the conditional workflow with various inputs."""
    
    app = create_conditional_workflow()
    
    test_cases = [
        "What is machine learning?",  # Question
        "How do neural networks work?",  # Question
        "Run the data processing pipeline",  # Command
        "Create a new user account",  # Command
        "This is a beautiful day!",  # General (positive)
        "I'm having trouble with this",  # General (negative)
        "Hello there",  # General (neutral)
        "Why is the sky blue?",  # Question
        "Delete the old files",  # Command
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}")
        print('='*60)
        
        initial_state = {
            "user_input": test_input,
            "input_type": "",
            "processing_path": "",
            "question_analysis": "",
            "command_execution": "",
            "general_response": "",
            "final_output": "",
            "metadata": {}
        }
        
        result = app.invoke(initial_state)
        print(result["final_output"])

if __name__ == "__main__":
    test_conditional_workflow()
```

### Enhancement Ideas
1. **Add confidence scoring** for routing decisions
2. **Implement retry logic** for uncertain classifications
3. **Add more sophisticated NLP** for better input analysis
4. **Create sub-workflows** for complex processing paths
5. **Add logging and monitoring** capabilities

---

## Project Extensions and Next Steps

### Integration Projects
1. **Combine both projects** into a single workflow that first determines complexity, then routes to either linear or conditional processing
2. **Add external API calls** (weather, news, search) to make the workflows more practical
3. **Implement user memory** to remember previous interactions

### Advanced Features to Explore
1. **Error handling and recovery**
2. **Parallel processing paths**
3. **Dynamic node creation**
4. **State persistence across sessions**
5. **Integration with LangChain tools and agents**

### Learning Validation
After completing these projects, you should be able to:
- ✅ Create and manage state schemas effectively
- ✅ Build nodes that transform state appropriately
- ✅ Implement conditional routing logic
- ✅ Connect nodes with both direct and conditional edges
- ✅ Debug and test LangGraph workflows
- ✅ Handle different types of user inputs and requirements

These foundational projects prepare you for more advanced LangGraph patterns including multi-agent systems, complex state management, and production-ready applications.
