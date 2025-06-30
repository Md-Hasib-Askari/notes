# State Management

## State Design Patterns

### Defining Effective State Schemas

Well-designed state schemas are the foundation of robust LangGraph applications:

```python
from typing import TypedDict, Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# Basic TypedDict approach
class BasicAgentState(TypedDict):
    user_input: str
    processing_stage: Literal["input", "analysis", "processing", "output"]
    results: List[str]
    metadata: Dict[str, Any]

# Enhanced Pydantic approach for validation
class ValidatedAgentState(BaseModel):
    user_input: str = Field(..., min_length=1, description="User's request")
    processing_stage: Literal["input", "analysis", "processing", "output"] = "input"
    results: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Control fields
    iteration_count: int = Field(default=0, ge=0, le=100)
    error_count: int = Field(default=0, ge=0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Timestamps for tracking
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        extra = "forbid"  # Prevent unexpected fields
```

### Hierarchical State Organization

```python
class AgentMemory(TypedDict):
    short_term: List[str]
    long_term: Dict[str, Any]
    context_window: List[str]

class ProcessingContext(TypedDict):
    current_task: str
    subtasks: List[str]
    progress: float
    dependencies: List[str]

class ComplexAgentState(TypedDict):
    # Core data
    input_data: str
    output_data: str
    
    # Nested contexts
    memory: AgentMemory
    processing: ProcessingContext
    
    # Control flow
    current_node: str
    next_actions: List[str]
    
    # Error handling
    errors: List[Dict[str, str]]
    retry_count: int
```

### State Validation and Type Safety

```python
from typing import get_type_hints

def validate_state_schema(state: dict, state_class: type) -> bool:
    """Runtime state validation."""
    try:
        type_hints = get_type_hints(state_class)
        
        for field_name, expected_type in type_hints.items():
            if field_name in state:
                value = state[field_name]
                if not isinstance(value, expected_type):
                    print(f"Type mismatch for {field_name}: expected {expected_type}, got {type(value)}")
                    return False
        
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def safe_state_update(current_state: dict, updates: dict, state_class: type) -> dict:
    """Safely update state with validation."""
    new_state = {**current_state, **updates}
    
    if validate_state_schema(new_state, state_class):
        return new_state
    else:
        print("State update rejected due to validation failure")
        return current_state
```

## Immutable vs Mutable State Patterns

### Immutable State Pattern (Recommended)

```python
def immutable_processing_node(state: ComplexAgentState) -> ComplexAgentState:
    """Always return new state objects."""
    # Create new memory with updated short-term
    new_memory = {
        **state["memory"],
        "short_term": state["memory"]["short_term"] + ["new_item"]
    }
    
    # Create completely new state
    return {
        **state,
        "memory": new_memory,
        "processing": {
            **state["processing"],
            "progress": min(state["processing"]["progress"] + 0.1, 1.0)
        },
        "current_node": "processing_complete"
    }

# Efficient immutable updates with helper functions
def update_nested_state(state: dict, path: List[str], value: Any) -> dict:
    """Update nested state immutably."""
    if not path:
        return value
    
    key = path[0]
    remaining_path = path[1:]
    
    current_level = state.get(key, {})
    updated_level = update_nested_state(current_level, remaining_path, value)
    
    return {**state, key: updated_level}

# Usage: update_nested_state(state, ["memory", "short_term"], new_list)
```

### State History and Versioning

```python
class VersionedState(TypedDict):
    current: ComplexAgentState
    history: List[ComplexAgentState]
    version: int
    checkpoints: Dict[str, ComplexAgentState]

def create_checkpoint(state: VersionedState, checkpoint_name: str) -> VersionedState:
    """Create named checkpoint for rollback."""
    return {
        **state,
        "checkpoints": {
            **state["checkpoints"],
            checkpoint_name: state["current"]
        }
    }

def rollback_to_checkpoint(state: VersionedState, checkpoint_name: str) -> VersionedState:
    """Rollback to named checkpoint."""
    if checkpoint_name in state["checkpoints"]:
        return {
            **state,
            "current": state["checkpoints"][checkpoint_name],
            "history": state["history"] + [state["current"]],
            "version": state["version"] + 1
        }
    return state
```

## Memory Integration

### Connecting with LangChain Memory Systems

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

class LangChainIntegratedState(TypedDict):
    query: str
    response: str
    langchain_memory: ConversationBufferWindowMemory
    conversation_history: List[BaseMessage]

def memory_integration_node(state: LangChainIntegratedState) -> LangChainIntegratedState:
    """Integrate with LangChain memory systems."""
    memory = state.get("langchain_memory")
    if not memory:
        memory = ConversationBufferWindowMemory(k=5, return_messages=True)
    
    # Add current interaction to memory
    if state["query"]:
        memory.chat_memory.add_user_message(state["query"])
    if state["response"]:
        memory.chat_memory.add_ai_message(state["response"])
    
    # Extract conversation history
    messages = memory.chat_memory.messages
    
    return {
        **state,
        "langchain_memory": memory,
        "conversation_history": messages
    }
```

### Persistent State Across Sessions

```python
import json
import pickle
from pathlib import Path

class PersistentStateManager:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_state(self, session_id: str, state: dict) -> bool:
        """Save state to persistent storage."""
        try:
            file_path = self.storage_path / f"{session_id}_state.json"
            
            # Convert to JSON-serializable format
            serializable_state = self._make_serializable(state)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save state: {e}")
            return False
    
    def load_state(self, session_id: str) -> Optional[dict]:
        """Load state from persistent storage."""
        try:
            file_path = self.storage_path / f"{session_id}_state.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            
            return None
        except Exception as e:
            print(f"Failed to load state: {e}")
            return None
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert complex objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

# Usage in LangGraph node
def persistent_state_node(state: dict) -> dict:
    """Node that automatically persists state."""
    session_id = state.get("session_id", "default")
    manager = PersistentStateManager("./sessions")
    
    # Process state
    processed_state = {
        **state,
        "last_processed": datetime.now().isoformat(),
        "processing_count": state.get("processing_count", 0) + 1
    }
    
    # Persist updated state
    manager.save_state(session_id, processed_state)
    
    return processed_state
```

### Context Management Strategies

```python
class ContextWindow:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.items: List[Dict] = []
    
    def add_context(self, item: Dict) -> None:
        """Add item to context with automatic pruning."""
        self.items.append({
            **item,
            "timestamp": datetime.now().isoformat(),
            "priority": item.get("priority", 1.0)
        })
        
        # Prune if exceeding max size
        if len(self.items) > self.max_size:
            self._prune_context()
    
    def _prune_context(self) -> None:
        """Remove least important items to maintain size limit."""
        # Sort by priority (descending) and recency
        self.items.sort(key=lambda x: (x["priority"], x["timestamp"]), reverse=True)
        self.items = self.items[:self.max_size]
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant context items."""
        # Simple relevance scoring (in practice, use embeddings)
        scored_items = []
        for item in self.items:
            relevance = self._calculate_relevance(query, item)
            scored_items.append((relevance, item))
        
        # Return top-k most relevant
        scored_items.sort(reverse=True)
        return [item for _, item in scored_items[:top_k]]
    
    def _calculate_relevance(self, query: str, item: Dict) -> float:
        """Simple relevance calculation."""
        item_text = str(item.get("content", ""))
        query_words = set(query.lower().split())
        item_words = set(item_text.lower().split())
        
        overlap = len(query_words.intersection(item_words))
        return overlap / max(len(query_words), 1)

# Integrate context management in state
class ContextManagedState(TypedDict):
    query: str
    response: str
    context_window: ContextWindow
    relevant_context: List[Dict]

def context_aware_node(state: ContextManagedState) -> ContextManagedState:
    """Node that uses context for better processing."""
    context_window = state.get("context_window", ContextWindow())
    
    # Get relevant context for current query
    relevant_context = context_window.get_relevant_context(state["query"])
    
    # Process with context awareness
    context_summary = "\n".join([
        f"- {ctx.get('content', '')}" for ctx in relevant_context
    ])
    
    enhanced_response = f"""
    Based on context: {context_summary}
    
    Response: {state.get('response', 'Processing...')}
    """
    
    # Add current interaction to context
    context_window.add_context({
        "content": f"Q: {state['query']} A: {enhanced_response}",
        "type": "interaction",
        "priority": 1.0
    })
    
    return {
        **state,
        "context_window": context_window,
        "relevant_context": relevant_context,
        "response": enhanced_response
    }
```

## State Performance Optimization

### Efficient State Updates

```python
def minimal_state_copy(state: dict, updates: dict) -> dict:
    """Only copy what's necessary."""
    # Identify what actually changed
    changed_keys = {k for k, v in updates.items() if k not in state or state[k] != v}
    
    if not changed_keys:
        return state  # No changes needed
    
    # Only update changed fields
    return {**state, **{k: updates[k] for k in changed_keys}}

def lazy_state_computation(state: dict) -> dict:
    """Compute expensive state fields only when needed."""
    def compute_summary():
        # Expensive computation
        return f"Summary of {len(state.get('items', []))} items"
    
    # Use property-like access for computed fields
    state_with_computed = {**state}
    state_with_computed["_compute_summary"] = compute_summary
    
    return state_with_computed
```

These state management patterns provide the foundation for building scalable, maintainable LangGraph applications with proper memory handling and performance optimization.
