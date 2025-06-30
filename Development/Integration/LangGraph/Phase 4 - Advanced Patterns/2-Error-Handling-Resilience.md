# Error Handling and Resilience

## Robust Error Management

### Try-Catch Patterns in Graphs

```python
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph
import traceback
from enum import Enum
from datetime import datetime

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResilientState(TypedDict):
    task_data: str
    processing_results: List[Dict[str, Any]]
    error_history: List[Dict[str, Any]]
    retry_count: int
    max_retries: int
    fallback_strategies: List[str]
    current_strategy: str
    error_recovery_log: List[str]
    system_health: Dict[str, Any]

class ErrorHandler:
    """Centralized error handling for LangGraph workflows."""
    
    def __init__(self):
        self.error_patterns = {}
        self.recovery_strategies = {}
        self.error_threshold = {"low": 10, "medium": 5, "high": 2, "critical": 1}
    
    def register_error_pattern(self, pattern_name: str, error_types: List[type], 
                             severity: ErrorSeverity, recovery_func: callable):
        """Register error patterns and their recovery strategies."""
        self.error_patterns[pattern_name] = {
            "error_types": error_types,
            "severity": severity,
            "recovery_func": recovery_func
        }
    
    def handle_error(self, error: Exception, state: ResilientState) -> ResilientState:
        """Handle errors based on registered patterns."""
        error_type = type(error)
        error_info = {
            "error_type": error_type.__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc(),
            "severity": ErrorSeverity.MEDIUM  # Default
        }
        
        # Find matching error pattern
        recovery_strategy = "default_retry"
        for pattern_name, pattern_info in self.error_patterns.items():
            if error_type in pattern_info["error_types"]:
                error_info["severity"] = pattern_info["severity"]
                recovery_strategy = pattern_name
                break
        
        # Update error history
        error_history = state["error_history"] + [error_info]
        
        # Determine if error threshold exceeded
        severity_count = sum(1 for err in error_history 
                           if err["severity"] == error_info["severity"])
        
        threshold = self.error_threshold[error_info["severity"].value]
        
        return {
            **state,
            "error_history": error_history,
            "current_strategy": recovery_strategy,
            "retry_count": state["retry_count"] + 1,
            "system_health": {
                **state["system_health"],
                "error_threshold_exceeded": severity_count >= threshold,
                "last_error": error_info
            }
        }

def safe_processing_node(state: ResilientState) -> ResilientState:
    """Processing node with comprehensive error handling."""
    error_handler = ErrorHandler()
    
    # Register error patterns
    error_handler.register_error_pattern(
        "connection_error",
        [ConnectionError, TimeoutError],
        ErrorSeverity.MEDIUM,
        lambda s: exponential_backoff_retry(s)
    )
    
    error_handler.register_error_pattern(
        "data_validation_error",
        [ValueError, TypeError],
        ErrorSeverity.HIGH,
        lambda s: data_sanitization_recovery(s)
    )
    
    try:
        # Main processing logic
        if not state["task_data"]:
            raise ValueError("Empty task data provided")
        
        # Simulate processing that might fail
        if "fail" in state["task_data"].lower():
            raise ConnectionError("Simulated connection failure")
        
        result = {
            "processed_data": f"Successfully processed: {state['task_data']}",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        return {
            **state,
            "processing_results": state["processing_results"] + [result],
            "current_strategy": "normal_processing"
        }
    
    except Exception as e:
        return error_handler.handle_error(e, state)

def exponential_backoff_retry(state: ResilientState) -> ResilientState:
    """Implement exponential backoff retry strategy."""
    retry_count = state["retry_count"]
    backoff_time = min(2 ** retry_count, 60)  # Cap at 60 seconds
    
    recovery_log = state["error_recovery_log"] + [
        f"Exponential backoff: waiting {backoff_time}s before retry {retry_count + 1}"
    ]
    
    return {
        **state,
        "error_recovery_log": recovery_log,
        "current_strategy": "exponential_backoff",
        "fallback_strategies": ["circuit_breaker", "alternative_service"]
    }

def data_sanitization_recovery(state: ResilientState) -> ResilientState:
    """Recover from data validation errors through sanitization."""
    sanitized_data = state["task_data"].replace("fail", "process")
    
    recovery_log = state["error_recovery_log"] + [
        f"Data sanitization applied: '{state['task_data']}' -> '{sanitized_data}'"
    ]
    
    return {
        **state,
        "task_data": sanitized_data,
        "error_recovery_log": recovery_log,
        "current_strategy": "data_sanitization"
    }
```

### Fallback Strategies

```python
class FallbackManager:
    """Manages fallback strategies for failed operations."""
    
    def __init__(self):
        self.strategies = [
            "primary_service",
            "backup_service", 
            "cached_result",
            "simplified_processing",
            "manual_intervention"
        ]
        self.current_strategy_index = 0
    
    def get_next_strategy(self) -> Optional[str]:
        """Get the next fallback strategy."""
        if self.current_strategy_index < len(self.strategies):
            strategy = self.strategies[self.current_strategy_index]
            self.current_strategy_index += 1
            return strategy
        return None
    
    def reset_strategies(self):
        """Reset to primary strategy."""
        self.current_strategy_index = 0

def fallback_coordinator(state: ResilientState) -> ResilientState:
    """Coordinate fallback strategies."""
    fallback_manager = FallbackManager()
    
    # Check if current strategy failed
    if state["system_health"].get("error_threshold_exceeded", False):
        next_strategy = fallback_manager.get_next_strategy()
        
        if next_strategy:
            recovery_log = state["error_recovery_log"] + [
                f"Switching to fallback strategy: {next_strategy}"
            ]
            
            return {
                **state,
                "current_strategy": next_strategy,
                "error_recovery_log": recovery_log,
                "retry_count": 0  # Reset retry count for new strategy
            }
        else:
            return {
                **state,
                "current_strategy": "all_fallbacks_exhausted",
                "system_health": {
                    **state["system_health"],
                    "critical_failure": True
                }
            }
    
    return state

def graceful_degradation_node(state: ResilientState) -> ResilientState:
    """Implement graceful degradation when full functionality fails."""
    
    if state["current_strategy"] == "simplified_processing":
        # Provide basic functionality instead of full processing
        simplified_result = {
            "processed_data": f"Basic processing of: {state['task_data']}",
            "timestamp": datetime.now().isoformat(),
            "status": "degraded",
            "functionality_level": "basic"
        }
        
        return {
            **state,
            "processing_results": state["processing_results"] + [simplified_result],
            "error_recovery_log": state["error_recovery_log"] + [
                "Graceful degradation: Providing basic functionality"
            ]
        }
    
    elif state["current_strategy"] == "cached_result":
        # Return cached or default result
        cached_result = {
            "processed_data": "Cached/default result",
            "timestamp": datetime.now().isoformat(),
            "status": "cached",
            "functionality_level": "limited"
        }
        
        return {
            **state,
            "processing_results": state["processing_results"] + [cached_result],
            "error_recovery_log": state["error_recovery_log"] + [
                "Using cached result due to service unavailability"
            ]
        }
    
    return state
```

## Monitoring and Observability

### Logging Graph Execution

```python
import logging
from typing import Dict, Any
import json
import time

class GraphExecutionLogger:
    """Comprehensive logging for LangGraph execution."""
    
    def __init__(self, graph_name: str):
        self.graph_name = graph_name
        self.logger = logging.getLogger(f"langgraph.{graph_name}")
        self.execution_id = None
        self.start_time = None
        self.node_timings = {}
        
        # Configure logger
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_execution_start(self, initial_state: Dict[str, Any]) -> str:
        """Log the start of graph execution."""
        self.execution_id = f"{self.graph_name}_{int(time.time())}"
        self.start_time = time.time()
        
        self.logger.info(f"Graph execution started: {self.execution_id}")
        self.logger.debug(f"Initial state: {json.dumps(initial_state, default=str)}")
        
        return self.execution_id
    
    def log_node_execution(self, node_name: str, input_state: Dict, 
                          output_state: Dict, execution_time: float):
        """Log individual node execution."""
        self.node_timings[node_name] = execution_time
        
        self.logger.info(
            f"Node '{node_name}' executed in {execution_time:.3f}s "
            f"[Execution: {self.execution_id}]"
        )
        
        # Log state changes
        state_changes = self._calculate_state_changes(input_state, output_state)
        if state_changes:
            self.logger.debug(f"State changes in '{node_name}': {state_changes}")
    
    def log_error(self, node_name: str, error: Exception, state: Dict):
        """Log errors during execution."""
        self.logger.error(
            f"Error in node '{node_name}': {str(error)} "
            f"[Execution: {self.execution_id}]"
        )
        self.logger.debug(f"State at error: {json.dumps(state, default=str)}")
    
    def log_execution_end(self, final_state: Dict[str, Any]):
        """Log the end of graph execution."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info(
            f"Graph execution completed: {self.execution_id} "
            f"in {total_time:.3f}s"
        )
        
        # Log performance summary
        self._log_performance_summary(total_time)
    
    def _calculate_state_changes(self, input_state: Dict, output_state: Dict) -> Dict:
        """Calculate what changed in the state."""
        changes = {}
        
        for key in output_state:
            if key not in input_state:
                changes[f"added_{key}"] = output_state[key]
            elif input_state[key] != output_state[key]:
                changes[f"modified_{key}"] = {
                    "from": input_state[key],
                    "to": output_state[key]
                }
        
        return changes
    
    def _log_performance_summary(self, total_time: float):
        """Log performance summary."""
        if self.node_timings:
            avg_node_time = sum(self.node_timings.values()) / len(self.node_timings)
            slowest_node = max(self.node_timings.items(), key=lambda x: x[1])
            
            self.logger.info(
                f"Performance summary: "
                f"Total: {total_time:.3f}s, "
                f"Avg node: {avg_node_time:.3f}s, "
                f"Slowest: {slowest_node[0]} ({slowest_node[1]:.3f}s)"
            )

def monitored_processing_node(state: ResilientState) -> ResilientState:
    """Processing node with monitoring and logging."""
    logger = GraphExecutionLogger("error_resilient_workflow")
    
    start_time = time.time()
    node_name = "monitored_processing"
    
    try:
        # Log node start
        input_state_copy = state.copy()
        
        # Main processing
        result = {
            "processed_data": f"Monitored processing of: {state['task_data']}",
            "timestamp": datetime.now().isoformat(),
            "monitoring_enabled": True
        }
        
        output_state = {
            **state,
            "processing_results": state["processing_results"] + [result],
            "system_health": {
                **state.get("system_health", {}),
                "last_successful_processing": datetime.now().isoformat()
            }
        }
        
        # Log successful execution
        execution_time = time.time() - start_time
        logger.log_node_execution(node_name, input_state_copy, output_state, execution_time)
        
        return output_state
    
    except Exception as e:
        # Log error
        logger.log_error(node_name, e, state)
        
        # Return error state
        return {
            **state,
            "error_history": state["error_history"] + [{
                "node": node_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }]
        }

# Build resilient workflow with error handling
def create_resilient_workflow():
    """Create workflow with comprehensive error handling and monitoring."""
    
    workflow = StateGraph(ResilientState)
    
    # Add resilient nodes
    workflow.add_node("safe_processing", safe_processing_node)
    workflow.add_node("fallback_coordinator", fallback_coordinator)
    workflow.add_node("graceful_degradation", graceful_degradation_node)
    workflow.add_node("monitored_processing", monitored_processing_node)
    workflow.add_node("final_health_check", lambda s: {
        **s,
        "final_system_status": "healthy" if not s.get("system_health", {}).get("critical_failure") else "degraded"
    })
    
    # Set up error-resilient flow
    workflow.set_entry_point("safe_processing")
    
    def error_router(state: ResilientState) -> str:
        if state.get("system_health", {}).get("error_threshold_exceeded"):
            return "fallback"
        elif state["retry_count"] >= state["max_retries"]:
            return "degradation"
        else:
            return "monitoring"
    
    workflow.add_conditional_edges(
        "safe_processing",
        error_router,
        {
            "fallback": "fallback_coordinator",
            "degradation": "graceful_degradation",
            "monitoring": "monitored_processing"
        }
    )
    
    workflow.add_edge("fallback_coordinator", "graceful_degradation")
    workflow.add_edge("graceful_degradation", "final_health_check")
    workflow.add_edge("monitored_processing", "final_health_check")
    workflow.add_edge("final_health_check", "__end__")
    
    return workflow.compile()

# Usage example
def test_error_resilience():
    """Test error handling and resilience patterns."""
    
    resilient_app = create_resilient_workflow()
    
    # Test with normal data
    normal_result = resilient_app.invoke({
        "task_data": "normal processing task",
        "processing_results": [],
        "error_history": [],
        "retry_count": 0,
        "max_retries": 3,
        "fallback_strategies": [],
        "current_strategy": "primary",
        "error_recovery_log": [],
        "system_health": {}
    })
    
    print("Normal Processing Result:")
    print(f"Final status: {normal_result.get('final_system_status')}")
    print(f"Processing results: {len(normal_result['processing_results'])}")
    
    # Test with error-inducing data
    error_result = resilient_app.invoke({
        "task_data": "fail processing task",
        "processing_results": [],
        "error_history": [],
        "retry_count": 0,
        "max_retries": 3,
        "fallback_strategies": [],
        "current_strategy": "primary",
        "error_recovery_log": [],
        "system_health": {}
    })
    
    print("\nError Recovery Result:")
    print(f"Final status: {error_result.get('final_system_status')}")
    print(f"Error recovery log: {len(error_result['error_recovery_log'])} entries")
    print(f"Current strategy: {error_result['current_strategy']}")

if __name__ == "__main__":
    test_error_resilience()
```

These patterns provide robust error handling, comprehensive monitoring, and graceful degradation capabilities essential for production LangGraph applications.
