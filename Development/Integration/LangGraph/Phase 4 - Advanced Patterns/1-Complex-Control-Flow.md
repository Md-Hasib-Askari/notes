# Complex Control Flow

## Dynamic Graph Construction

### Runtime Graph Modification

Dynamic graph construction allows workflows to modify themselves during execution, adapting to changing conditions and requirements.

```python
from typing import TypedDict, List, Dict, Any, Callable
from langgraph.graph import StateGraph
import copy

class DynamicState(TypedDict):
    task_complexity: str
    available_resources: List[str]
    performance_metrics: Dict[str, float]
    current_strategy: str
    graph_modifications: List[str]
    execution_path: List[str]
    adaptive_nodes: Dict[str, Callable]

class DynamicGraphBuilder:
    def __init__(self):
        self.base_workflow = None
        self.node_registry = {}
        self.modification_history = []
    
    def register_node(self, name: str, node_func: Callable):
        """Register available nodes for dynamic addition."""
        self.node_registry[name] = node_func
    
    def modify_graph_runtime(self, state: DynamicState) -> StateGraph:
        """Modify graph structure based on current state."""
        workflow = StateGraph(DynamicState)
        
        # Always include core nodes
        workflow.add_node("analyzer", self.task_analyzer)
        workflow.add_node("strategy_selector", self.strategy_selector)
        
        # Add nodes based on complexity
        complexity = state["task_complexity"]
        if complexity == "high":
            workflow.add_node("detailed_processor", self.detailed_processor)
            workflow.add_node("quality_checker", self.quality_checker)
        elif complexity == "medium":
            workflow.add_node("standard_processor", self.standard_processor)
        else:
            workflow.add_node("simple_processor", self.simple_processor)
        
        # Add resource-specific nodes
        resources = state["available_resources"]
        if "gpu" in resources:
            workflow.add_node("gpu_accelerated", self.gpu_processor)
        if "external_api" in resources:
            workflow.add_node("api_integration", self.api_processor)
        
        # Dynamic edge creation
        workflow.set_entry_point("analyzer")
        workflow.add_edge("analyzer", "strategy_selector")
        
        # Conditional routing based on complexity
        workflow.add_conditional_edges(
            "strategy_selector",
            self.dynamic_router,
            self._create_routing_map(complexity, resources)
        )
        
        return workflow
    
    def _create_routing_map(self, complexity: str, resources: List[str]) -> Dict[str, str]:
        """Create routing map based on current conditions."""
        routing = {}
        
        if complexity == "high":
            routing["detailed"] = "detailed_processor"
            routing["quality_check"] = "quality_checker"
        elif complexity == "medium":
            routing["standard"] = "standard_processor"
        else:
            routing["simple"] = "simple_processor"
        
        if "gpu" in resources:
            routing["gpu_process"] = "gpu_accelerated"
        if "external_api" in resources:
            routing["api_call"] = "api_integration"
        
        routing["end"] = "__end__"
        return routing

def adaptive_workflow_node(state: DynamicState) -> DynamicState:
    """Node that can modify the workflow during execution."""
    
    # Analyze current performance
    performance = state["performance_metrics"]
    avg_performance = sum(performance.values()) / len(performance) if performance else 0.5
    
    modifications = []
    
    # Add optimization nodes if performance is low
    if avg_performance < 0.6:
        modifications.append("Added optimization node due to low performance")
        # In practice, would modify the actual graph here
    
    # Add parallel processing if resources allow
    if "cpu_cores" in state["available_resources"]:
        modifications.append("Enabled parallel processing")
    
    # Update execution strategy
    new_strategy = "adaptive_optimization" if avg_performance < 0.7 else "standard_execution"
    
    return {
        **state,
        "current_strategy": new_strategy,
        "graph_modifications": state["graph_modifications"] + modifications,
        "execution_path": state["execution_path"] + ["adaptive_workflow"]
    }
```

### Self-Modifying Agent Behaviors

```python
class SelfModifyingAgent:
    def __init__(self):
        self.behavior_patterns = {
            "conservative": {"risk_threshold": 0.3, "exploration_rate": 0.1},
            "balanced": {"risk_threshold": 0.5, "exploration_rate": 0.3},
            "aggressive": {"risk_threshold": 0.7, "exploration_rate": 0.5}
        }
        self.current_pattern = "balanced"
        self.learning_history = []
    
    def self_modify_behavior(self, state: DynamicState) -> DynamicState:
        """Modify agent behavior based on performance feedback."""
        performance = state["performance_metrics"]
        
        # Analyze recent performance trends
        if len(self.learning_history) >= 3:
            recent_performance = self.learning_history[-3:]
            trend = self._calculate_trend(recent_performance)
            
            # Adjust behavior pattern based on trend
            if trend < -0.1:  # Performance declining
                self.current_pattern = "conservative"
            elif trend > 0.1:  # Performance improving
                self.current_pattern = "aggressive"
            else:
                self.current_pattern = "balanced"
        
        # Update behavior parameters
        current_behavior = self.behavior_patterns[self.current_pattern]
        
        # Record learning
        self.learning_history.append({
            "performance": performance,
            "pattern": self.current_pattern,
            "timestamp": len(self.learning_history)
        })
        
        return {
            **state,
            "current_strategy": f"self_modified_{self.current_pattern}",
            "adaptive_nodes": {
                **state.get("adaptive_nodes", {}),
                "behavior_modifier": current_behavior
            }
        }
    
    def _calculate_trend(self, history: List[Dict]) -> float:
        """Calculate performance trend from history."""
        if len(history) < 2:
            return 0.0
        
        performances = [h["performance"].get("overall", 0.5) for h in history]
        return (performances[-1] - performances[0]) / len(performances)
```

## Nested Subgraphs

### Hierarchical Graph Structures

```python
class NestedGraphState(TypedDict):
    main_task: str
    subtask_results: Dict[str, Any]
    subgraph_contexts: Dict[str, Dict]
    hierarchical_level: int
    coordination_data: Dict[str, Any]

def create_data_processing_subgraph() -> StateGraph:
    """Create specialized subgraph for data processing."""
    
    class DataProcessingState(TypedDict):
        raw_data: Any
        cleaned_data: Any
        processed_data: Any
        validation_results: Dict[str, bool]
    
    def data_cleaner(state: DataProcessingState) -> DataProcessingState:
        # Simulate data cleaning
        cleaned = f"cleaned_{state['raw_data']}" if state['raw_data'] else None
        return {**state, "cleaned_data": cleaned}
    
    def data_processor(state: DataProcessingState) -> DataProcessingState:
        # Simulate data processing
        processed = f"processed_{state['cleaned_data']}" if state['cleaned_data'] else None
        return {**state, "processed_data": processed}
    
    def data_validator(state: DataProcessingState) -> DataProcessingState:
        # Simulate validation
        validation = {
            "data_quality": True,
            "completeness": True,
            "format_correct": True
        }
        return {**state, "validation_results": validation}
    
    # Build subgraph
    subgraph = StateGraph(DataProcessingState)
    subgraph.add_node("clean", data_cleaner)
    subgraph.add_node("process", data_processor)
    subgraph.add_node("validate", data_validator)
    
    subgraph.set_entry_point("clean")
    subgraph.add_edge("clean", "process")
    subgraph.add_edge("process", "validate")
    subgraph.add_edge("validate", "__end__")
    
    return subgraph.compile()

def create_analysis_subgraph() -> StateGraph:
    """Create specialized subgraph for analysis."""
    
    class AnalysisState(TypedDict):
        input_data: Any
        statistical_analysis: Dict[str, float]
        pattern_analysis: Dict[str, Any]
        insights: List[str]
    
    def statistical_analyzer(state: AnalysisState) -> AnalysisState:
        # Simulate statistical analysis
        stats = {
            "mean": 0.75,
            "std_dev": 0.15,
            "confidence": 0.85
        }
        return {**state, "statistical_analysis": stats}
    
    def pattern_analyzer(state: AnalysisState) -> AnalysisState:
        # Simulate pattern analysis
        patterns = {
            "trend": "increasing",
            "seasonality": "moderate",
            "anomalies": ["outlier_1", "outlier_2"]
        }
        return {**state, "pattern_analysis": patterns}
    
    def insight_generator(state: AnalysisState) -> AnalysisState:
        # Generate insights
        insights = [
            "Strong upward trend detected",
            "Seasonal patterns are consistent",
            "Few anomalies require attention"
        ]
        return {**state, "insights": insights}
    
    # Build analysis subgraph
    subgraph = StateGraph(AnalysisState)
    subgraph.add_node("stats", statistical_analyzer)
    subgraph.add_node("patterns", pattern_analyzer)
    subgraph.add_node("insights", insight_generator)
    
    subgraph.set_entry_point("stats")
    subgraph.add_edge("stats", "patterns")
    subgraph.add_edge("patterns", "insights")
    subgraph.add_edge("insights", "__end__")
    
    return subgraph.compile()

def subgraph_coordinator(state: NestedGraphState) -> NestedGraphState:
    """Coordinate execution of nested subgraphs."""
    
    # Initialize subgraph contexts
    data_context = {
        "raw_data": state["main_task"],
        "cleaned_data": None,
        "processed_data": None,
        "validation_results": {}
    }
    
    analysis_context = {
        "input_data": None,  # Will be populated from data processing
        "statistical_analysis": {},
        "pattern_analysis": {},
        "insights": []
    }
    
    return {
        **state,
        "subgraph_contexts": {
            "data_processing": data_context,
            "analysis": analysis_context
        },
        "hierarchical_level": state["hierarchical_level"] + 1
    }

def execute_subgraph_sequence(state: NestedGraphState) -> NestedGraphState:
    """Execute subgraphs in sequence with context isolation."""
    
    results = {}
    
    # Execute data processing subgraph
    data_subgraph = create_data_processing_subgraph()
    data_context = state["subgraph_contexts"]["data_processing"]
    data_result = data_subgraph.invoke(data_context)
    results["data_processing"] = data_result
    
    # Execute analysis subgraph with data processing results
    analysis_subgraph = create_analysis_subgraph()
    analysis_context = state["subgraph_contexts"]["analysis"]
    analysis_context["input_data"] = data_result["processed_data"]
    analysis_result = analysis_subgraph.invoke(analysis_context)
    results["analysis"] = analysis_result
    
    # Coordinate results
    coordination_data = {
        "data_quality": data_result["validation_results"],
        "analysis_insights": analysis_result["insights"],
        "processing_chain": ["data_processing", "analysis"],
        "completion_status": "all_subgraphs_completed"
    }
    
    return {
        **state,
        "subtask_results": results,
        "coordination_data": coordination_data
    }
```

### Context Isolation Techniques

```python
class IsolatedContext:
    """Provides context isolation for subgraphs."""
    
    def __init__(self, context_id: str, parent_context: Dict = None):
        self.context_id = context_id
        self.local_state = {}
        self.parent_context = parent_context or {}
        self.isolation_level = "strict"  # "strict", "partial", "shared"
    
    def get_isolated_state(self, full_state: Dict) -> Dict:
        """Extract isolated state for subgraph execution."""
        if self.isolation_level == "strict":
            # Only pass explicitly allowed data
            return {
                "context_id": self.context_id,
                "local_data": self.local_state,
                "allowed_parent_data": self._get_allowed_parent_data()
            }
        elif self.isolation_level == "partial":
            # Pass most data but isolate sensitive information
            isolated = full_state.copy()
            isolated.pop("sensitive_data", None)
            isolated["context_id"] = self.context_id
            return isolated
        else:  # shared
            # Pass all data
            return {**full_state, "context_id": self.context_id}
    
    def _get_allowed_parent_data(self) -> Dict:
        """Get data allowed to pass from parent context."""
        allowed_keys = ["task_id", "user_id", "permissions"]
        return {k: v for k, v in self.parent_context.items() if k in allowed_keys}
    
    def merge_results_to_parent(self, subgraph_result: Dict, parent_state: Dict) -> Dict:
        """Safely merge subgraph results back to parent state."""
        # Extract only safe results
        safe_results = {
            "subgraph_output": subgraph_result.get("final_output"),
            "subgraph_metadata": {
                "context_id": self.context_id,
                "execution_time": subgraph_result.get("execution_time"),
                "success": subgraph_result.get("success", False)
            }
        }
        
        return {
            **parent_state,
            f"subgraph_results_{self.context_id}": safe_results
        }

def context_isolated_subgraph_executor(state: NestedGraphState) -> NestedGraphState:
    """Execute subgraph with proper context isolation."""
    
    # Create isolated contexts for different subgraphs
    data_context = IsolatedContext("data_processing", state)
    analysis_context = IsolatedContext("analysis", state)
    
    # Set different isolation levels
    data_context.isolation_level = "partial"  # Needs some parent data
    analysis_context.isolation_level = "strict"  # Highly isolated
    
    # Execute with isolation
    results = {}
    
    # Data processing with partial isolation
    data_isolated_state = data_context.get_isolated_state(state)
    data_subgraph = create_data_processing_subgraph()
    data_result = data_subgraph.invoke(data_isolated_state)
    results["data"] = data_context.merge_results_to_parent(data_result, state)
    
    # Analysis with strict isolation
    analysis_isolated_state = analysis_context.get_isolated_state(state)
    analysis_subgraph = create_analysis_subgraph()
    analysis_result = analysis_subgraph.invoke(analysis_isolated_state)
    results["analysis"] = analysis_context.merge_results_to_parent(analysis_result, state)
    
    return {
        **state,
        "subtask_results": results,
        "isolation_summary": {
            "contexts_created": ["data_processing", "analysis"],
            "isolation_levels": {
                "data_processing": "partial",
                "analysis": "strict"
            }
        }
    }

# Build main workflow with nested subgraphs
def create_hierarchical_workflow():
    """Create main workflow that coordinates nested subgraphs."""
    
    workflow = StateGraph(NestedGraphState)
    
    # Add coordination nodes
    workflow.add_node("coordinator", subgraph_coordinator)
    workflow.add_node("execute_subgraphs", execute_subgraph_sequence)
    workflow.add_node("isolated_execution", context_isolated_subgraph_executor)
    workflow.add_node("result_aggregator", lambda s: {
        **s, 
        "final_result": f"Completed hierarchical processing for: {s['main_task']}"
    })
    
    # Set up workflow
    workflow.set_entry_point("coordinator")
    workflow.add_edge("coordinator", "execute_subgraphs")
    workflow.add_edge("execute_subgraphs", "isolated_execution")
    workflow.add_edge("isolated_execution", "result_aggregator")
    workflow.add_edge("result_aggregator", "__end__")
    
    return workflow.compile()

# Usage example
def test_complex_control_flow():
    """Test dynamic graphs and nested subgraphs."""
    
    # Test hierarchical workflow
    hierarchical_app = create_hierarchical_workflow()
    
    result = hierarchical_app.invoke({
        "main_task": "Process customer feedback data",
        "subtask_results": {},
        "subgraph_contexts": {},
        "hierarchical_level": 0,
        "coordination_data": {}
    })
    
    print("Hierarchical Workflow Result:")
    print(f"Final result: {result['final_result']}")
    print(f"Subtasks completed: {len(result['subtask_results'])}")
    print(f"Hierarchical level: {result['hierarchical_level']}")
    print(f"Coordination status: {result['coordination_data'].get('completion_status')}")

if __name__ == "__main__":
    test_complex_control_flow()
```

These advanced control flow patterns enable building highly sophisticated, adaptive systems that can modify themselves at runtime and manage complex hierarchical processing with proper isolation and coordination.
