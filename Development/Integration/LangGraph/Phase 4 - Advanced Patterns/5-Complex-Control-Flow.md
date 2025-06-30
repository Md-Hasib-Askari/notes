# Complex Control Flow in LangGraph

Advanced patterns for building dynamic, adaptive graph structures that can modify themselves at runtime.

## Dynamic Graph Construction

### Runtime Graph Modification

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Callable
import asyncio

class DynamicState(TypedDict):
    task_type: str
    data: dict
    graph_config: dict
    results: List[dict]

class DynamicGraphBuilder:
    def __init__(self):
        self.node_registry = {}
        self.condition_registry = {}
    
    def register_node(self, name: str, func: Callable):
        """Register a node function for dynamic use"""
        self.node_registry[name] = func
    
    def build_graph(self, config: dict) -> StateGraph:
        """Build graph based on runtime configuration"""
        graph = StateGraph(DynamicState)
        
        # Add nodes based on config
        for node_config in config.get("nodes", []):
            name = node_config["name"]
            func = self.node_registry[node_config["type"]]
            graph.add_node(name, func)
        
        # Add edges based on config
        for edge_config in config.get("edges", []):
            if edge_config.get("condition"):
                condition_func = self.condition_registry[edge_config["condition"]]
                graph.add_conditional_edges(
                    edge_config["from"],
                    condition_func,
                    edge_config["mapping"]
                )
            else:
                graph.add_edge(edge_config["from"], edge_config["to"])
        
        graph.set_entry_point(config["entry_point"])
        return graph.compile()

# Example dynamic nodes
async def data_processor(state: DynamicState) -> DynamicState:
    """Process data based on task type"""
    task_type = state["task_type"]
    
    if task_type == "analysis":
        result = {"analysis": f"Analyzed {len(state['data'])} items"}
    elif task_type == "transformation":
        result = {"transformed": state["data"]}
    else:
        result = {"processed": True}
    
    return {
        **state,
        "results": state["results"] + [result]
    }

async def validator(state: DynamicState) -> DynamicState:
    """Validate results"""
    valid = len(state["results"]) > 0
    return {
        **state,
        "validation_passed": valid
    }

# Dynamic condition
def needs_reprocessing(state: DynamicState) -> str:
    """Decide if reprocessing is needed"""
    if not state.get("validation_passed", False):
        return "reprocess"
    return "complete"
```

### Adaptive Workflow Patterns

```python
class AdaptiveWorkflow:
    def __init__(self):
        self.performance_metrics = {}
        self.builder = DynamicGraphBuilder()
        self.setup_nodes()
    
    def setup_nodes(self):
        """Register adaptive nodes"""
        self.builder.register_node("fast_processor", self.fast_process)
        self.builder.register_node("thorough_processor", self.thorough_process)
        self.builder.register_node("performance_monitor", self.monitor_performance)
        self.builder.condition_registry["select_processor"] = self.select_processor
    
    async def fast_process(self, state: DynamicState) -> DynamicState:
        """Quick processing with basic validation"""
        start_time = asyncio.get_event_loop().time()
        # Simulate fast processing
        await asyncio.sleep(0.1)
        end_time = asyncio.get_event_loop().time()
        
        self.performance_metrics["fast"] = end_time - start_time
        
        return {
            **state,
            "results": state["results"] + [{"method": "fast", "quality": 0.7}]
        }
    
    async def thorough_process(self, state: DynamicState) -> DynamicState:
        """Comprehensive processing with high accuracy"""
        start_time = asyncio.get_event_loop().time()
        # Simulate thorough processing
        await asyncio.sleep(0.5)
        end_time = asyncio.get_event_loop().time()
        
        self.performance_metrics["thorough"] = end_time - start_time
        
        return {
            **state,
            "results": state["results"] + [{"method": "thorough", "quality": 0.95}]
        }
    
    async def monitor_performance(self, state: DynamicState) -> DynamicState:
        """Monitor and adapt based on performance"""
        avg_time = sum(self.performance_metrics.values()) / len(self.performance_metrics)
        
        # Adapt graph configuration based on performance
        if avg_time > 0.3:  # Too slow, prefer fast processing
            new_config = {
                "nodes": [
                    {"name": "processor", "type": "fast_processor"},
                    {"name": "monitor", "type": "performance_monitor"}
                ],
                "edges": [{"from": "processor", "to": "monitor"}],
                "entry_point": "processor"
            }
        else:  # Good performance, can use thorough processing
            new_config = {
                "nodes": [
                    {"name": "processor", "type": "thorough_processor"},
                    {"name": "monitor", "type": "performance_monitor"}
                ],
                "edges": [{"from": "processor", "to": "monitor"}],
                "entry_point": "processor"
            }
        
        return {
            **state,
            "graph_config": new_config,
            "performance_metrics": self.performance_metrics
        }
    
    def select_processor(self, state: DynamicState) -> str:
        """Dynamically select processing method"""
        if state.get("priority") == "speed":
            return "fast"
        elif state.get("priority") == "quality":
            return "thorough"
        return "balanced"
```

## Nested Subgraphs

### Hierarchical Graph Structures

```python
from langgraph.graph import StateGraph

class SubgraphState(TypedDict):
    subtask_id: str
    input_data: dict
    output_data: dict
    metadata: dict

class HierarchicalSystem:
    def __init__(self):
        self.subgraphs = {}
    
    def create_analysis_subgraph(self) -> StateGraph:
        """Create a specialized analysis subgraph"""
        subgraph = StateGraph(SubgraphState)
        
        subgraph.add_node("extract", self.extract_features)
        subgraph.add_node("analyze", self.analyze_features)
        subgraph.add_node("summarize", self.summarize_analysis)
        
        subgraph.add_edge("extract", "analyze")
        subgraph.add_edge("analyze", "summarize")
        subgraph.set_entry_point("extract")
        
        return subgraph.compile()
    
    def create_processing_subgraph(self) -> StateGraph:
        """Create a data processing subgraph"""
        subgraph = StateGraph(SubgraphState)
        
        subgraph.add_node("validate", self.validate_data)
        subgraph.add_node("transform", self.transform_data)
        subgraph.add_node("enrich", self.enrich_data)
        
        subgraph.add_edge("validate", "transform")
        subgraph.add_conditional_edges(
            "transform",
            self.needs_enrichment,
            {"enrich": "enrich", "skip": END}
        )
        subgraph.set_entry_point("validate")
        
        return subgraph.compile()
    
    async def extract_features(self, state: SubgraphState) -> SubgraphState:
        """Extract features from input data"""
        features = {
            "text_length": len(str(state["input_data"])),
            "data_type": type(state["input_data"]).__name__
        }
        
        return {
            **state,
            "output_data": {"features": features}
        }
    
    async def analyze_features(self, state: SubgraphState) -> SubgraphState:
        """Analyze extracted features"""
        features = state["output_data"]["features"]
        analysis = {
            "complexity": "high" if features["text_length"] > 1000 else "low",
            "processing_time": features["text_length"] * 0.001
        }
        
        return {
            **state,
            "output_data": {**state["output_data"], "analysis": analysis}
        }
    
    async def summarize_analysis(self, state: SubgraphState) -> SubgraphState:
        """Summarize analysis results"""
        summary = {
            "status": "complete",
            "recommendations": ["optimize", "validate"] if 
                state["output_data"]["analysis"]["complexity"] == "high" else ["proceed"]
        }
        
        return {
            **state,
            "output_data": {**state["output_data"], "summary": summary}
        }
```

### Subgraph Composition

```python
class MasterOrchestrator:
    def __init__(self):
        self.hierarchical_system = HierarchicalSystem()
        self.master_graph = self.build_master_graph()
    
    def build_master_graph(self) -> StateGraph:
        """Build the master graph that orchestrates subgraphs"""
        graph = StateGraph(DynamicState)
        
        graph.add_node("route_task", self.route_task)
        graph.add_node("run_analysis", self.run_analysis_subgraph)
        graph.add_node("run_processing", self.run_processing_subgraph)
        graph.add_node("aggregate", self.aggregate_results)
        
        graph.add_conditional_edges(
            "route_task",
            self.task_router,
            {
                "analysis": "run_analysis",
                "processing": "run_processing",
                "both": "run_analysis"  # Start with analysis
            }
        )
        
        graph.add_conditional_edges(
            "run_analysis",
            self.next_step,
            {
                "processing": "run_processing",
                "aggregate": "aggregate"
            }
        )
        
        graph.add_edge("run_processing", "aggregate")
        graph.set_entry_point("route_task")
        
        return graph.compile()
    
    async def route_task(self, state: DynamicState) -> DynamicState:
        """Route tasks to appropriate subgraphs"""
        task_requirements = self.analyze_requirements(state["data"])
        
        return {
            **state,
            "routing_decision": task_requirements,
            "subgraph_results": []
        }
    
    async def run_analysis_subgraph(self, state: DynamicState) -> DynamicState:
        """Execute the analysis subgraph"""
        analysis_graph = self.hierarchical_system.create_analysis_subgraph()
        
        subgraph_state = SubgraphState(
            subtask_id="analysis_1",
            input_data=state["data"],
            output_data={},
            metadata={"parent_task": state.get("task_type")}
        )
        
        result = await analysis_graph.ainvoke(subgraph_state)
        
        return {
            **state,
            "subgraph_results": state["subgraph_results"] + [result]
        }
    
    def task_router(self, state: DynamicState) -> str:
        """Route based on task requirements"""
        requirements = state["routing_decision"]
        
        if requirements["needs_analysis"] and requirements["needs_processing"]:
            return "both"
        elif requirements["needs_analysis"]:
            return "analysis"
        else:
            return "processing"
```

## Best Practices

### Context Isolation
- Use separate state types for subgraphs
- Implement clear data contracts between graph levels
- Maintain metadata for tracking graph hierarchy

### Performance Optimization
- Cache compiled subgraphs for reuse
- Monitor execution paths and optimize hot paths
- Use conditional routing to avoid unnecessary processing

### Testing Strategies
- Unit test individual subgraphs separately
- Integration test full hierarchical flows
- Performance test dynamic graph construction overhead

### Error Handling
- Implement fallback graphs for critical paths
- Use circuit breaker patterns for unstable subgraphs
- Maintain execution context for debugging complex flows
