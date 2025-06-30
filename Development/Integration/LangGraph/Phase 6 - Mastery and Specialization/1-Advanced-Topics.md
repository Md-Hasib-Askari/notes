# Advanced Topics - Phase 6

Master-level concepts for building sophisticated, reusable, and highly optimized LangGraph components and analytics systems.

## Custom Node Types

### Building Reusable Node Libraries

```python
# src/nodes/library.py
from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Any, Callable, Optional, List
import asyncio
import time
from dataclasses import dataclass
from enum import Enum

class NodeCategory(Enum):
    DATA_PROCESSING = "data_processing"
    AI_REASONING = "ai_reasoning"
    INTEGRATION = "integration"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"

@dataclass
class NodeMetadata:
    name: str
    description: str
    category: NodeCategory
    version: str
    author: str
    dependencies: List[str]
    performance_profile: Dict[str, Any]

class BaseCustomNode(ABC):
    """Base class for all custom nodes"""
    
    def __init__(self, metadata: NodeMetadata):
        self.metadata = metadata
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node logic"""
        pass
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for execution with performance tracking"""
        start_time = time.time()
        self.execution_count += 1
        
        try:
            result = await self.execute(state)
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            # Add execution metadata
            result["_node_metadata"] = {
                "node_name": self.metadata.name,
                "execution_time": execution_time,
                "execution_count": self.execution_count
            }
            
            return result
            
        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            # Add error metadata
            error_result = {
                **state,
                "_node_error": {
                    "node_name": self.metadata.name,
                    "error_message": str(e),
                    "execution_time": execution_time,
                    "error_count": self.error_count
                }
            }
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_execution_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 else 0
        )
        
        return {
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.execution_count, 1)
        }

class DataValidationNode(BaseCustomNode):
    """Reusable data validation node"""
    
    def __init__(self, validation_schema: Dict[str, Any]):
        metadata = NodeMetadata(
            name="DataValidator",
            description="Validates data against a schema",
            category=NodeCategory.VALIDATION,
            version="1.0.0",
            author="LangGraph Team",
            dependencies=["pydantic"],
            performance_profile={"expected_latency": "< 10ms", "memory_usage": "low"}
        )
        super().__init__(metadata)
        self.validation_schema = validation_schema
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state data against schema"""
        validation_errors = []
        
        for field, requirements in self.validation_schema.items():
            if field not in state:
                if requirements.get("required", False):
                    validation_errors.append(f"Required field '{field}' is missing")
                continue
            
            value = state[field]
            field_type = requirements.get("type")
            
            if field_type and not isinstance(value, field_type):
                validation_errors.append(f"Field '{field}' must be of type {field_type.__name__}")
            
            min_value = requirements.get("min")
            if min_value is not None and value < min_value:
                validation_errors.append(f"Field '{field}' must be >= {min_value}")
            
            max_value = requirements.get("max")
            if max_value is not None and value > max_value:
                validation_errors.append(f"Field '{field}' must be <= {max_value}")
        
        return {
            **state,
            "validation_passed": len(validation_errors) == 0,
            "validation_errors": validation_errors
        }

class LLMReasoningNode(BaseCustomNode):
    """High-performance LLM reasoning node"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        metadata = NodeMetadata(
            name="LLMReasoner",
            description="Optimized LLM reasoning with caching",
            category=NodeCategory.AI_REASONING,
            version="2.0.0",
            author="LangGraph Team",
            dependencies=["langchain", "openai"],
            performance_profile={"expected_latency": "< 2s", "memory_usage": "medium"}
        )
        super().__init__(metadata)
        self.llm_config = llm_config
        self.response_cache = {}
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM reasoning with caching and optimization"""
        prompt = state.get("prompt", "")
        context = state.get("context", {})
        
        # Create cache key
        cache_key = hash(f"{prompt}:{str(sorted(context.items()))}")
        
        # Check cache first
        if cache_key in self.response_cache:
            return {
                **state,
                "llm_response": self.response_cache[cache_key],
                "cache_hit": True
            }
        
        # Execute LLM reasoning (simulated)
        reasoning_result = await self._execute_llm_reasoning(prompt, context)
        
        # Cache the result
        self.response_cache[cache_key] = reasoning_result
        
        return {
            **state,
            "llm_response": reasoning_result,
            "cache_hit": False
        }
    
    async def _execute_llm_reasoning(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LLM reasoning"""
        await asyncio.sleep(0.5)  # Simulate API call
        
        return {
            "reasoning": f"Analyzed prompt: {prompt[:50]}...",
            "confidence": 0.85,
            "tokens_used": len(prompt.split()) * 1.3,
            "reasoning_steps": ["analysis", "synthesis", "conclusion"]
        }

# Node library registry
class NodeLibrary:
    def __init__(self):
        self.nodes: Dict[str, BaseCustomNode] = {}
        self.categories: Dict[NodeCategory, List[str]] = {}
    
    def register_node(self, node: BaseCustomNode):
        """Register a node in the library"""
        self.nodes[node.metadata.name] = node
        
        category = node.metadata.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(node.metadata.name)
    
    def get_node(self, name: str) -> Optional[BaseCustomNode]:
        """Get a node by name"""
        return self.nodes.get(name)
    
    def list_nodes_by_category(self, category: NodeCategory) -> List[str]:
        """List all nodes in a category"""
        return self.categories.get(category, [])
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get overall library statistics"""
        total_executions = sum(node.execution_count for node in self.nodes.values())
        total_errors = sum(node.error_count for node in self.nodes.values())
        
        performance_by_node = {
            name: node.get_performance_stats() 
            for name, node in self.nodes.items()
        }
        
        return {
            "total_nodes": len(self.nodes),
            "total_executions": total_executions,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / max(total_executions, 1),
            "nodes_by_category": {cat.value: len(nodes) for cat, nodes in self.categories.items()},
            "performance_by_node": performance_by_node
        }

# Global node library instance
node_library = NodeLibrary()

# Register common nodes
validation_schema = {
    "user_id": {"type": str, "required": True},
    "score": {"type": float, "min": 0.0, "max": 1.0}
}
node_library.register_node(DataValidationNode(validation_schema))
node_library.register_node(LLMReasoningNode({"model": "gpt-4", "temperature": 0.7}))
```

## Graph Analytics

### Execution Path Analysis

```python
# src/analytics/execution_analyzer.py
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict, deque

@dataclass
class ExecutionStep:
    node_name: str
    start_time: datetime
    end_time: datetime
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    execution_time: float
    memory_usage: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class ExecutionTrace:
    execution_id: str
    graph_id: str
    start_time: datetime
    end_time: Optional[datetime]
    steps: List[ExecutionStep] = field(default_factory=list)
    total_execution_time: float = 0.0
    success: bool = True
    final_state: Dict[str, Any] = field(default_factory=dict)

class ExecutionPathAnalyzer:
    def __init__(self):
        self.execution_traces: Dict[str, ExecutionTrace] = {}
        self.path_frequencies: Dict[str, int] = defaultdict(int)
        self.node_performance: Dict[str, List[float]] = defaultdict(list)
        self.error_patterns: Dict[str, int] = defaultdict(int)
    
    def record_execution(self, trace: ExecutionTrace):
        """Record a complete execution trace"""
        self.execution_traces[trace.execution_id] = trace
        
        # Extract execution path
        path = " -> ".join(step.node_name for step in trace.steps)
        self.path_frequencies[path] += 1
        
        # Record node performance
        for step in trace.steps:
            self.node_performance[step.node_name].append(step.execution_time)
            
            if not step.success:
                error_pattern = f"{step.node_name}:{step.error_message}"
                self.error_patterns[error_pattern] += 1
    
    def analyze_execution_paths(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze execution paths within a time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_traces = [
            trace for trace in self.execution_traces.values()
            if trace.start_time > cutoff_time
        ]
        
        if not recent_traces:
            return {"message": "No recent executions found"}
        
        # Path frequency analysis
        path_stats = {}
        for trace in recent_traces:
            path = " -> ".join(step.node_name for step in trace.steps)
            if path not in path_stats:
                path_stats[path] = {
                    "frequency": 0,
                    "avg_execution_time": 0,
                    "success_rate": 0,
                    "execution_times": []
                }
            
            path_stats[path]["frequency"] += 1
            path_stats[path]["execution_times"].append(trace.total_execution_time)
            
            if trace.success:
                path_stats[path]["success_rate"] += 1
        
        # Calculate averages
        for path_data in path_stats.values():
            execution_times = path_data["execution_times"]
            path_data["avg_execution_time"] = sum(execution_times) / len(execution_times)
            path_data["success_rate"] = path_data["success_rate"] / path_data["frequency"]
            del path_data["execution_times"]  # Remove raw data
        
        return {
            "analysis_period": f"Last {time_window_hours} hours",
            "total_executions": len(recent_traces),
            "unique_paths": len(path_stats),
            "most_common_paths": sorted(
                path_stats.items(), 
                key=lambda x: x[1]["frequency"], 
                reverse=True
            )[:10],
            "path_statistics": path_stats
        }
    
    def identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks in the graph"""
        bottlenecks = {}
        
        for node_name, execution_times in self.node_performance.items():
            if len(execution_times) < 5:  # Need sufficient data
                continue
            
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            std_dev = (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
            
            # Identify bottleneck criteria
            is_bottleneck = (
                avg_time > 2.0 or  # Average execution > 2 seconds
                max_time > 10.0 or  # Any execution > 10 seconds
                std_dev > avg_time * 0.5  # High variability
            )
            
            if is_bottleneck:
                bottlenecks[node_name] = {
                    "average_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time,
                    "std_deviation": std_dev,
                    "execution_count": len(execution_times),
                    "bottleneck_reasons": []
                }
                
                # Classify bottleneck type
                if avg_time > 2.0:
                    bottlenecks[node_name]["bottleneck_reasons"].append("high_average_time")
                if max_time > 10.0:
                    bottlenecks[node_name]["bottleneck_reasons"].append("extreme_outliers")
                if std_dev > avg_time * 0.5:
                    bottlenecks[node_name]["bottleneck_reasons"].append("high_variability")
        
        return {
            "bottleneck_nodes": bottlenecks,
            "recommendations": self._generate_optimization_recommendations(bottlenecks)
        }
    
    def _generate_optimization_recommendations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks"""
        recommendations = []
        
        for node_name, stats in bottlenecks.items():
            reasons = stats["bottleneck_reasons"]
            
            if "high_average_time" in reasons:
                recommendations.append(f"Optimize {node_name}: Consider caching or algorithm optimization")
            
            if "extreme_outliers" in reasons:
                recommendations.append(f"Add timeout handling to {node_name}: Implement circuit breaker pattern")
            
            if "high_variability" in reasons:
                recommendations.append(f"Stabilize {node_name}: Investigate inconsistent performance causes")
        
        return recommendations
    
    def generate_execution_graph(self, execution_id: str) -> nx.DiGraph:
        """Generate a NetworkX graph from an execution trace"""
        if execution_id not in self.execution_traces:
            raise ValueError(f"Execution {execution_id} not found")
        
        trace = self.execution_traces[execution_id]
        graph = nx.DiGraph()
        
        # Add nodes with execution metrics
        for step in trace.steps:
            graph.add_node(step.node_name, 
                          execution_time=step.execution_time,
                          memory_usage=step.memory_usage,
                          success=step.success)
        
        # Add edges representing execution flow
        for i in range(len(trace.steps) - 1):
            current_step = trace.steps[i]
            next_step = trace.steps[i + 1]
            graph.add_edge(current_step.node_name, next_step.node_name,
                          transition_time=next_step.start_time - current_step.end_time)
        
        return graph
    
    def compare_execution_patterns(self, execution_ids: List[str]) -> Dict[str, Any]:
        """Compare execution patterns across multiple runs"""
        if not execution_ids:
            return {"error": "No execution IDs provided"}
        
        traces = [self.execution_traces.get(eid) for eid in execution_ids]
        traces = [t for t in traces if t is not None]
        
        if not traces:
            return {"error": "No valid traces found"}
        
        # Compare execution paths
        paths = [" -> ".join(step.node_name for step in trace.steps) for trace in traces]
        unique_paths = set(paths)
        
        # Compare performance metrics
        performance_comparison = {}
        for trace in traces:
            for step in trace.steps:
                node_name = step.node_name
                if node_name not in performance_comparison:
                    performance_comparison[node_name] = []
                performance_comparison[node_name].append(step.execution_time)
        
        # Calculate statistics
        node_stats = {}
        for node_name, times in performance_comparison.items():
            node_stats[node_name] = {
                "min_time": min(times),
                "max_time": max(times),
                "avg_time": sum(times) / len(times),
                "executions": len(times)
            }
        
        return {
            "compared_executions": len(traces),
            "unique_paths": len(unique_paths),
            "path_consistency": len(unique_paths) == 1,
            "performance_comparison": node_stats,
            "execution_paths": list(unique_paths)
        }

# Usage example
analyzer = ExecutionPathAnalyzer()

# Create sample execution trace
sample_trace = ExecutionTrace(
    execution_id="exec_123",
    graph_id="workflow_graph",
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(seconds=5),
    total_execution_time=5.0
)

# Add execution steps
sample_trace.steps = [
    ExecutionStep(
        node_name="validator",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(seconds=1),
        input_state={"data": "test"},
        output_state={"validated": True},
        execution_time=1.0,
        memory_usage=1024,
        success=True
    ),
    ExecutionStep(
        node_name="processor",
        start_time=datetime.now() + timedelta(seconds=1),
        end_time=datetime.now() + timedelta(seconds=4),
        input_state={"validated": True},
        output_state={"processed": True},
        execution_time=3.0,
        memory_usage=2048,
        success=True
    )
]

analyzer.record_execution(sample_trace)
```

## Performance Profiling

### Advanced Profiling Tools

```python
# src/analytics/profiler.py
import cProfile
import pstats
import io
from typing import Dict, Any, List
from contextlib import contextmanager
import memory_profiler
import time
import asyncio

class LangGraphProfiler:
    def __init__(self):
        self.profiling_data = {}
        self.memory_snapshots = {}
        self.performance_metrics = defaultdict(list)
    
    @contextmanager
    def profile_execution(self, execution_id: str):
        """Context manager for profiling graph execution"""
        profiler = cProfile.Profile()
        start_time = time.time()
        initial_memory = memory_profiler.memory_usage()[0]
        
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            end_time = time.time()
            final_memory = memory_profiler.memory_usage()[0]
            
            # Store profiling results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative').print_stats(20)
            
            self.profiling_data[execution_id] = {
                "execution_time": end_time - start_time,
                "memory_delta": final_memory - initial_memory,
                "cpu_profile": s.getvalue(),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_performance_report(self, execution_ids: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if execution_ids is None:
            execution_ids = list(self.profiling_data.keys())
        
        report_data = []
        for exec_id in execution_ids:
            if exec_id in self.profiling_data:
                report_data.append(self.profiling_data[exec_id])
        
        if not report_data:
            return {"message": "No profiling data available"}
        
        # Calculate aggregate statistics
        total_executions = len(report_data)
        avg_execution_time = sum(d["execution_time"] for d in report_data) / total_executions
        avg_memory_usage = sum(d["memory_delta"] for d in report_data) / total_executions
        
        return {
            "total_executions_analyzed": total_executions,
            "average_execution_time": avg_execution_time,
            "average_memory_delta": avg_memory_usage,
            "performance_trends": self._analyze_performance_trends(report_data),
            "optimization_suggestions": self._generate_optimization_suggestions(report_data)
        }
    
    def _analyze_performance_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        execution_times = [d["execution_time"] for d in data]
        memory_deltas = [d["memory_delta"] for d in data]
        
        return {
            "execution_time_trend": {
                "min": min(execution_times),
                "max": max(execution_times),
                "trend": "improving" if execution_times[0] > execution_times[-1] else "degrading"
            },
            "memory_usage_trend": {
                "min": min(memory_deltas),
                "max": max(memory_deltas),
                "average": sum(memory_deltas) / len(memory_deltas)
            }
        }
    
    def _generate_optimization_suggestions(self, data: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization suggestions based on profiling data"""
        suggestions = []
        
        avg_time = sum(d["execution_time"] for d in data) / len(data)
        avg_memory = sum(d["memory_delta"] for d in data) / len(data)
        
        if avg_time > 5.0:
            suggestions.append("Consider implementing caching for frequently accessed data")
            suggestions.append("Optimize algorithm complexity in bottleneck nodes")
        
        if avg_memory > 100:  # MB
            suggestions.append("Implement memory-efficient data structures")
            suggestions.append("Add garbage collection triggers for large objects")
        
        return suggestions

# Integration with execution analyzer
profiler = LangGraphProfiler()

async def profile_graph_execution(graph, initial_state):
    """Profile a complete graph execution"""
    execution_id = f"profile_{datetime.now().timestamp()}"
    
    with profiler.profile_execution(execution_id):
        result = await graph.ainvoke(initial_state)
    
    return result, execution_id
```

## Best Practices

1. **Node Design**: Create modular, reusable nodes with clear interfaces
2. **Performance Tracking**: Implement comprehensive performance monitoring
3. **Analytics Integration**: Use execution path analysis to optimize workflows
4. **Profiling**: Regular profiling to identify optimization opportunities
5. **Library Management**: Maintain organized node libraries with proper versioning
6. **Documentation**: Document custom nodes with usage examples and performance characteristics
