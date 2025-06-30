# Performance Optimization in LangGraph

Production-ready techniques for optimizing LangGraph performance, focusing on efficient design patterns and intelligent caching strategies.

## Efficient Graph Design

### Minimizing Unnecessary State Copies

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any
import copy
from dataclasses import dataclass
import weakref

class OptimizedState(TypedDict):
    # Use references for large objects
    large_data_ref: str  # Reference ID instead of actual data
    processing_flags: Dict[str, bool]
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class StateManager:
    """Manages shared state objects to minimize copies"""
    _data_store: Dict[str, Any] = None
    _ref_counter: Dict[str, int] = None
    
    def __post_init__(self):
        if self._data_store is None:
            self._data_store = {}
        if self._ref_counter is None:
            self._ref_counter = {}
    
    def store_data(self, data: Any) -> str:
        """Store data and return reference ID"""
        ref_id = f"ref_{len(self._data_store)}"
        self._data_store[ref_id] = data
        self._ref_counter[ref_id] = 1
        return ref_id
    
    def get_data(self, ref_id: str) -> Any:
        """Retrieve data by reference ID"""
        return self._data_store.get(ref_id)
    
    def increment_ref(self, ref_id: str):
        """Increment reference counter"""
        self._ref_counter[ref_id] = self._ref_counter.get(ref_id, 0) + 1
    
    def decrement_ref(self, ref_id: str):
        """Decrement reference counter and cleanup if needed"""
        if ref_id in self._ref_counter:
            self._ref_counter[ref_id] -= 1
            if self._ref_counter[ref_id] <= 0:
                self._data_store.pop(ref_id, None)
                self._ref_counter.pop(ref_id, None)

# Global state manager instance
state_manager = StateManager()

# Efficient node implementations
async def efficient_processor(state: OptimizedState) -> OptimizedState:
    """Process data without creating unnecessary copies"""
    # Get data by reference
    data = state_manager.get_data(state["large_data_ref"])
    
    # Process without copying large objects
    processed_result = {
        "summary": f"Processed {len(data)} items",
        "status": "complete"
    }
    
    # Only copy small, necessary data
    return {
        **state,
        "results": state["results"] + [processed_result],
        "processing_flags": {**state["processing_flags"], "processed": True}
    }

async def memory_efficient_transformer(state: OptimizedState) -> OptimizedState:
    """Transform data with minimal memory footprint"""
    # Use generators for large data processing
    def process_chunks(data_ref: str, chunk_size: int = 1000):
        data = state_manager.get_data(data_ref)
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    # Process in chunks to avoid memory spikes
    chunk_results = []
    for chunk in process_chunks(state["large_data_ref"]):
        chunk_result = {"chunk_size": len(chunk), "processed": True}
        chunk_results.append(chunk_result)
    
    return {
        **state,
        "results": state["results"] + chunk_results,
        "metadata": {**state["metadata"], "chunks_processed": len(chunk_results)}
    }
```

### Optimizing Node Execution Order

```python
from typing import Set, Dict, List
import asyncio
from collections import defaultdict, deque

class ExecutionOptimizer:
    def __init__(self):
        self.dependency_graph = defaultdict(set)
        self.execution_metrics = {}
        self.node_priorities = {}
    
    def analyze_dependencies(self, graph_config: Dict) -> Dict[str, Set[str]]:
        """Analyze node dependencies for optimal execution order"""
        dependencies = defaultdict(set)
        
        for edge in graph_config.get("edges", []):
            dependencies[edge["to"]].add(edge["from"])
        
        return dependencies
    
    def calculate_optimal_order(self, dependencies: Dict[str, Set[str]]) -> List[str]:
        """Calculate optimal execution order using topological sort"""
        in_degree = defaultdict(int)
        nodes = set()
        
        # Calculate in-degrees
        for node in dependencies:
            nodes.add(node)
            for dep in dependencies[node]:
                nodes.add(dep)
                in_degree[node] += 1
        
        # Topological sort with priority
        queue = deque([node for node in nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            # Sort by priority if available
            current = min(queue, key=lambda x: self.node_priorities.get(x, 0))
            queue.remove(current)
            result.append(current)
            
            # Update in-degrees
            for node in nodes:
                if current in dependencies[node]:
                    in_degree[node] -= 1
                    if in_degree[node] == 0:
                        queue.append(node)
        
        return result
    
    def set_node_priority(self, node: str, priority: int):
        """Set execution priority for a node (higher = earlier execution)"""
        self.node_priorities[node] = priority

class OptimizedGraph:
    def __init__(self):
        self.optimizer = ExecutionOptimizer()
        self.parallel_groups = []
    
    def identify_parallel_opportunities(self, dependencies: Dict[str, Set[str]]) -> List[Set[str]]:
        """Identify nodes that can be executed in parallel"""
        parallel_groups = []
        processed = set()
        
        while len(processed) < len(dependencies):
            # Find nodes with no unprocessed dependencies
            ready_nodes = set()
            for node, deps in dependencies.items():
                if node not in processed and deps.issubset(processed):
                    ready_nodes.add(node)
            
            if ready_nodes:
                parallel_groups.append(ready_nodes)
                processed.update(ready_nodes)
            else:
                break  # Circular dependency or error
        
        return parallel_groups
    
    async def execute_parallel_group(self, nodes: Set[str], state: OptimizedState) -> OptimizedState:
        """Execute a group of nodes in parallel"""
        tasks = []
        
        for node in nodes:
            if hasattr(self, f"execute_{node}"):
                task = getattr(self, f"execute_{node}")(state)
                tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results (implement based on your merging strategy)
        merged_state = state
        for result in results:
            if not isinstance(result, Exception):
                merged_state = self.merge_states(merged_state, result)
        
        return merged_state
    
    def merge_states(self, state1: OptimizedState, state2: OptimizedState) -> OptimizedState:
        """Merge two states efficiently"""
        return {
            "large_data_ref": state1["large_data_ref"],  # Keep reference
            "processing_flags": {**state1["processing_flags"], **state2["processing_flags"]},
            "results": state1["results"] + state2["results"],
            "metadata": {**state1["metadata"], **state2["metadata"]}
        }
```

## Caching Strategies

### Result Caching Across Graph Runs

```python
import hashlib
import json
from typing import Optional, Any
from datetime import datetime, timedelta
import asyncio

class GraphCache:
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 60):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _generate_key(self, node_name: str, state_snapshot: Dict) -> str:
        """Generate a unique cache key for the state"""
        # Create a deterministic hash of the relevant state
        state_str = json.dumps(state_snapshot, sort_keys=True, default=str)
        hash_obj = hashlib.md5(f"{node_name}:{state_str}".encode())
        return hash_obj.hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - timestamp > self.ttl
    
    def get(self, node_name: str, state_snapshot: Dict) -> Optional[Any]:
        """Get cached result if available and valid"""
        key = self._generate_key(node_name, state_snapshot)
        
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            
            if not self._is_expired(timestamp):
                self.access_times[key] = datetime.now()
                return cached_data
            else:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def set(self, node_name: str, state_snapshot: Dict, result: Any):
        """Cache the result"""
        key = self._generate_key(node_name, state_snapshot)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = (result, datetime.now())
        self.access_times[key] = datetime.now()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.access_times:
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]

# Global cache instance
graph_cache = GraphCache()

def cached_node(cache_key_fields: List[str]):
    """Decorator for caching node results"""
    def decorator(func):
        async def wrapper(state: OptimizedState):
            # Create cache key from specified state fields
            cache_snapshot = {field: state.get(field) for field in cache_key_fields}
            
            # Try to get cached result
            cached_result = graph_cache.get(func.__name__, cache_snapshot)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(state)
            graph_cache.set(func.__name__, cache_snapshot, result)
            
            return result
        return wrapper
    return decorator

@cached_node(["large_data_ref", "processing_flags"])
async def expensive_analysis(state: OptimizedState) -> OptimizedState:
    """Expensive analysis that benefits from caching"""
    # Simulate expensive computation
    await asyncio.sleep(2)
    
    data = state_manager.get_data(state["large_data_ref"])
    analysis_result = {
        "complexity_score": len(data) * 0.1,
        "analysis_timestamp": datetime.now().isoformat(),
        "recommendations": ["optimize", "cache", "parallel"]
    }
    
    return {
        **state,
        "results": state["results"] + [analysis_result]
    }
```

### Intelligent Cache Invalidation

```python
class SmartCacheManager:
    def __init__(self):
        self.cache = GraphCache()
        self.dependency_tracker = defaultdict(set)
        self.invalidation_rules = {}
    
    def register_dependency(self, dependent_node: str, dependency_field: str):
        """Register that a node depends on a specific state field"""
        self.dependency_tracker[dependency_field].add(dependent_node)
    
    def register_invalidation_rule(self, field: str, invalidation_func):
        """Register custom invalidation logic for a field"""
        self.invalidation_rules[field] = invalidation_func
    
    def invalidate_dependent_caches(self, changed_fields: Set[str]):
        """Invalidate caches for nodes that depend on changed fields"""
        nodes_to_invalidate = set()
        
        for field in changed_fields:
            nodes_to_invalidate.update(self.dependency_tracker[field])
            
            # Apply custom invalidation rules
            if field in self.invalidation_rules:
                additional_nodes = self.invalidation_rules[field](field)
                nodes_to_invalidate.update(additional_nodes)
        
        # Clear cache entries for affected nodes
        for node in nodes_to_invalidate:
            self._clear_node_cache(node)
    
    def _clear_node_cache(self, node_name: str):
        """Clear all cache entries for a specific node"""
        keys_to_remove = [
            key for key in self.cache.cache.keys() 
            if key.startswith(f"{node_name}:")
        ]
        
        for key in keys_to_remove:
            self.cache.cache.pop(key, None)
            self.cache.access_times.pop(key, None)

# Usage example
smart_cache = SmartCacheManager()

# Register dependencies
smart_cache.register_dependency("analysis_node", "large_data_ref")
smart_cache.register_dependency("processing_node", "processing_flags")

# Custom invalidation rule
def data_change_invalidation(field: str) -> Set[str]:
    """Custom rule: if data changes, invalidate all analysis nodes"""
    if field == "large_data_ref":
        return {"analysis_node", "summary_node", "report_node"}
    return set()

smart_cache.register_invalidation_rule("large_data_ref", data_change_invalidation)
```

## Memory-Efficient Patterns

### Resource Usage Optimization

```python
import psutil
import gc
from contextlib import asynccontextmanager

class ResourceMonitor:
    def __init__(self, memory_threshold_mb: int = 500):
        self.memory_threshold = memory_threshold_mb * 1024 * 1024  # Convert to bytes
        self.peak_memory = 0
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        process = psutil.Process()
        return process.memory_info().rss
    
    @asynccontextmanager
    async def monitor_execution(self, node_name: str):
        """Context manager to monitor resource usage during node execution"""
        start_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            memory_delta = end_memory - start_memory
            
            if end_memory > self.peak_memory:
                self.peak_memory = end_memory
            
            # Trigger garbage collection if memory usage is high
            if end_memory > self.memory_threshold:
                gc.collect()
            
            print(f"Node {node_name}: Memory delta: {memory_delta / 1024 / 1024:.2f} MB")

# Global resource monitor
resource_monitor = ResourceMonitor()

async def memory_optimized_node(state: OptimizedState) -> OptimizedState:
    """Node with built-in memory optimization"""
    async with resource_monitor.monitor_execution("memory_optimized_node"):
        # Use generators for large data processing
        def process_in_batches(data_ref: str, batch_size: int = 100):
            data = state_manager.get_data(data_ref)
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                # Process batch and yield results immediately
                yield {"batch_id": i // batch_size, "processed_count": len(batch)}
                # Clear batch from memory
                del batch
        
        # Collect results from generator
        batch_results = list(process_in_batches(state["large_data_ref"]))
        
        return {
            **state,
            "results": state["results"] + batch_results,
            "memory_optimized": True
        }
```

## Performance Best Practices

1. **State Design**: Use references for large objects, minimize state copying
2. **Execution Order**: Analyze dependencies and execute independent nodes in parallel
3. **Caching**: Implement intelligent caching with proper invalidation strategies
4. **Memory Management**: Monitor resource usage and implement cleanup strategies
5. **Profiling**: Regularly profile your graphs to identify bottlenecks

These optimization techniques ensure your LangGraph applications scale efficiently in production environments.
