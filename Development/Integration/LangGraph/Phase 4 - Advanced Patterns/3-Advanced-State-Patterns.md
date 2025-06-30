# Advanced State Patterns

## State Versioning

### Temporal State Management

```python
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from copy import deepcopy
import uuid

@dataclass
class StateSnapshot:
    """Represents a point-in-time state snapshot."""
    snapshot_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    version: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class VersionedState(TypedDict):
    current_data: Dict[str, Any]
    version_history: List[StateSnapshot]
    current_version: int
    checkpoints: Dict[str, StateSnapshot]
    rollback_points: List[str]
    temporal_queries: List[Dict[str, Any]]

class StateVersionManager:
    """Manages state versioning and temporal operations."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.auto_checkpoint_interval = timedelta(minutes=5)
        self.last_auto_checkpoint = datetime.now()
    
    def create_snapshot(self, state_data: Dict[str, Any], 
                       metadata: Dict[str, Any] = None) -> StateSnapshot:
        """Create a new state snapshot."""
        return StateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            state_data=deepcopy(state_data),
            version=len(state_data.get("version_history", [])) + 1,
            metadata=metadata or {}
        )
    
    def update_versioned_state(self, state: VersionedState, 
                             updates: Dict[str, Any],
                             create_checkpoint: bool = False) -> VersionedState:
        """Update state while maintaining version history."""
        
        # Create snapshot of current state
        current_snapshot = self.create_snapshot(
            state["current_data"],
            {"update_type": "automatic", "changes": list(updates.keys())}
        )
        
        # Update current data
        updated_data = {**state["current_data"], **updates}
        
        # Manage version history
        updated_history = state["version_history"] + [current_snapshot]
        if len(updated_history) > self.max_history:
            # Remove oldest snapshots but keep important checkpoints
            updated_history = self._prune_history(updated_history, state["checkpoints"])
        
        # Auto-checkpoint if needed
        checkpoints = state["checkpoints"].copy()
        if (datetime.now() - self.last_auto_checkpoint) > self.auto_checkpoint_interval:
            checkpoint_name = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoints[checkpoint_name] = current_snapshot
            self.last_auto_checkpoint = datetime.now()
        
        # Manual checkpoint if requested
        if create_checkpoint:
            checkpoint_name = f"manual_{current_snapshot.snapshot_id[:8]}"
            checkpoints[checkpoint_name] = current_snapshot
        
        return {
            **state,
            "current_data": updated_data,
            "version_history": updated_history,
            "current_version": state["current_version"] + 1,
            "checkpoints": checkpoints
        }
    
    def _prune_history(self, history: List[StateSnapshot], 
                      checkpoints: Dict[str, StateSnapshot]) -> List[StateSnapshot]:
        """Prune history while preserving important snapshots."""
        checkpoint_ids = {cp.snapshot_id for cp in checkpoints.values()}
        
        # Keep recent snapshots and checkpointed snapshots
        pruned = []
        for snapshot in history[-self.max_history//2:]:  # Keep recent half
            pruned.append(snapshot)
        
        # Add checkpointed snapshots not in recent history
        for snapshot in history:
            if snapshot.snapshot_id in checkpoint_ids and snapshot not in pruned:
                pruned.append(snapshot)
        
        return sorted(pruned, key=lambda x: x.timestamp)

def temporal_query_node(state: VersionedState) -> VersionedState:
    """Node that can query historical state data."""
    
    # Example temporal queries
    queries = [
        {"type": "state_at_time", "timestamp": datetime.now() - timedelta(hours=1)},
        {"type": "changes_since", "since": datetime.now() - timedelta(minutes=30)},
        {"type": "version_comparison", "version1": 1, "version2": "current"}
    ]
    
    query_results = []
    
    for query in queries:
        if query["type"] == "state_at_time":
            result = find_state_at_time(state, query["timestamp"])
            query_results.append({
                "query": query,
                "result": result,
                "found": result is not None
            })
        
        elif query["type"] == "changes_since":
            changes = find_changes_since(state, query["since"])
            query_results.append({
                "query": query,
                "result": changes,
                "change_count": len(changes)
            })
    
    return {
        **state,
        "temporal_queries": query_results
    }

def find_state_at_time(state: VersionedState, target_time: datetime) -> Optional[Dict[str, Any]]:
    """Find state as it existed at a specific time."""
    # Find the latest snapshot before or at the target time
    valid_snapshots = [
        snapshot for snapshot in state["version_history"]
        if snapshot.timestamp <= target_time
    ]
    
    if valid_snapshots:
        latest_snapshot = max(valid_snapshots, key=lambda x: x.timestamp)
        return latest_snapshot.state_data
    
    return None

def find_changes_since(state: VersionedState, since_time: datetime) -> List[Dict[str, Any]]:
    """Find all changes since a specific time."""
    recent_snapshots = [
        snapshot for snapshot in state["version_history"]
        if snapshot.timestamp >= since_time
    ]
    
    changes = []
    for i, snapshot in enumerate(recent_snapshots):
        if i == 0:
            # First snapshot - compare with previous if available
            prev_snapshots = [s for s in state["version_history"] 
                            if s.timestamp < since_time]
            if prev_snapshots:
                prev_snapshot = max(prev_snapshots, key=lambda x: x.timestamp)
                changes.append(compare_snapshots(prev_snapshot, snapshot))
        else:
            # Compare with previous snapshot in the range
            changes.append(compare_snapshots(recent_snapshots[i-1], snapshot))
    
    return changes

def compare_snapshots(snapshot1: StateSnapshot, snapshot2: StateSnapshot) -> Dict[str, Any]:
    """Compare two state snapshots and return differences."""
    changes = {
        "from_version": snapshot1.version,
        "to_version": snapshot2.version,
        "time_diff": (snapshot2.timestamp - snapshot1.timestamp).total_seconds(),
        "added": {},
        "modified": {},
        "removed": {}
    }
    
    data1, data2 = snapshot1.state_data, snapshot2.state_data
    
    # Find added and modified
    for key, value in data2.items():
        if key not in data1:
            changes["added"][key] = value
        elif data1[key] != value:
            changes["modified"][key] = {"from": data1[key], "to": value}
    
    # Find removed
    for key in data1:
        if key not in data2:
            changes["removed"][key] = data1[key]
    
    return changes
```

### State Rollback Mechanisms

```python
class StateRollbackManager:
    """Manages state rollback operations."""
    
    def __init__(self):
        self.rollback_strategies = {
            "checkpoint": self._rollback_to_checkpoint,
            "version": self._rollback_to_version,
            "time": self._rollback_to_time,
            "condition": self._rollback_to_condition
        }
    
    def rollback_state(self, state: VersionedState, rollback_config: Dict[str, Any]) -> VersionedState:
        """Perform state rollback based on configuration."""
        strategy = rollback_config.get("strategy", "checkpoint")
        
        if strategy in self.rollback_strategies:
            return self.rollback_strategies[strategy](state, rollback_config)
        else:
            raise ValueError(f"Unknown rollback strategy: {strategy}")
    
    def _rollback_to_checkpoint(self, state: VersionedState, config: Dict[str, Any]) -> VersionedState:
        """Rollback to a named checkpoint."""
        checkpoint_name = config["checkpoint_name"]
        
        if checkpoint_name not in state["checkpoints"]:
            raise ValueError(f"Checkpoint '{checkpoint_name}' not found")
        
        target_snapshot = state["checkpoints"][checkpoint_name]
        
        # Create snapshot of current state before rollback
        current_snapshot = StateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            state_data=deepcopy(state["current_data"]),
            version=state["current_version"],
            metadata={"rollback_point": True, "rolled_back_from": state["current_version"]}
        )
        
        return {
            **state,
            "current_data": deepcopy(target_snapshot.state_data),
            "current_version": target_snapshot.version,
            "version_history": state["version_history"] + [current_snapshot],
            "rollback_points": state["rollback_points"] + [current_snapshot.snapshot_id]
        }
    
    def _rollback_to_version(self, state: VersionedState, config: Dict[str, Any]) -> VersionedState:
        """Rollback to a specific version number."""
        target_version = config["version"]
        
        # Find snapshot with target version
        target_snapshot = None
        for snapshot in state["version_history"]:
            if snapshot.version == target_version:
                target_snapshot = snapshot
                break
        
        if not target_snapshot:
            raise ValueError(f"Version {target_version} not found in history")
        
        return self._perform_rollback(state, target_snapshot)
    
    def _rollback_to_time(self, state: VersionedState, config: Dict[str, Any]) -> VersionedState:
        """Rollback to a specific point in time."""
        target_time = config["timestamp"]
        
        # Find the latest snapshot before the target time
        valid_snapshots = [
            snapshot for snapshot in state["version_history"]
            if snapshot.timestamp <= target_time
        ]
        
        if not valid_snapshots:
            raise ValueError("No snapshots found before the target time")
        
        target_snapshot = max(valid_snapshots, key=lambda x: x.timestamp)
        return self._perform_rollback(state, target_snapshot)
    
    def _rollback_to_condition(self, state: VersionedState, config: Dict[str, Any]) -> VersionedState:
        """Rollback to the last state that met a specific condition."""
        condition_func = config["condition"]
        
        # Find the latest snapshot that meets the condition
        for snapshot in reversed(state["version_history"]):
            if condition_func(snapshot.state_data):
                return self._perform_rollback(state, snapshot)
        
        raise ValueError("No snapshot found that meets the rollback condition")
    
    def _perform_rollback(self, state: VersionedState, target_snapshot: StateSnapshot) -> VersionedState:
        """Perform the actual rollback operation."""
        # Record current state before rollback
        pre_rollback_snapshot = StateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            state_data=deepcopy(state["current_data"]),
            version=state["current_version"],
            metadata={"pre_rollback": True, "rollback_target": target_snapshot.version}
        )
        
        return {
            **state,
            "current_data": deepcopy(target_snapshot.state_data),
            "current_version": target_snapshot.version,
            "version_history": state["version_history"] + [pre_rollback_snapshot],
            "rollback_points": state["rollback_points"] + [pre_rollback_snapshot.snapshot_id]
        }

def rollback_controller_node(state: VersionedState) -> VersionedState:
    """Node that handles rollback decisions."""
    rollback_manager = StateRollbackManager()
    
    # Example rollback scenarios
    current_data = state["current_data"]
    
    # Check if rollback is needed based on data quality
    if current_data.get("error_count", 0) > 5:
        # Rollback to last checkpoint
        try:
            if state["checkpoints"]:
                latest_checkpoint = max(
                    state["checkpoints"].items(), 
                    key=lambda x: x[1].timestamp
                )[0]
                
                rollback_config = {
                    "strategy": "checkpoint",
                    "checkpoint_name": latest_checkpoint
                }
                
                return rollback_manager.rollback_state(state, rollback_config)
        except Exception as e:
            # Log rollback failure but continue
            pass
    
    # Check if rollback needed based on performance
    elif current_data.get("performance_score", 1.0) < 0.3:
        # Rollback to last state with good performance
        def good_performance_condition(data):
            return data.get("performance_score", 0) >= 0.7
        
        try:
            rollback_config = {
                "strategy": "condition", 
                "condition": good_performance_condition
            }
            
            return rollback_manager.rollback_state(state, rollback_config)
        except Exception as e:
            # No good performance state found
            pass
    
    return state
```

### Checkpoint and Restore Patterns

```python
def checkpoint_creator_node(state: VersionedState) -> VersionedState:
    """Node that creates strategic checkpoints."""
    version_manager = StateVersionManager()
    
    current_data = state["current_data"]
    should_checkpoint = False
    checkpoint_reason = ""
    
    # Strategic checkpoint conditions
    if current_data.get("milestone_reached", False):
        should_checkpoint = True
        checkpoint_reason = "milestone_checkpoint"
    
    elif current_data.get("data_quality_score", 0) >= 0.9:
        should_checkpoint = True
        checkpoint_reason = "high_quality_checkpoint"
    
    elif len(state["version_history"]) % 10 == 0:  # Every 10 versions
        should_checkpoint = True
        checkpoint_reason = "periodic_checkpoint"
    
    if should_checkpoint:
        checkpoint_name = f"{checkpoint_reason}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_snapshot = version_manager.create_snapshot(
            current_data,
            {"checkpoint_reason": checkpoint_reason, "strategic": True}
        )
        
        updated_checkpoints = state["checkpoints"].copy()
        updated_checkpoints[checkpoint_name] = checkpoint_snapshot
        
        return {
            **state,
            "checkpoints": updated_checkpoints,
            "current_data": {
                **current_data,
                "last_checkpoint": checkpoint_name,
                "checkpoint_count": len(updated_checkpoints)
            }
        }
    
    return state

# Build workflow with advanced state patterns
def create_advanced_state_workflow():
    """Create workflow demonstrating advanced state patterns."""
    
    workflow = StateGraph(VersionedState)
    
    # Add state management nodes
    workflow.add_node("version_update", lambda s: StateVersionManager().update_versioned_state(
        s, {"processing_step": len(s["version_history"]) + 1}, create_checkpoint=False
    ))
    
    workflow.add_node("temporal_query", temporal_query_node)
    workflow.add_node("checkpoint_creator", checkpoint_creator_node)
    workflow.add_node("rollback_controller", rollback_controller_node)
    workflow.add_node("state_analyzer", lambda s: {
        **s,
        "current_data": {
            **s["current_data"],
            "total_versions": len(s["version_history"]),
            "checkpoint_count": len(s["checkpoints"]),
            "rollback_count": len(s["rollback_points"])
        }
    })
    
    # Set up workflow
    workflow.set_entry_point("version_update")
    workflow.add_edge("version_update", "temporal_query")
    workflow.add_edge("temporal_query", "checkpoint_creator")
    workflow.add_edge("checkpoint_creator", "rollback_controller")
    workflow.add_edge("rollback_controller", "state_analyzer")
    workflow.add_edge("state_analyzer", "__end__")
    
    return workflow.compile()

# Usage example
def test_advanced_state_patterns():
    """Test advanced state management patterns."""
    
    advanced_state_app = create_advanced_state_workflow()
    
    # Initialize state
    initial_state = {
        "current_data": {
            "task_id": "advanced_state_test",
            "data_quality_score": 0.8,
            "performance_score": 0.9,
            "error_count": 0,
            "milestone_reached": True
        },
        "version_history": [],
        "current_version": 0,
        "checkpoints": {},
        "rollback_points": [],
        "temporal_queries": []
    }
    
    result = advanced_state_app.invoke(initial_state)
    
    print("Advanced State Management Result:")
    print(f"Final version: {result['current_version']}")
    print(f"Total checkpoints: {len(result['checkpoints'])}")
    print(f"Version history length: {len(result['version_history'])}")
    print(f"Temporal queries executed: {len(result['temporal_queries'])}")
    print(f"Current data keys: {list(result['current_data'].keys())}")

if __name__ == "__main__":
    test_advanced_state_patterns()
```

These advanced state patterns provide sophisticated state management capabilities including versioning, temporal queries, rollback mechanisms, and strategic checkpointing essential for complex, long-running LangGraph applications.
