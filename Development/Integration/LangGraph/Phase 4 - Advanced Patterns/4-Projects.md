# Phase 4 Projects

## Project 1: Adaptive Learning System

### Overview
Build a graph that modifies itself based on performance feedback, implementing dynamic strategy selection and self-improvement mechanisms.

```python
from typing import TypedDict, List, Dict, Any, Callable
from langgraph.graph import StateGraph
from datetime import datetime, timedelta
import statistics
import random

class AdaptiveLearningState(TypedDict):
    learning_task: str
    performance_history: List[Dict[str, float]]
    current_strategy: str
    available_strategies: List[str]
    strategy_performance: Dict[str, List[float]]
    adaptation_count: int
    learning_rate: float
    confidence_threshold: float
    graph_modifications: List[str]
    performance_metrics: Dict[str, float]

class StrategyManager:
    """Manages learning strategies and their performance."""
    
    def __init__(self):
        self.strategies = {
            "conservative": {
                "learning_rate": 0.1,
                "exploration_factor": 0.2,
                "risk_tolerance": 0.3,
                "description": "Safe, gradual learning approach"
            },
            "balanced": {
                "learning_rate": 0.3,
                "exploration_factor": 0.5,
                "risk_tolerance": 0.5,
                "description": "Balanced learning and exploration"
            },
            "aggressive": {
                "learning_rate": 0.7,
                "exploration_factor": 0.8,
                "risk_tolerance": 0.8,
                "description": "Fast learning with high exploration"
            },
            "adaptive": {
                "learning_rate": 0.5,
                "exploration_factor": 0.6,
                "risk_tolerance": 0.6,
                "description": "Self-adjusting parameters"
            }
        }
    
    def evaluate_strategy_performance(self, strategy: str, 
                                    performance_history: List[float]) -> Dict[str, float]:
        """Evaluate how well a strategy is performing."""
        if not performance_history:
            return {"score": 0.5, "confidence": 0.0, "trend": 0.0}
        
        recent_performance = performance_history[-5:]  # Last 5 results
        
        # Calculate metrics
        avg_performance = statistics.mean(recent_performance)
        
        # Calculate trend (improvement over time)
        if len(recent_performance) >= 2:
            trend = recent_performance[-1] - recent_performance[0]
        else:
            trend = 0.0
        
        # Calculate consistency (lower std dev = more consistent)
        consistency = 1.0 - (statistics.stdev(recent_performance) if len(recent_performance) > 1 else 0.5)
        
        # Overall score combines performance, trend, and consistency
        score = (avg_performance * 0.6) + (max(trend, 0) * 0.3) + (consistency * 0.1)
        
        confidence = min(len(recent_performance) / 5.0, 1.0)  # More data = higher confidence
        
        return {
            "score": score,
            "confidence": confidence,
            "trend": trend,
            "avg_performance": avg_performance,
            "consistency": consistency
        }
    
    def select_best_strategy(self, strategy_performance: Dict[str, List[float]]) -> str:
        """Select the best performing strategy."""
        strategy_scores = {}
        
        for strategy, performance_list in strategy_performance.items():
            if strategy in self.strategies:
                evaluation = self.evaluate_strategy_performance(strategy, performance_list)
                # Weight score by confidence
                weighted_score = evaluation["score"] * evaluation["confidence"]
                strategy_scores[strategy] = weighted_score
        
        if strategy_scores:
            return max(strategy_scores.items(), key=lambda x: x[1])[0]
        else:
            return "balanced"  # Default fallback

def performance_evaluator_node(state: AdaptiveLearningState) -> AdaptiveLearningState:
    """Evaluate current performance and learning effectiveness."""
    
    # Simulate task performance based on current strategy
    strategy_config = StrategyManager().strategies.get(state["current_strategy"], {})
    base_performance = 0.6  # Base performance level
    
    # Strategy affects performance differently based on task complexity
    learning_rate = strategy_config.get("learning_rate", 0.3)
    exploration = strategy_config.get("exploration_factor", 0.5)
    
    # Simulate performance with some randomness and learning effects
    adaptation_bonus = min(state["adaptation_count"] * 0.05, 0.3)  # Learning improves over time
    exploration_variance = exploration * 0.2 * (random.random() - 0.5)  # Higher exploration = more variance
    
    current_performance = min(max(
        base_performance + (learning_rate * 0.3) + adaptation_bonus + exploration_variance,
        0.0
    ), 1.0)
    
    # Record performance
    performance_record = {
        "timestamp": datetime.now().isoformat(),
        "strategy": state["current_strategy"],
        "performance": current_performance,
        "learning_rate": learning_rate,
        "adaptation_count": state["adaptation_count"]
    }
    
    # Update strategy performance tracking
    updated_strategy_performance = state["strategy_performance"].copy()
    if state["current_strategy"] not in updated_strategy_performance:
        updated_strategy_performance[state["current_strategy"]] = []
    
    updated_strategy_performance[state["current_strategy"]].append(current_performance)
    
    # Calculate overall metrics
    all_performances = [p["performance"] for p in state["performance_history"]] + [current_performance]
    performance_metrics = {
        "current_performance": current_performance,
        "average_performance": statistics.mean(all_performances),
        "performance_trend": current_performance - (all_performances[-2] if len(all_performances) > 1 else current_performance),
        "total_evaluations": len(all_performances)
    }
    
    return {
        **state,
        "performance_history": state["performance_history"] + [performance_record],
        "strategy_performance": updated_strategy_performance,
        "performance_metrics": performance_metrics
    }

def adaptive_strategy_selector(state: AdaptiveLearningState) -> AdaptiveLearningState:
    """Dynamically select and adapt learning strategies."""
    strategy_manager = StrategyManager()
    
    # Only adapt if we have enough data
    if len(state["performance_history"]) < 3:
        return state
    
    # Evaluate all strategies
    best_strategy = strategy_manager.select_best_strategy(state["strategy_performance"])
    
    modifications = []
    
    # Check if strategy change is warranted
    current_strategy_performance = state["strategy_performance"].get(state["current_strategy"], [])
    if current_strategy_performance:
        recent_avg = statistics.mean(current_strategy_performance[-3:])
        
        # Switch strategy if performance is poor or if better strategy found
        if recent_avg < 0.4 or (best_strategy != state["current_strategy"] and recent_avg < 0.7):
            modifications.append(f"Strategy changed from {state['current_strategy']} to {best_strategy}")
            new_strategy = best_strategy
            adaptation_count = state["adaptation_count"] + 1
        else:
            new_strategy = state["current_strategy"]
            adaptation_count = state["adaptation_count"]
    else:
        new_strategy = best_strategy
        adaptation_count = state["adaptation_count"] + 1
    
    # Adaptive learning rate adjustment
    performance_trend = state["performance_metrics"].get("performance_trend", 0)
    current_learning_rate = state["learning_rate"]
    
    if performance_trend > 0.1:  # Performance improving
        # Slightly reduce learning rate for stability
        new_learning_rate = max(current_learning_rate * 0.95, 0.05)
        modifications.append("Learning rate decreased for stability")
    elif performance_trend < -0.1:  # Performance declining
        # Increase learning rate for faster adaptation
        new_learning_rate = min(current_learning_rate * 1.1, 0.9)
        modifications.append("Learning rate increased for faster adaptation")
    else:
        new_learning_rate = current_learning_rate
    
    return {
        **state,
        "current_strategy": new_strategy,
        "adaptation_count": adaptation_count,
        "learning_rate": new_learning_rate,
        "graph_modifications": state["graph_modifications"] + modifications
    }

def dynamic_graph_modifier(state: AdaptiveLearningState) -> AdaptiveLearningState:
    """Modify graph structure based on learning progress."""
    modifications = []
    
    # Add specialized nodes based on performance patterns
    avg_performance = state["performance_metrics"].get("average_performance", 0.5)
    
    if avg_performance < 0.4:
        # Poor performance - add remedial processing
        modifications.append("Added remedial processing node due to poor performance")
    elif avg_performance > 0.8:
        # Excellent performance - add advanced optimization
        modifications.append("Added advanced optimization node due to excellent performance")
    
    # Add exploration nodes if using aggressive strategy
    if state["current_strategy"] == "aggressive":
        modifications.append("Added exploration enhancement for aggressive strategy")
    
    # Modify based on adaptation count
    if state["adaptation_count"] > 5:
        modifications.append("Added meta-learning node due to high adaptation count")
    
    return {
        **state,
        "graph_modifications": state["graph_modifications"] + modifications
    }

def learning_performance_monitor(state: AdaptiveLearningState) -> AdaptiveLearningState:
    """Monitor and report on learning system performance."""
    
    # Generate learning summary
    total_evaluations = len(state["performance_history"])
    if total_evaluations > 0:
        performances = [p["performance"] for p in state["performance_history"]]
        
        learning_summary = {
            "total_adaptations": state["adaptation_count"],
            "current_strategy": state["current_strategy"],
            "performance_improvement": performances[-1] - performances[0] if len(performances) > 1 else 0,
            "average_performance": statistics.mean(performances),
            "best_performance": max(performances),
            "performance_variance": statistics.stdev(performances) if len(performances) > 1 else 0,
            "learning_efficiency": state["adaptation_count"] / total_evaluations if total_evaluations > 0 else 0
        }
    else:
        learning_summary = {"status": "insufficient_data"}
    
    return {
        **state,
        "performance_metrics": {
            **state["performance_metrics"],
            "learning_summary": learning_summary
        }
    }

## Project 2: Fault-Tolerant Processing Pipeline

### Overview
Create a robust pipeline with comprehensive error handling, multiple fallback strategies, and automatic recovery mechanisms.

```python
class FaultTolerantState(TypedDict):
    input_data: List[str]
    processed_data: List[Dict[str, Any]]
    failed_items: List[Dict[str, Any]]
    error_log: List[Dict[str, Any]]
    fallback_history: List[str]
    current_fallback_level: int
    max_fallback_levels: int
    recovery_attempts: int
    pipeline_health: Dict[str, Any]
    processing_statistics: Dict[str, Any]

class FaultTolerantProcessor:
    """Handles fault-tolerant processing with multiple fallback strategies."""
    
    def __init__(self):
        self.fallback_strategies = [
            "primary_processing",
            "simplified_processing", 
            "cached_processing",
            "manual_intervention"
        ]
        self.error_patterns = {}
        self.recovery_strategies = {}
    
    def register_error_pattern(self, pattern_name: str, error_handler: Callable):
        """Register custom error handling patterns."""
        self.error_patterns[pattern_name] = error_handler
    
    def process_with_fallback(self, data: Any, fallback_level: int) -> Dict[str, Any]:
        """Process data with appropriate fallback strategy."""
        strategy = self.fallback_strategies[min(fallback_level, len(self.fallback_strategies) - 1)]
        
        try:
            if strategy == "primary_processing":
                return self._primary_process(data)
            elif strategy == "simplified_processing":
                return self._simplified_process(data)
            elif strategy == "cached_processing":
                return self._cached_process(data)
            else:  # manual_intervention
                return self._manual_intervention_process(data)
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "strategy_used": strategy,
                "data": data
            }
    
    def _primary_process(self, data: str) -> Dict[str, Any]:
        """Primary processing logic - may fail."""
        if "error" in data.lower():
            raise ValueError(f"Processing error for: {data}")
        
        if "timeout" in data.lower():
            raise TimeoutError(f"Processing timeout for: {data}")
        
        return {
            "success": True,
            "result": f"Primary processed: {data.upper()}",
            "strategy_used": "primary_processing",
            "processing_time": 1.0
        }
    
    def _simplified_process(self, data: str) -> Dict[str, Any]:
        """Simplified processing - more reliable but less sophisticated."""
        return {
            "success": True,
            "result": f"Simple processed: {data}",
            "strategy_used": "simplified_processing",
            "processing_time": 0.5
        }
    
    def _cached_process(self, data: str) -> Dict[str, Any]:
        """Use cached or default processing."""
        return {
            "success": True,
            "result": f"Cached result for: {data}",
            "strategy_used": "cached_processing",
            "processing_time": 0.1
        }
    
    def _manual_intervention_process(self, data: str) -> Dict[str, Any]:
        """Flag for manual processing."""
        return {
            "success": True,
            "result": f"Flagged for manual processing: {data}",
            "strategy_used": "manual_intervention",
            "processing_time": 0.0,
            "requires_manual_review": True
        }

def fault_tolerant_processor_node(state: FaultTolerantState) -> FaultTolerantState:
    """Main processing node with fault tolerance."""
    processor = FaultTolerantProcessor()
    
    processed_data = state["processed_data"].copy()
    failed_items = state["failed_items"].copy()
    error_log = state["error_log"].copy()
    
    # Process each item in input_data
    for item in state["input_data"]:
        max_attempts = 3
        attempts = 0
        success = False
        
        while attempts < max_attempts and not success:
            try:
                result = processor.process_with_fallback(item, state["current_fallback_level"])
                
                if result["success"]:
                    processed_data.append({
                        "original_data": item,
                        "result": result["result"],
                        "strategy": result["strategy_used"],
                        "attempts": attempts + 1,
                        "processing_time": result.get("processing_time", 0)
                    })
                    success = True
                else:
                    attempts += 1
                    error_log.append({
                        "item": item,
                        "attempt": attempts,
                        "error": result["error"],
                        "timestamp": datetime.now().isoformat()
                    })
            
            except Exception as e:
                attempts += 1
                error_log.append({
                    "item": item,
                    "attempt": attempts,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # If all attempts failed, add to failed_items
        if not success:
            failed_items.append({
                "data": item,
                "attempts": attempts,
                "final_error": error_log[-1] if error_log else "Unknown error"
            })
    
    # Update processing statistics
    total_items = len(state["input_data"])
    success_count = len(processed_data) - len(state["processed_data"])
    failure_count = len(failed_items) - len(state["failed_items"])
    
    processing_stats = {
        "total_processed": total_items,
        "successful": success_count,
        "failed": failure_count,
        "success_rate": success_count / total_items if total_items > 0 else 0,
        "current_batch_errors": len(error_log) - len(state["error_log"])
    }
    
    return {
        **state,
        "processed_data": processed_data,
        "failed_items": failed_items,
        "error_log": error_log,
        "processing_statistics": processing_stats,
        "recovery_attempts": state["recovery_attempts"] + (failure_count > 0)
    }

def fallback_escalation_controller(state: FaultTolerantState) -> FaultTolerantState:
    """Control fallback strategy escalation."""
    
    current_success_rate = state["processing_statistics"].get("success_rate", 1.0)
    current_errors = state["processing_statistics"].get("current_batch_errors", 0)
    
    escalate_fallback = False
    fallback_reason = ""
    
    # Escalate if success rate is too low
    if current_success_rate < 0.7:
        escalate_fallback = True
        fallback_reason = f"Low success rate: {current_success_rate:.2f}"
    
    # Escalate if too many errors in current batch
    elif current_errors > 3:
        escalate_fallback = True
        fallback_reason = f"High error count: {current_errors}"
    
    if escalate_fallback and state["current_fallback_level"] < state["max_fallback_levels"]:
        new_fallback_level = state["current_fallback_level"] + 1
        fallback_history = state["fallback_history"] + [
            f"Escalated to level {new_fallback_level}: {fallback_reason}"
        ]
        
        return {
            **state,
            "current_fallback_level": new_fallback_level,
            "fallback_history": fallback_history
        }
    
    return state

def pipeline_health_monitor(state: FaultTolerantState) -> FaultTolerantState:
    """Monitor overall pipeline health and performance."""
    
    # Calculate health metrics
    total_processed = sum(stats.get("total_processed", 0) for stats in [state["processing_statistics"]])
    total_successful = sum(stats.get("successful", 0) for stats in [state["processing_statistics"]])
    total_failed = len(state["failed_items"])
    
    overall_success_rate = total_successful / max(total_processed, 1)
    error_rate = len(state["error_log"]) / max(total_processed, 1)
    
    # Determine health status
    if overall_success_rate >= 0.9 and error_rate <= 0.1:
        health_status = "excellent"
    elif overall_success_rate >= 0.7 and error_rate <= 0.3:
        health_status = "good"
    elif overall_success_rate >= 0.5 and error_rate <= 0.5:
        health_status = "fair"
    else:
        health_status = "poor"
    
    pipeline_health = {
        "status": health_status,
        "overall_success_rate": overall_success_rate,
        "error_rate": error_rate,
        "total_recovery_attempts": state["recovery_attempts"],
        "current_fallback_level": state["current_fallback_level"],
        "recommendation": get_health_recommendation(health_status, state)
    }
    
    return {
        **state,
        "pipeline_health": pipeline_health
    }

def get_health_recommendation(health_status: str, state: FaultTolerantState) -> str:
    """Get recommendation based on pipeline health."""
    if health_status == "poor":
        return "Consider system maintenance or data quality review"
    elif health_status == "fair":
        return "Monitor closely and investigate frequent error patterns"
    elif health_status == "good":
        return "Pipeline performing well, continue monitoring"
    else:
        return "Pipeline operating optimally"

# Build both project workflows
def create_adaptive_learning_workflow():
    """Create the adaptive learning system workflow."""
    
    workflow = StateGraph(AdaptiveLearningState)
    
    workflow.add_node("evaluate_performance", performance_evaluator_node)
    workflow.add_node("adapt_strategy", adaptive_strategy_selector)
    workflow.add_node("modify_graph", dynamic_graph_modifier)
    workflow.add_node("monitor_learning", learning_performance_monitor)
    
    workflow.set_entry_point("evaluate_performance")
    workflow.add_edge("evaluate_performance", "adapt_strategy")
    workflow.add_edge("adapt_strategy", "modify_graph")
    workflow.add_edge("modify_graph", "monitor_learning")
    workflow.add_edge("monitor_learning", "__end__")
    
    return workflow.compile()

def create_fault_tolerant_pipeline():
    """Create the fault-tolerant processing pipeline."""
    
    workflow = StateGraph(FaultTolerantState)
    
    workflow.add_node("process_data", fault_tolerant_processor_node)
    workflow.add_node("escalate_fallback", fallback_escalation_controller)
    workflow.add_node("monitor_health", pipeline_health_monitor)
    
    workflow.set_entry_point("process_data")
    workflow.add_edge("process_data", "escalate_fallback")
    workflow.add_edge("escalate_fallback", "monitor_health")
    workflow.add_edge("monitor_health", "__end__")
    
    return workflow.compile()

# Test both projects
def test_phase4_projects():
    """Test both Phase 4 projects."""
    
    # Test Adaptive Learning System
    print("=== Testing Adaptive Learning System ===")
    adaptive_app = create_adaptive_learning_workflow()
    
    adaptive_result = adaptive_app.invoke({
        "learning_task": "classification_optimization",
        "performance_history": [],
        "current_strategy": "balanced",
        "available_strategies": ["conservative", "balanced", "aggressive", "adaptive"],
        "strategy_performance": {},
        "adaptation_count": 0,
        "learning_rate": 0.3,
        "confidence_threshold": 0.7,
        "graph_modifications": [],
        "performance_metrics": {}
    })
    
    print(f"Final strategy: {adaptive_result['current_strategy']}")
    print(f"Adaptations made: {adaptive_result['adaptation_count']}")
    print(f"Graph modifications: {len(adaptive_result['graph_modifications'])}")
    print(f"Performance evaluations: {len(adaptive_result['performance_history'])}")
    
    # Test Fault-Tolerant Pipeline
    print("\n=== Testing Fault-Tolerant Pipeline ===")
    fault_tolerant_app = create_fault_tolerant_pipeline()
    
    test_data = [
        "normal_data_1",
        "error_data",  # Will cause errors
        "normal_data_2", 
        "timeout_data",  # Will cause timeout
        "normal_data_3"
    ]
    
    pipeline_result = fault_tolerant_app.invoke({
        "input_data": test_data,
        "processed_data": [],
        "failed_items": [],
        "error_log": [],
        "fallback_history": [],
        "current_fallback_level": 0,
        "max_fallback_levels": 3,
        "recovery_attempts": 0,
        "pipeline_health": {},
        "processing_statistics": {}
    })
    
    print(f"Processed items: {len(pipeline_result['processed_data'])}")
    print(f"Failed items: {len(pipeline_result['failed_items'])}")
    print(f"Error log entries: {len(pipeline_result['error_log'])}")
    print(f"Pipeline health: {pipeline_result['pipeline_health']['status']}")
    print(f"Success rate: {pipeline_result['pipeline_health']['overall_success_rate']:.2f}")

if __name__ == "__main__":
    test_phase4_projects()
```

These projects demonstrate sophisticated adaptive systems and robust fault tolerance patterns essential for production-grade LangGraph applications that can self-improve and handle failures gracefully.
