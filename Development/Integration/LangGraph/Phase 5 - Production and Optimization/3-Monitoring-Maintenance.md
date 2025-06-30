# Monitoring and Maintenance for LangGraph

Production monitoring, analytics, and version management strategies for LangGraph applications in production environments.

## Production Monitoring

### Performance Metrics

```python
# src/monitoring/metrics.py
import time
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
from collections import defaultdict, deque

@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.retention_period = timedelta(hours=retention_hours)
        self.node_execution_times = defaultdict(list)
        self.graph_execution_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
    
    def record_execution_time(self, node_name: str, execution_time: float, labels: Dict[str, str] = None):
        """Record node execution time"""
        metric = MetricPoint(
            timestamp=datetime.now(),
            value=execution_time,
            labels=labels or {}
        )
        self.metrics[f"node_execution_time_{node_name}"].append(metric)
        self.node_execution_times[node_name].append(execution_time)
    
    def record_graph_execution(self, graph_id: str, success: bool = True):
        """Record graph execution"""
        self.graph_execution_counts[graph_id] += 1
        
        metric = MetricPoint(
            timestamp=datetime.now(),
            value=1,
            labels={"graph_id": graph_id, "status": "success" if success else "error"}
        )
        self.metrics["graph_executions"].append(metric)
        
        if not success:
            self.error_counts[graph_id] += 1
    
    def record_system_metrics(self):
        """Record system resource usage"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        self.metrics["system_cpu"].append(MetricPoint(
            timestamp=datetime.now(),
            value=cpu_percent
        ))
        
        self.metrics["system_memory"].append(MetricPoint(
            timestamp=datetime.now(),
            value=memory.percent
        ))
    
    def get_average_execution_time(self, node_name: str, minutes: int = 60) -> float:
        """Get average execution time for a node in the last N minutes"""
        if node_name not in self.node_execution_times:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics[f"node_execution_time_{node_name}"]
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return 0.0
        
        return sum(m.value for m in recent_metrics) / len(recent_metrics)
    
    def get_error_rate(self, graph_id: str = None, minutes: int = 60) -> float:
        """Calculate error rate percentage"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        if graph_id:
            recent_executions = [
                m for m in self.metrics["graph_executions"]
                if m.timestamp > cutoff_time and m.labels.get("graph_id") == graph_id
            ]
        else:
            recent_executions = [
                m for m in self.metrics["graph_executions"]
                if m.timestamp > cutoff_time
            ]
        
        if not recent_executions:
            return 0.0
        
        error_count = sum(1 for m in recent_executions if m.labels.get("status") == "error")
        return (error_count / len(recent_executions)) * 100

# Performance monitoring decorator
metrics_collector = MetricsCollector()

def monitor_performance(node_name: str = None):
    """Decorator to monitor node performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            name = node_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                metrics_collector.record_execution_time(name, execution_time, {"status": "success"})
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                metrics_collector.record_execution_time(name, execution_time, {"status": "error"})
                raise
        
        return wrapper
    return decorator

@monitor_performance("data_processor")
async def monitored_data_processor(state: Dict[str, Any]) -> Dict[str, Any]:
    """Example monitored node"""
    # Simulate processing
    await asyncio.sleep(0.1)
    return {**state, "processed": True}
```

### Error Tracking and Alerting

```python
# src/monitoring/alerting.py
import asyncio
import aiohttp
from typing import List, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False

class AlertManager:
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Callable] = []
        self.notification_channels = []
        self.alert_history = []
    
    def add_alert_rule(self, rule_func: Callable):
        """Add a custom alert rule"""
        self.alert_rules.append(rule_func)
    
    def add_notification_channel(self, channel):
        """Add notification channel (email, slack, etc.)"""
        self.notification_channels.append(channel)
    
    async def check_alerts(self):
        """Check all alert rules and trigger notifications"""
        while True:
            new_alerts = []
            
            for rule in self.alert_rules:
                try:
                    alert = await rule(metrics_collector)
                    if alert:
                        new_alerts.append(alert)
                except Exception as e:
                    print(f"Error checking alert rule: {e}")
            
            for alert in new_alerts:
                await self._handle_alert(alert)
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _handle_alert(self, alert: Alert):
        """Handle a new alert"""
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                await channel.send_alert(alert)
            except Exception as e:
                print(f"Failed to send alert via {channel}: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                break

# Alert rules
async def high_error_rate_rule(metrics: MetricsCollector) -> Alert:
    """Alert if error rate exceeds 10% in the last 10 minutes"""
    error_rate = metrics.get_error_rate(minutes=10)
    
    if error_rate > 10:
        return Alert(
            id=f"high_error_rate_{datetime.now().timestamp()}",
            severity=AlertSeverity.HIGH,
            title="High Error Rate Detected",
            description=f"Error rate is {error_rate:.2f}% in the last 10 minutes",
            timestamp=datetime.now(),
            metadata={"error_rate": error_rate}
        )
    return None

async def slow_execution_rule(metrics: MetricsCollector) -> Alert:
    """Alert if average execution time is too high"""
    for node_name in metrics.node_execution_times.keys():
        avg_time = metrics.get_average_execution_time(node_name, minutes=15)
        
        if avg_time > 5.0:  # 5 seconds threshold
            return Alert(
                id=f"slow_execution_{node_name}_{datetime.now().timestamp()}",
                severity=AlertSeverity.MEDIUM,
                title=f"Slow Execution: {node_name}",
                description=f"Average execution time is {avg_time:.2f}s",
                timestamp=datetime.now(),
                metadata={"node_name": node_name, "avg_time": avg_time}
            )
    return None

# Notification channels
class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_alert(self, alert: Alert):
        """Send alert to Slack"""
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning", 
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=payload)

# Initialize alert system
alert_manager = AlertManager()
alert_manager.add_alert_rule(high_error_rate_rule)
alert_manager.add_alert_rule(slow_execution_rule)
```

### Usage Analytics

```python
# src/monitoring/analytics.py
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json

class UsageAnalytics:
    def __init__(self):
        self.graph_usage = defaultdict(int)
        self.node_usage = defaultdict(int)
        self.user_sessions = defaultdict(list)
        self.execution_patterns = []
    
    def track_graph_execution(self, graph_id: str, user_id: str = None, metadata: Dict[str, Any] = None):
        """Track graph execution for analytics"""
        self.graph_usage[graph_id] += 1
        
        execution_record = {
            "graph_id": graph_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.execution_patterns.append(execution_record)
        
        if user_id:
            self.user_sessions[user_id].append(execution_record)
    
    def track_node_execution(self, node_name: str, graph_id: str):
        """Track individual node executions"""
        self.node_usage[f"{graph_id}:{node_name}"] += 1
    
    def get_popular_graphs(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get most popular graphs in the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_executions = defaultdict(int)
        for execution in self.execution_patterns:
            exec_time = datetime.fromisoformat(execution["timestamp"])
            if exec_time > cutoff_date:
                recent_executions[execution["graph_id"]] += 1
        
        return sorted(
            [{"graph_id": k, "executions": v} for k, v in recent_executions.items()],
            key=lambda x: x["executions"],
            reverse=True
        )
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        user_executions = self.user_sessions.get(user_id, [])
        
        if not user_executions:
            return {"total_executions": 0}
        
        graph_counts = defaultdict(int)
        for execution in user_executions:
            graph_counts[execution["graph_id"]] += 1
        
        return {
            "total_executions": len(user_executions),
            "favorite_graphs": dict(graph_counts),
            "first_execution": user_executions[0]["timestamp"],
            "last_execution": user_executions[-1]["timestamp"]
        }

analytics = UsageAnalytics()
```

## Version Management

### Graph Versioning Strategies

```python
# src/versioning/manager.py
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class GraphVersion:
    version_id: str
    graph_id: str
    version_number: str
    graph_definition: Dict[str, Any]
    created_at: datetime
    created_by: str
    description: str
    is_active: bool = False
    rollback_info: Optional[Dict[str, Any]] = None

class GraphVersionManager:
    def __init__(self):
        self.versions: Dict[str, List[GraphVersion]] = {}
        self.active_versions: Dict[str, str] = {}  # graph_id -> version_id
    
    def create_version(self, graph_id: str, graph_definition: Dict[str, Any], 
                      created_by: str, description: str = "") -> str:
        """Create a new version of a graph"""
        
        # Generate version ID based on content hash
        content_hash = hashlib.sha256(
            json.dumps(graph_definition, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        # Generate version number
        existing_versions = self.versions.get(graph_id, [])
        version_number = f"v{len(existing_versions) + 1}.0"
        
        version = GraphVersion(
            version_id=f"{graph_id}_{content_hash}",
            graph_id=graph_id,
            version_number=version_number,
            graph_definition=graph_definition,
            created_at=datetime.now(),
            created_by=created_by,
            description=description
        )
        
        if graph_id not in self.versions:
            self.versions[graph_id] = []
        
        self.versions[graph_id].append(version)
        return version.version_id
    
    def activate_version(self, graph_id: str, version_id: str) -> bool:
        """Activate a specific version"""
        versions = self.versions.get(graph_id, [])
        
        for version in versions:
            if version.version_id == version_id:
                # Deactivate current active version
                current_active = self.active_versions.get(graph_id)
                if current_active:
                    self._set_version_active(graph_id, current_active, False)
                
                # Activate new version
                version.is_active = True
                self.active_versions[graph_id] = version_id
                return True
        
        return False
    
    def get_active_version(self, graph_id: str) -> Optional[GraphVersion]:
        """Get the currently active version"""
        active_version_id = self.active_versions.get(graph_id)
        if not active_version_id:
            return None
        
        versions = self.versions.get(graph_id, [])
        return next((v for v in versions if v.version_id == active_version_id), None)
    
    def rollback_to_version(self, graph_id: str, target_version_id: str, 
                           rollback_reason: str) -> bool:
        """Rollback to a previous version"""
        current_version = self.get_active_version(graph_id)
        
        if self.activate_version(graph_id, target_version_id):
            # Store rollback information
            if current_version:
                current_version.rollback_info = {
                    "rolled_back_at": datetime.now().isoformat(),
                    "rollback_reason": rollback_reason,
                    "rolled_back_to": target_version_id
                }
            return True
        
        return False

# A/B Testing Framework
class ABTestManager:
    def __init__(self, version_manager: GraphVersionManager):
        self.version_manager = version_manager
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_ab_test(self, test_id: str, graph_id: str, 
                      version_a: str, version_b: str, 
                      traffic_split: float = 0.5) -> bool:
        """Create an A/B test between two versions"""
        
        # Validate versions exist
        versions = self.version_manager.versions.get(graph_id, [])
        version_ids = [v.version_id for v in versions]
        
        if version_a not in version_ids or version_b not in version_ids:
            return False
        
        self.active_tests[test_id] = {
            "graph_id": graph_id,
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "started_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.test_results[test_id] = []
        return True
    
    def get_version_for_request(self, test_id: str, user_id: str = None) -> str:
        """Determine which version to use for a request"""
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        
        # Simple hash-based assignment for consistent user experience
        if user_id:
            hash_value = hash(user_id) % 100
            return test["version_a"] if hash_value < (test["traffic_split"] * 100) else test["version_b"]
        
        # Random assignment for anonymous users
        import random
        return test["version_a"] if random.random() < test["traffic_split"] else test["version_b"]
    
    def record_test_result(self, test_id: str, version_id: str, 
                          success: bool, execution_time: float):
        """Record test execution result"""
        if test_id in self.test_results:
            self.test_results[test_id].append({
                "version_id": version_id,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })
    
    def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if test_id not in self.test_results:
            return {}
        
        results = self.test_results[test_id]
        test = self.active_tests[test_id]
        
        version_a_results = [r for r in results if r["version_id"] == test["version_a"]]
        version_b_results = [r for r in results if r["version_id"] == test["version_b"]]
        
        def calculate_metrics(version_results):
            if not version_results:
                return {"success_rate": 0, "avg_execution_time": 0, "sample_size": 0}
            
            success_count = sum(1 for r in version_results if r["success"])
            avg_time = sum(r["execution_time"] for r in version_results) / len(version_results)
            
            return {
                "success_rate": (success_count / len(version_results)) * 100,
                "avg_execution_time": avg_time,
                "sample_size": len(version_results)
            }
        
        return {
            "test_id": test_id,
            "version_a_metrics": calculate_metrics(version_a_results),
            "version_b_metrics": calculate_metrics(version_b_results),
            "total_executions": len(results)
        }

# Usage example
version_manager = GraphVersionManager()
ab_test_manager = ABTestManager(version_manager)

# Create versions
v1_id = version_manager.create_version("my_graph", {"nodes": ["a", "b"]}, "developer", "Initial version")
v2_id = version_manager.create_version("my_graph", {"nodes": ["a", "b", "c"]}, "developer", "Added node c")

# Start A/B test
ab_test_manager.create_ab_test("test_1", "my_graph", v1_id, v2_id, 0.5)
```

## Best Practices

1. **Metrics Collection**: Implement comprehensive performance and system metrics
2. **Alerting**: Set up proactive alerting with appropriate thresholds
3. **Analytics**: Track usage patterns for optimization insights
4. **Version Control**: Maintain proper versioning with rollback capabilities
5. **A/B Testing**: Use controlled experiments to validate improvements
6. **Monitoring Automation**: Automate monitoring tasks and alert responses
