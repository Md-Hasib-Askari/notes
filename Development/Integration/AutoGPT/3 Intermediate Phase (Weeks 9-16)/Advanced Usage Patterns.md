# Advanced Usage Patterns

## Overview
This section explores sophisticated AutoGPT usage patterns including multi-agent workflows, custom evaluation systems, code generation, external integrations, and robust error handling.

## Learning Objectives
- Chain multiple AutoGPT instances for complex workflows
- Implement custom evaluation metrics for task success
- Use AutoGPT for code generation and testing
- Integrate with external tools and services
- Handle error scenarios with retry logic

## 1. Chaining Multiple AutoGPT Instances

### Multi-Agent Workflow Architecture
```python
class WorkflowOrchestrator:
    def __init__(self):
        self.agents = {
            "researcher": AutoGPTAgent("research_config.yaml"),
            "analyst": AutoGPTAgent("analysis_config.yaml"),
            "writer": AutoGPTAgent("writing_config.yaml")
        }
        self.workflow_state = {}
    
    def execute_research_workflow(self, topic):
        # Step 1: Research phase
        research_result = self.agents["researcher"].execute({
            "goal": f"Research comprehensive information about {topic}",
            "output_format": "structured_data"
        })
        
        # Step 2: Analysis phase
        analysis_result = self.agents["analyst"].execute({
            "goal": "Analyze research data and extract key insights",
            "input_data": research_result,
            "output_format": "analytical_summary"
        })
        
        # Step 3: Writing phase
        final_report = self.agents["writer"].execute({
            "goal": "Create professional report from analysis",
            "input_data": analysis_result,
            "output_format": "markdown_report"
        })
        
        return final_report
```

### Pipeline Configuration
```yaml
# workflow_pipeline.yaml
pipeline:
  stages:
    - name: "data_collection"
      agent: "collector_agent"
      config: "collector_config.yaml"
      output: "raw_data"
    
    - name: "data_processing"
      agent: "processor_agent"
      config: "processor_config.yaml"
      input: "raw_data"
      output: "processed_data"
    
    - name: "report_generation"
      agent: "writer_agent"
      config: "writer_config.yaml"
      input: "processed_data"
      output: "final_report"
```

### Parallel Agent Execution
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelAgentManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def run_parallel_tasks(self, tasks):
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in tasks:
            future = loop.run_in_executor(
                self.executor, 
                self.execute_agent_task, 
                task
            )
            futures.append(future)
        
        results = await asyncio.gather(*futures)
        return self.merge_results(results)
```

## 2. Custom Evaluation Metrics

### Evaluation Framework
```python
class TaskEvaluator:
    def __init__(self):
        self.metrics = {
            "completion": CompletionMetric(),
            "accuracy": AccuracyMetric(),
            "efficiency": EfficiencyMetric(),
            "quality": QualityMetric()
        }
    
    def evaluate_task(self, task_result, expected_outcome):
        scores = {}
        for metric_name, metric in self.metrics.items():
            scores[metric_name] = metric.calculate(task_result, expected_outcome)
        
        return {
            "overall_score": self.calculate_weighted_score(scores),
            "detailed_scores": scores,
            "recommendations": self.generate_recommendations(scores)
        }
```

### Custom Metrics Implementation
```python
class CompletionMetric:
    def calculate(self, result, expected):
        completed_goals = len(result.get("completed_goals", []))
        total_goals = len(expected.get("goals", []))
        return (completed_goals / total_goals) * 100 if total_goals > 0 else 0

class QualityMetric:
    def __init__(self):
        self.quality_criteria = [
            "relevance", "accuracy", "completeness", "clarity"
        ]
    
    def calculate(self, result, expected):
        quality_scores = []
        for criterion in self.quality_criteria:
            score = self.assess_criterion(result, criterion)
            quality_scores.append(score)
        return sum(quality_scores) / len(quality_scores)

class EfficiencyMetric:
    def calculate(self, result, expected):
        actual_time = result.get("execution_time", 0)
        expected_time = expected.get("max_time", float('inf'))
        api_calls = result.get("api_calls", 0)
        
        time_efficiency = min(100, (expected_time / actual_time) * 100)
        cost_efficiency = max(0, 100 - (api_calls * 0.1))
        
        return (time_efficiency + cost_efficiency) / 2
```

## 3. Code Generation and Testing

### Code Generation Agent
```python
class CodeGenerationAgent:
    def __init__(self):
        self.templates = self.load_code_templates()
        self.test_runner = TestRunner()
    
    def generate_code(self, specification):
        prompt = f"""
        Generate Python code based on this specification:
        {specification}
        
        Requirements:
        - Include proper error handling
        - Add comprehensive docstrings
        - Follow PEP 8 standards
        - Include basic unit tests
        """
        
        code = self.agent.execute(prompt)
        return self.validate_and_test_code(code)
    
    def validate_and_test_code(self, code):
        # Syntax validation
        if not self.is_valid_syntax(code):
            return self.fix_syntax_errors(code)
        
        # Run tests
        test_results = self.test_runner.run_tests(code)
        
        if test_results["passed"]:
            return code
        else:
            return self.fix_failing_tests(code, test_results)
```

### Automated Testing Pipeline
```python
class AutomatedTestingAgent:
    def generate_tests(self, code, test_type="unit"):
        test_prompt = f"""
        Generate {test_type} tests for this code:
        {code}
        
        Include:
        - Happy path tests
        - Edge case tests
        - Error condition tests
        - Mock external dependencies
        """
        
        tests = self.agent.execute(test_prompt)
        return self.validate_tests(tests)
    
    def run_continuous_testing(self, code_repo):
        while True:
            changes = self.detect_code_changes(code_repo)
            if changes:
                test_results = self.run_full_test_suite(code_repo)
                self.report_results(test_results)
            time.sleep(300)  # Check every 5 minutes
```

## 4. External Tool Integration

### Tool Integration Framework
```python
class ExternalToolManager:
    def __init__(self):
        self.tools = {
            "slack": SlackIntegration(),
            "jira": JiraIntegration(),
            "github": GitHubIntegration(),
            "database": DatabaseIntegration()
        }
    
    def execute_tool_command(self, tool_name, command, args):
        if tool_name not in self.tools:
            raise ToolNotFoundError(f"Tool {tool_name} not available")
        
        tool = self.tools[tool_name]
        return tool.execute(command, args)
```

### Service Integration Examples
```python
class SlackIntegration:
    def __init__(self, token):
        self.client = WebClient(token=token)
    
    def send_message(self, channel, message):
        return self.client.chat_postMessage(
            channel=channel,
            text=message
        )

class GitHubIntegration:
    def __init__(self, token):
        self.github = Github(token)
    
    def create_issue(self, repo_name, title, body):
        repo = self.github.get_repo(repo_name)
        return repo.create_issue(title=title, body=body)
    
    def create_pull_request(self, repo_name, title, body, head, base):
        repo = self.github.get_repo(repo_name)
        return repo.create_pull(
            title=title, 
            body=body, 
            head=head, 
            base=base
        )
```

### API Gateway Pattern
```python
class APIGateway:
    def __init__(self):
        self.rate_limiters = {}
        self.circuit_breakers = {}
    
    def make_request(self, service, endpoint, data=None):
        # Rate limiting
        if not self.check_rate_limit(service):
            raise RateLimitExceeded()
        
        # Circuit breaker
        if self.is_circuit_open(service):
            raise CircuitBreakerOpen()
        
        try:
            response = self.execute_request(service, endpoint, data)
            self.record_success(service)
            return response
        except Exception as e:
            self.record_failure(service)
            raise
```

## 5. Error Handling and Retry Logic

### Robust Error Handling
```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustAgent:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def execute_with_retry(self, task):
        try:
            return self.execute_task(task)
        except (APIError, NetworkError) as e:
            self.log_error(f"Retryable error: {e}")
            raise
        except CriticalError as e:
            self.log_error(f"Critical error, not retrying: {e}")
            raise StopRetrying(e)
```

### Circuit Breaker Implementation
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen()
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

### Error Recovery Strategies
```python
class ErrorRecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            "API_RATE_LIMIT": self.handle_rate_limit,
            "NETWORK_TIMEOUT": self.handle_timeout,
            "MEMORY_ERROR": self.handle_memory_error,
            "PERMISSION_DENIED": self.handle_permission_error
        }
    
    def recover_from_error(self, error_type, context):
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](context)
        else:
            return self.default_recovery(error_type, context)
    
    def handle_rate_limit(self, context):
        wait_time = context.get("retry_after", 60)
        time.sleep(wait_time)
        return "RETRY"
    
    def handle_memory_error(self, context):
        # Clear caches, reduce batch size
        self.clear_memory_caches()
        context["batch_size"] = context.get("batch_size", 100) // 2
        return "RETRY_WITH_REDUCED_LOAD"
```

## Assessment Checklist
- [ ] Implemented multi-agent workflow orchestration
- [ ] Created custom evaluation metrics for task assessment
- [ ] Built code generation and testing automation
- [ ] Integrated with external tools and services
- [ ] Implemented robust error handling and retry logic
- [ ] Applied circuit breaker and recovery patterns
- [ ] Ready for advanced phase topics
