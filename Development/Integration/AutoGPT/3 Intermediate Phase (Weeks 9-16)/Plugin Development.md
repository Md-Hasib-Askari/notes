# Plugin Development

## Overview
This section teaches you to develop custom plugins for AutoGPT, extending its capabilities with specialized functionality, API integrations, and database connections while maintaining security best practices.

## Learning Objectives
- Analyze existing plugins and understand their structure
- Create custom plugins for specific tasks
- Master plugin API and integration patterns
- Implement security best practices for plugins
- Build plugins with database and external API connections

## 1. Studying Existing Plugins

### Plugin Directory Structure
```
plugins/
├── __init__.py
├── web_search/
│   ├── __init__.py
│   ├── plugin.py
│   └── requirements.txt
├── file_operations/
│   ├── __init__.py
│   ├── plugin.py
│   └── config.yaml
└── email_integration/
    ├── __init__.py
    ├── plugin.py
    └── auth.py
```

### Analyzing Built-in Plugins
**Web Search Plugin Structure**:
```python
# plugins/web_search/plugin.py
from autogpt.plugins import BasePlugin

class WebSearchPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "WebSearch"
        self.version = "1.0.0"
        self.description = "Search the web for information"
        
    def can_handle(self, command):
        return command.name in ["web_search", "google_search"]
    
    def handle(self, command):
        # Implementation details
        return self.search_web(command.args["query"])
```

### Common Plugin Patterns
- **Command-based**: Respond to specific commands
- **Event-driven**: React to agent events
- **Service-based**: Provide ongoing services
- **Integration-based**: Connect to external systems

## 2. Creating Simple Custom Plugins

### Basic Plugin Template
```python
# plugins/my_custom_plugin/plugin.py
from autogpt.plugins import BasePlugin
from typing import Dict, Any

class MyCustomPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "MyCustomPlugin"
        self.version = "1.0.0"
        self.description = "Custom plugin for specific tasks"
        self.commands = ["custom_command", "helper_function"]
    
    def can_handle(self, command) -> bool:
        return command.name in self.commands
    
    def handle(self, command) -> Dict[str, Any]:
        if command.name == "custom_command":
            return self.execute_custom_logic(command.args)
        elif command.name == "helper_function":
            return self.helper_function(command.args)
    
    def execute_custom_logic(self, args):
        # Your custom implementation
        return {"status": "success", "result": "Custom logic executed"}
```

### Weather Plugin Example
```python
# plugins/weather/plugin.py
import requests
from autogpt.plugins import BasePlugin

class WeatherPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "Weather"
        self.description = "Get weather information"
        self.api_key = self.config.get("weather_api_key")
    
    def can_handle(self, command):
        return command.name == "get_weather"
    
    def handle(self, command):
        location = command.args.get("location")
        return self.get_weather(location)
    
    def get_weather(self, location):
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric"
        }
        response = requests.get(url, params=params)
        return response.json()
```

### Plugin Registration
```python
# plugins/__init__.py
from .weather.plugin import WeatherPlugin
from .database.plugin import DatabasePlugin

AVAILABLE_PLUGINS = [
    WeatherPlugin,
    DatabasePlugin
]
```

## 3. Plugin API and Integration Patterns

### Plugin Base Class Interface
```python
class BasePlugin:
    def __init__(self):
        self.name = ""
        self.version = ""
        self.description = ""
        self.config = self.load_config()
    
    def can_handle(self, command) -> bool:
        """Check if plugin can handle the command"""
        raise NotImplementedError
    
    def handle(self, command) -> Dict[str, Any]:
        """Execute the command"""
        raise NotImplementedError
    
    def validate_args(self, args) -> bool:
        """Validate command arguments"""
        return True
    
    def get_schema(self) -> Dict:
        """Return command schema for validation"""
        return {}
```

### Event-Driven Pattern
```python
class EventDrivenPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.event_handlers = {
            "task_started": self.on_task_started,
            "task_completed": self.on_task_completed,
            "error_occurred": self.on_error
        }
    
    def on_task_started(self, event_data):
        # Handle task start event
        pass
    
    def on_task_completed(self, event_data):
        # Handle task completion
        pass
```

### Async Plugin Pattern
```python
import asyncio
from autogpt.plugins import AsyncBasePlugin

class AsyncPlugin(AsyncBasePlugin):
    async def handle(self, command):
        # Asynchronous operations
        result = await self.async_operation(command.args)
        return result
    
    async def async_operation(self, args):
        # Long-running async task
        await asyncio.sleep(1)
        return {"status": "completed"}
```

## 4. Security Considerations

### Input Validation
```python
def validate_command_args(self, args):
    """Validate and sanitize input arguments"""
    if "query" in args:
        # Sanitize SQL injection attempts
        args["query"] = self.sanitize_sql(args["query"])
    
    if "file_path" in args:
        # Prevent directory traversal
        if ".." in args["file_path"]:
            raise SecurityError("Directory traversal not allowed")
    
    return args

def sanitize_sql(self, query):
    """Basic SQL injection prevention"""
    dangerous_chars = ["'", '"', ";", "--", "/*", "*/"]
    for char in dangerous_chars:
        query = query.replace(char, "")
    return query
```

### Permission Control
```python
class SecurePlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.required_permissions = ["file_read", "network_access"]
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    def check_permissions(self):
        for permission in self.required_permissions:
            if not self.has_permission(permission):
                raise PermissionError(f"Missing permission: {permission}")
    
    def handle(self, command):
        self.check_permissions()
        return super().handle(command)
```

### Rate Limiting
```python
from time import time
from collections import defaultdict

class RateLimitedPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.call_counts = defaultdict(list)
        self.rate_limit = 10  # calls per minute
    
    def check_rate_limit(self, user_id):
        now = time()
        calls = self.call_counts[user_id]
        # Remove calls older than 1 minute
        calls[:] = [call_time for call_time in calls if now - call_time < 60]
        
        if len(calls) >= self.rate_limit:
            raise RateLimitError("Rate limit exceeded")
        
        calls.append(now)
```

## 5. Database and External API Integration

### Database Plugin Example
```python
import sqlite3
from autogpt.plugins import BasePlugin

class DatabasePlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "Database"
        self.db_path = self.config.get("db_path", "data.db")
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def handle(self, command):
        if command.name == "save_task":
            return self.save_task(command.args)
        elif command.name == "get_tasks":
            return self.get_tasks()
    
    def save_task(self, args):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tasks (name, status) VALUES (?, ?)",
                (args["name"], args["status"])
            )
            return {"status": "success", "message": "Task saved"}
```

### External API Plugin
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class APIPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.session = self.create_session()
        self.base_url = self.config.get("api_base_url")
        self.api_key = self.config.get("api_key")
    
    def create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def make_api_call(self, endpoint, data=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = self.session.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
```

### Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

class AdvancedDatabasePlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.engine = create_engine(
            self.config.get("database_url"),
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20
        )
    
    def execute_query(self, query, params=None):
        with self.engine.connect() as conn:
            result = conn.execute(query, params or {})
            return result.fetchall()
```

## 6. Plugin Testing and Deployment

### Unit Testing
```python
import unittest
from unittest.mock import Mock, patch
from plugins.weather.plugin import WeatherPlugin

class TestWeatherPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = WeatherPlugin()
    
    @patch('requests.get')
    def test_get_weather(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {"temp": 20, "condition": "sunny"}
        mock_get.return_value = mock_response
        
        result = self.plugin.get_weather("London")
        self.assertEqual(result["temp"], 20)
```

### Plugin Configuration
```yaml
# plugins/weather/config.yaml
name: "Weather"
version: "1.0.0"
dependencies:
  - requests>=2.25.0
settings:
  api_key: ${WEATHER_API_KEY}
  cache_duration: 300
  rate_limit: 100
```

## Assessment Checklist
- [ ] Analyzed existing plugin structures and patterns
- [ ] Created custom plugins for specific tasks
- [ ] Implemented plugin API and integration patterns
- [ ] Applied security best practices in plugin development
- [ ] Built plugins with database and external API connections
- [ ] Tested plugins thoroughly
- [ ] Ready for advanced usage patterns
