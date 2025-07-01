# LangChain Framework Structure

This document outlines the internal architecture and structure of the LangChain framework, explaining its core components, design patterns, and how different modules interact with each other.

## Framework Overview

LangChain is a modular framework designed for building applications with Large Language Models (LLMs). The framework follows a composable architecture where different components can be combined to create complex AI applications.

## Core Architecture

```
LangChain Framework
├── langchain-core/                     # Core abstractions and interfaces
│   ├── language_models/               # Base LLM abstractions
│   ├── prompts/                       # Prompt templates and management
│   ├── output_parsers/                # Response parsing utilities
│   ├── runnables/                     # Execution interfaces
│   ├── callbacks/                     # Event handling system
│   ├── memory/                        # Memory abstractions
│   └── tools/                         # Tool interfaces
│
├── langchain-community/               # Community integrations
│   ├── llms/                         # Third-party LLM integrations
│   ├── chat_models/                  # Chat model implementations
│   ├── embeddings/                   # Embedding model integrations
│   ├── vectorstores/                 # Vector database connectors
│   ├── document_loaders/             # Document loading utilities
│   ├── tools/                        # External tool integrations
│   └── utilities/                    # Helper utilities
│
├── langchain/                        # Main framework package
│   ├── chains/                       # Pre-built chain implementations
│   ├── agents/                       # Agent frameworks and executors
│   ├── memory/                       # Memory implementations
│   ├── schema/                       # Data schemas and types
│   ├── callbacks/                    # Callback implementations
│   └── cache/                        # Caching mechanisms
│
└── langchain-experimental/           # Experimental features
    ├── autonomous_agents/            # Advanced agent patterns
    ├── generative_agents/            # Generative agent implementations
    ├── plan_and_execute/             # Planning-based agents
    └── smart_llm_chain/              # Advanced chain patterns
```

## 1. Core Components (langchain-core)

### Language Models
```python
from langchain_core.language_models import BaseLanguageModel, BaseChatModel, BaseLLM

# Abstract base classes for all language models
class BaseLanguageModel:
    """Base class for all language models."""
    
class BaseLLM(BaseLanguageModel):
    """Base class for Large Language Models."""
    
class BaseChatModel(BaseLanguageModel):
    """Base class for Chat Models."""
```

**Key Components:**
- `BaseLanguageModel`: Abstract base for all language model types
- `BaseLLM`: Traditional completion-style models
- `BaseChatModel`: Conversational models with message-based interfaces
- `BaseEmbeddings`: Embedding model abstractions

### Prompts
```python
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate

# Prompt management system
class BasePromptTemplate:
    """Base class for all prompt templates."""
    
class PromptTemplate(BasePromptTemplate):
    """Simple string-based prompt template."""
    
class ChatPromptTemplate(BasePromptTemplate):
    """Chat-based prompt template with message roles."""
```

**Key Features:**
- Template-based prompt management
- Variable substitution
- Message formatting for chat models
- Prompt composition and chaining

### Runnables (LCEL - LangChain Expression Language)
```python
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda

# Core execution interface
class Runnable:
    """Base class for all runnable components."""
    
    def invoke(self, input: Any) -> Any:
        """Synchronous execution."""
        
    async def ainvoke(self, input: Any) -> Any:
        """Asynchronous execution."""
        
    def stream(self, input: Any) -> Iterator[Any]:
        """Streaming execution."""
        
    def batch(self, inputs: List[Any]) -> List[Any]:
        """Batch execution."""
```

**Key Concepts:**
- Universal execution interface
- Composable operations with `|` operator
- Parallel execution support
- Streaming and batching capabilities

### Output Parsers
```python
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser

class BaseOutputParser:
    """Base class for parsing LLM outputs."""
    
    def parse(self, text: str) -> Any:
        """Parse the output."""
        
    def get_format_instructions(self) -> str:
        """Get formatting instructions for the LLM."""
```

**Parser Types:**
- `PydanticOutputParser`: Structured data parsing
- `JSONOutputParser`: JSON format parsing
- `ListOutputParser`: List format parsing
- `DatetimeOutputParser`: Date/time parsing

### Callbacks
```python
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager

class BaseCallbackHandler:
    """Base callback handler for monitoring execution."""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running."""
```

## 2. Community Integrations (langchain-community)

### Language Model Integrations
```
langchain_community/llms/
├── openai.py                  # OpenAI GPT models
├── anthropic.py               # Anthropic Claude models
├── huggingface_hub.py         # Hugging Face models
├── ollama.py                  # Local Ollama models
├── azure_openai.py            # Azure OpenAI
├── google_palm.py             # Google PaLM
└── ...                        # 50+ model integrations
```

### Vector Store Integrations
```
langchain_community/vectorstores/
├── chroma.py                  # Chroma vector database
├── pinecone.py                # Pinecone vector database
├── faiss.py                   # Facebook AI Similarity Search
├── weaviate.py                # Weaviate vector database
├── qdrant.py                  # Qdrant vector database
├── milvus.py                  # Milvus vector database
└── ...                        # 40+ vector store integrations
```

### Document Loaders
```
langchain_community/document_loaders/
├── pdf.py                     # PDF document loading
├── web_base_loader.py         # Web page loading
├── csv_loader.py              # CSV file loading
├── text_loader.py             # Plain text loading
├── notion.py                  # Notion workspace loading
├── confluence.py              # Confluence loading
└── ...                        # 100+ document loaders
```

## 3. Main Framework (langchain)

### Chains
```python
from langchain.chains import LLMChain, SequentialChain, ConversationChain

# Chain hierarchy
class Chain:
    """Base chain class."""
    
class LLMChain(Chain):
    """Chain that calls an LLM with a prompt."""
    
class SequentialChain(Chain):
    """Chain that runs multiple chains in sequence."""
    
class ConversationChain(Chain):
    """Chain for conversational interactions."""
```

**Chain Types:**
- **Simple Chains**: `LLMChain`, `TransformChain`
- **Sequential Chains**: `SimpleSequentialChain`, `SequentialChain`
- **Routing Chains**: `MultiPromptChain`, `MultiRetrievalQAChain`
- **Map-Reduce Chains**: `MapReduceDocumentsChain`, `RefineDocumentsChain`

### Agents
```python
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent

# Agent architecture
class Agent:
    """Base agent class."""
    
class AgentExecutor:
    """Executor that runs agents with tools."""
    
# Agent types
def create_react_agent():
    """Creates a ReAct (Reasoning + Acting) agent."""
    
def create_tool_calling_agent():
    """Creates a tool-calling agent."""
```

**Agent Types:**
- **ReAct Agent**: Reasoning and acting in interleaved manner
- **Tool Calling Agent**: Uses function calling for tool selection
- **Plan and Execute Agent**: Plans then executes steps
- **Self-Ask Agent**: Asks follow-up questions to itself

### Memory Systems
```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

class BaseMemory:
    """Base memory class."""
    
class ConversationBufferMemory(BaseMemory):
    """Stores conversation history in a buffer."""
    
class ConversationSummaryMemory(BaseMemory):
    """Summarizes conversation history."""
```

**Memory Types:**
- **Buffer Memory**: Stores raw conversation history
- **Summary Memory**: Summarizes old conversations
- **Window Memory**: Keeps sliding window of recent messages
- **Entity Memory**: Tracks entities mentioned in conversation

## 4. Design Patterns and Architecture

### Composition Pattern
```python
# Components can be composed using the | operator (LCEL)
chain = prompt | llm | output_parser

# Parallel execution
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    response=prompt | llm,
    summary=prompt | llm | summarizer
)
```

### Factory Pattern
```python
# Different implementations behind common interfaces
from langchain_community.llms import OpenAI, Anthropic, HuggingFacePipeline

# All implement BaseLLM interface
llm1 = OpenAI()
llm2 = Anthropic()
llm3 = HuggingFacePipeline()
```

### Observer Pattern
```python
# Callbacks for monitoring and logging
from langchain_core.callbacks import StdOutCallbackHandler

callback = StdOutCallbackHandler()
chain = prompt | llm
chain.invoke({"input": "Hello"}, config={"callbacks": [callback]})
```

### Strategy Pattern
```python
# Different retrieval strategies
from langchain.retrievers import VectorStoreRetriever, MultiQueryRetriever

retriever1 = VectorStoreRetriever(vectorstore=chroma)
retriever2 = MultiQueryRetriever.from_llm(retriever=retriever1, llm=llm)
```

## 5. Data Flow Architecture

```
Input → Prompt Template → LLM → Output Parser → Output
  ↓         ↓              ↓         ↓           ↓
Memory ← Callbacks ← Callbacks ← Callbacks ← Callbacks
```

### Typical RAG Flow
```
Query → Query Transformation → Retrieval → Context Injection → LLM → Response
  ↓            ↓                  ↓             ↓           ↓        ↓
Memory ← Vector Store ← Document Loader ← Text Splitter ← Documents ← Source
```

### Agent Flow
```
Input → Agent → Tool Selection → Tool Execution → Result Processing → Output
  ↓       ↓           ↓              ↓               ↓              ↓
Memory ← Planning ← Tool Registry ← External APIs ← Result Parser ← Final Answer
```

## 6. Extension Points

### Custom Components
```python
# Custom LLM
class MyCustomLLM(BaseLLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Custom implementation
        pass

# Custom Tool
from langchain_core.tools import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Useful for..."
    
    def _run(self, query: str) -> str:
        # Tool implementation
        pass
```

### Custom Chains
```python
from langchain.chains.base import Chain

class MyCustomChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["input"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["output"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Custom chain logic
        pass
```

## 7. Configuration and Environment

### Environment Variables
```python
import os
from langchain_community.llms import OpenAI

# API key configuration
os.environ["OPENAI_API_KEY"] = "your-api-key"
llm = OpenAI()

# Model configuration
llm = OpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=150
)
```

### Settings Management
```python
from langchain.globals import set_debug, set_verbose

# Enable debugging
set_debug(True)
set_verbose(True)

# Configure callbacks globally
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "Hello"})
    print(f"Total cost: ${cb.total_cost}")
```

This framework structure shows how LangChain is architected as a modular, extensible system that separates concerns between core abstractions, community integrations, and high-level patterns, making it easy to build complex LLM applications while maintaining flexibility and reusability.
