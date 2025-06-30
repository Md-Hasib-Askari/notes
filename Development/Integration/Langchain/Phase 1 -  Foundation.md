# Phase 1: Foundation - Detailed Notes
*Weeks 1-2*

## 1.1 Setup and Environment

### Installation
```bash
# Basic installation
pip install langchain

# With OpenAI support
pip install langchain-openai

# With additional providers
pip install langchain-anthropic langchain-google-genai

# For development
pip install langchain-community python-dotenv
```

### Environment Setup
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Project Structure
```
my-langchain-project/
├── .env
├── requirements.txt
├── main.py
├── config/
│   └── settings.py
└── examples/
    └── hello_world.py
```

### Hello World Example
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create a message and get response
message = HumanMessage(content="Hello! What is LangChain?")
response = llm.invoke([message])
print(response.content)
```

### Key Concepts to Understand
- **API Keys**: Never hardcode them, always use environment variables
- **Rate Limits**: Be aware of provider limitations
- **Cost Management**: Monitor token usage from day one

---

## 1.2 Core Components

### LLMs vs Chat Models

#### LLMs (Language Models)
- Take text input, return text output
- Good for completion tasks
- Examples: GPT-3, Claude-instant

```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.9)
response = llm.invoke("The future of AI is")
print(response)
```

#### Chat Models
- Take messages as input, return messages
- Designed for conversational interfaces
- Support system/human/AI message roles

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is machine learning?")
]
response = chat.invoke(messages)
print(response.content)
```

### Prompts and Prompt Templates

#### Basic Prompt Templates
```python
from langchain.prompts import PromptTemplate

# Simple template
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate.from_template(template)
formatted_prompt = prompt.format(product="colorful socks")
print(formatted_prompt)
```

#### Chat Prompt Templates
```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])

formatted_prompt = template.format_messages(
    input_language="English",
    output_language="French",
    text="I love programming."
)
```

#### Advanced Prompt Features
```python
from langchain.prompts import PromptTemplate

# Multiple variables
template = """
Role: You are a {role}
Task: {task}
Context: {context}
Question: {question}

Please provide a detailed response.
"""

prompt = PromptTemplate(
    input_variables=["role", "task", "context", "question"],
    template=template
)
```

### Output Parsers

#### String Output Parser
```python
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Chain components together
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# Create chain
chain = prompt | model | output_parser
result = chain.invoke({"topic": "programming"})
print(result)  # Returns string instead of AIMessage object
```

#### Structured Output Parsers
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(description="person's age")
    occupation: str = Field(description="person's occupation")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="Extract information about a person.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

### Basic Chains

#### Simple Chain
```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("eco-friendly water bottles")
print(result)
```

#### Using LCEL (LangChain Expression Language)
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# Modern way using LCEL
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# Chain using pipe operator
chain = prompt | model | output_parser
result = chain.invoke({"topic": "quantum computing"})
```

---

## 1.3 First Projects

### Project 1: Simple Q&A Chatbot

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

class SimpleQABot:
    def __init__(self):
        self.model = ChatOpenAI(temperature=0.7)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer questions clearly and concisely."),
            ("human", "{question}")
        ])
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.model | self.output_parser
    
    def ask(self, question: str) -> str:
        return self.chain.invoke({"question": question})

# Usage
bot = SimpleQABot()
while True:
    user_input = input("Ask a question (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    response = bot.ask(user_input)
    print(f"Bot: {response}\n")
```

### Project 2: Text Summarization Tool

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

class TextSummarizer:
    def __init__(self):
        self.model = ChatOpenAI(temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(
            """
            Please summarize the following text in {summary_type} format:
            
            Text: {text}
            
            Summary:
            """
        )
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def summarize(self, text: str, summary_type: str = "bullet points") -> str:
        return self.chain.invoke({
            "text": text,
            "summary_type": summary_type
        })

# Usage example
summarizer = TextSummarizer()
long_text = """
LangChain is a framework for developing applications powered by language models.
It enables applications that are data-aware and agentic, allowing language models
to connect with other sources of data and interact with their environment.
The main value props of LangChain are: 1) Components for working with language models,
2) Chains for combining components, 3) Agents for high-level decision making.
"""

summary = summarizer.summarize(long_text, "paragraph")
print(summary)
```

### Project 3: Basic Prompt Templating System

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
import json

class PromptTemplateManager:
    def __init__(self):
        self.model = ChatOpenAI(temperature=0.7)
        self.output_parser = StrOutputParser()
        self.templates = {
            "email": """
            Write a professional email with the following details:
            Subject: {subject}
            Recipient: {recipient}
            Tone: {tone}
            Main message: {message}
            """,
            "story": """
            Write a short story with these elements:
            Genre: {genre}
            Main character: {character}
            Setting: {setting}
            Theme: {theme}
            Length: {length} words approximately
            """,
            "explanation": """
            Explain {topic} to someone who is a {audience_level}.
            Make it {style} and include {num_examples} examples.
            """
        }
    
    def generate_content(self, template_type: str, **kwargs) -> str:
        if template_type not in self.templates:
            raise ValueError(f"Template type '{template_type}' not found")
        
        template = PromptTemplate.from_template(self.templates[template_type])
        chain = template | self.model | self.output_parser
        
        return chain.invoke(kwargs)
    
    def add_template(self, name: str, template: str):
        self.templates[name] = template
    
    def list_templates(self):
        return list(self.templates.keys())

# Usage
manager = PromptTemplateManager()

# Generate an email
email = manager.generate_content(
    "email",
    subject="Project Update",
    recipient="team",
    tone="professional but friendly",
    message="The new feature is ready for testing"
)
print("Generated Email:")
print(email)

# Generate a story
story = manager.generate_content(
    "story",
    genre="science fiction",
    character="a curious robot",
    setting="Mars colony in 2150",
    theme="friendship",
    length="200"
)
print("\nGenerated Story:")
print(story)
```

---

## Key Takeaways from Phase 1

### Essential Concepts Mastered
1. **Environment Setup**: Proper configuration and security practices
2. **Core Components**: Understanding LLMs, Chat Models, Prompts, and Output Parsers
3. **Chain Basics**: Sequential operations and LCEL syntax
4. **Practical Applications**: Building real, working applications

### Best Practices Learned
- Always use environment variables for API keys
- Start with simple chains before moving to complex ones
- Use appropriate temperature settings for different tasks
- Implement proper error handling from the beginning

### Common Mistakes to Avoid
- Hardcoding API keys in source code
- Not handling API rate limits
- Ignoring token costs and usage
- Making prompts too complex initially

### Next Steps Preparation
Before moving to Phase 2, ensure you can:
- Set up a LangChain project from scratch
- Create and use prompt templates
- Build basic chains using LCEL
- Implement simple applications with proper structure

### Recommended Practice
Spend extra time on:
- Experimenting with different prompt templates
- Understanding the difference between LLMs and Chat Models
- Building variations of the three projects
- Reading LangChain documentation for components used

---

## Additional Resources for Phase 1

### Documentation Links
- [LangChain Quickstart Guide](https://python.langchain.com/docs/get_started/quickstart)
- [Prompt Templates Documentation](https://python.langchain.com/docs/modules/model_io/prompts/)
- [Output Parsers Guide](https://python.langchain.com/docs/modules/model_io/output_parsers/)

### Practice Exercises
1. Build a personal assistant that can answer questions about your schedule
2. Create a content generator for social media posts
3. Implement a code explanation tool for different programming languages
4. Build a creative writing assistant with multiple genres

### Troubleshooting Common Issues
- **API Key Errors**: Double-check environment variables and key validity
- **Rate Limiting**: Implement delays between requests
- **Cost Management**: Monitor usage through provider dashboards
- **Import Errors**: Ensure all required packages are installed
