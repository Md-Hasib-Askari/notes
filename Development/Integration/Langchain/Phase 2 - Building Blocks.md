# Phase 2: Building Blocks - Detailed Notes
*Weeks 3-4*

## 2.1 Advanced Prompting

### Few-Shot Prompting Techniques

Few-shot prompting provides examples to guide the model's behavior and output format.

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {
        "input": "happy",
        "output": "I am feeling quite joyful and content today."
    },
    {
        "input": "frustrated",
        "output": "I am experiencing some annoyance and impatience right now."
    },
    {
        "input": "curious",
        "output": "I am filled with wonder and eager to learn more about this topic."
    }
]

# Create the example template
example_template = """
Emotion: {input}
Expression: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

# Create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Convert simple emotions into more expressive descriptions:",
    suffix="Emotion: {input}\nExpression:",
    input_variables=["input"]
)

# Usage
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

model = ChatOpenAI(temperature=0.7)
chain = few_shot_prompt | model | StrOutputParser()

result = chain.invoke({"input": "excited"})
print(result)
```

### Dynamic Few-Shot Selection

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()

# More examples for selection
examples = [
    {"input": "Task: Write a poem", "output": "I'll create a beautiful poem with rhythm and rhyme."},
    {"input": "Task: Solve math problem", "output": "Let me work through this step-by-step mathematically."},
    {"input": "Task: Explain concept", "output": "I'll break this down into clear, understandable parts."},
    {"input": "Task: Create story", "output": "I'll craft an engaging narrative with characters and plot."},
    {"input": "Task: Analyze data", "output": "I'll examine the data systematically and draw insights."}
]

# Create example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=2  # Select 2 most similar examples
)

# Create dynamic few-shot prompt
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="User: {input}\nAssistant: {output}"
    ),
    prefix="Here are some examples of how to respond to different tasks:",
    suffix="User: {input}\nAssistant:",
    input_variables=["input"]
)
```

### Chain-of-Thought Prompting

```python
from langchain.prompts import ChatPromptTemplate

# Chain-of-thought template
cot_template = ChatPromptTemplate.from_messages([
    ("system", """You are a logical reasoning assistant. When solving problems:
1. Break down the problem into steps
2. Show your reasoning for each step
3. Provide the final answer clearly

Use this format:
Problem: [restate the problem]
Step 1: [first step with reasoning]
Step 2: [second step with reasoning]
...
Final Answer: [clear answer]"""),
    ("human", "{problem}")
])

class ChainOfThoughtSolver:
    def __init__(self):
        self.model = ChatOpenAI(temperature=0.1)  # Low temperature for logical consistency
        self.chain = cot_template | self.model | StrOutputParser()
    
    def solve(self, problem: str) -> str:
        return self.chain.invoke({"problem": problem})

# Usage
solver = ChainOfThoughtSolver()
result = solver.solve("""
A store is having a sale. If a shirt originally costs $40 and is discounted by 25%, 
then an additional 10% discount is applied to the sale price, what is the final price?
""")
print(result)
```

### Custom Prompt Engineering Patterns

```python
class AdvancedPromptEngineer:
    def __init__(self):
        self.model = ChatOpenAI(temperature=0.7)
        
        # Role-based prompting
        self.role_template = ChatPromptTemplate.from_messages([
            ("system", "You are a {role} with {experience} years of experience. {context}"),
            ("human", "{task}")
        ])
        
        # Constraint-based prompting
        self.constraint_template = ChatPromptTemplate.from_messages([
            ("system", """Follow these constraints strictly:
- {constraint_1}
- {constraint_2}
- {constraint_3}
Format: {format_instruction}"""),
            ("human", "{content}")
        ])
        
        # Perspective prompting
        self.perspective_template = ChatPromptTemplate.from_messages([
            ("system", "Analyze the following from {num_perspectives} different perspectives: {perspectives}"),
            ("human", "{topic}")
        ])
    
    def role_based_response(self, role: str, experience: int, context: str, task: str):
        chain = self.role_template | self.model | StrOutputParser()
        return chain.invoke({
            "role": role,
            "experience": experience,
            "context": context,
            "task": task
        })
    
    def constrained_response(self, constraints: list, format_instruction: str, content: str):
        chain = self.constraint_template | self.model | StrOutputParser()
        return chain.invoke({
            "constraint_1": constraints[0],
            "constraint_2": constraints[1],
            "constraint_3": constraints[2],
            "format_instruction": format_instruction,
            "content": content
        })
    
    def multi_perspective_analysis(self, perspectives: list, topic: str):
        chain = self.perspective_template | self.model | StrOutputParser()
        return chain.invoke({
            "num_perspectives": len(perspectives),
            "perspectives": ", ".join(perspectives),
            "topic": topic
        })

# Usage examples
engineer = AdvancedPromptEngineer()

# Role-based example
role_response = engineer.role_based_response(
    role="senior software architect",
    experience=15,
    context="You specialize in microservices and cloud architecture",
    task="Design a scalable e-commerce system for 1 million users"
)
```

---

## 2.2 Memory Systems

### ConversationBufferMemory

Stores the entire conversation history as is.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

class BasicChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True  # Shows what's happening
        )
    
    def chat(self, message: str) -> str:
        return self.conversation.predict(input=message)
    
    def get_memory(self) -> str:
        return self.memory.buffer
    
    def clear_memory(self):
        self.memory.clear()

# Usage
chatbot = BasicChatbot()
print(chatbot.chat("Hi, my name is Alice"))
print(chatbot.chat("What's my name?"))
print(chatbot.chat("What did we talk about?"))

# Check memory content
print("\nMemory content:")
print(chatbot.get_memory())
```

### ConversationSummaryMemory

Summarizes conversation history to manage token limits.

```python
from langchain.memory import ConversationSummaryMemory

class SummarizingChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            return_messages=True
        )
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def chat(self, message: str) -> str:
        return self.conversation.predict(input=message)
    
    def get_summary(self) -> str:
        return self.memory.buffer

# Usage
summarizing_bot = SummarizingChatbot()
# Have a long conversation...
for i in range(5):
    response = summarizing_bot.chat(f"Tell me about topic number {i+1}")
    print(f"Response {i+1}: {response[:100]}...")

print(f"\nConversation summary: {summarizing_bot.get_summary()}")
```

### ConversationBufferWindowMemory

Keeps only the last k exchanges in memory.

```python
from langchain.memory import ConversationBufferWindowMemory

class WindowedChatbot:
    def __init__(self, window_size: int = 3):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationBufferWindowMemory(
            k=window_size,  # Keep last k exchanges
            return_messages=True
        )
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory
        )
    
    def chat(self, message: str) -> str:
        return self.conversation.predict(input=message)
    
    def get_memory_messages(self):
        return self.memory.chat_memory.messages

# Usage
windowed_bot = WindowedChatbot(window_size=2)
windowed_bot.chat("My favorite color is blue")
windowed_bot.chat("I like pizza")
windowed_bot.chat("I work as a developer")
windowed_bot.chat("What do you know about me?")  # Should only remember recent context
```

### Custom Memory Implementation

```python
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

class SelectiveMemory(BaseChatMemory):
    """Custom memory that only stores important information"""
    
    def __init__(self, important_keywords: list = None):
        super().__init__()
        self.important_keywords = important_keywords or ["name", "age", "job", "location", "preference"]
        self.important_info = {}
    
    def save_context(self, inputs, outputs):
        """Save context only if it contains important information"""
        human_message = inputs.get("input", "")
        ai_message = outputs.get("response", "")
        
        # Check if message contains important keywords
        if any(keyword in human_message.lower() for keyword in self.important_keywords):
            self.chat_memory.add_user_message(human_message)
            self.chat_memory.add_ai_message(ai_message)
            
            # Extract and store important information
            self._extract_important_info(human_message)
    
    def _extract_important_info(self, message: str):
        """Extract important information from message"""
        message_lower = message.lower()
        
        # Simple keyword extraction (in practice, you'd use NLP)
        if "name is" in message_lower:
            name = message_lower.split("name is")[1].strip().split()[0]
            self.important_info["name"] = name
        
        if "work as" in message_lower or "job is" in message_lower:
            # Extract job information
            if "work as" in message_lower:
                job = message_lower.split("work as")[1].strip().split()[0]
            else:
                job = message_lower.split("job is")[1].strip().split()[0]
            self.important_info["job"] = job
    
    def get_important_info(self):
        return self.important_info

# Usage
selective_memory = SelectiveMemory()
```

---

## 2.3 Chains Deep Dive

### LLMChain with Complex Logic

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ContentGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.8)
        
        # Different chains for different content types
        self.blog_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Write a blog post about {topic}.
                Target audience: {audience}
                Tone: {tone}
                Length: {length} words
                Include: {requirements}
                
                Blog Post:""",
                input_variables=["topic", "audience", "tone", "length", "requirements"]
            )
        )
        
        self.social_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Create a {platform} post about {topic}.
                Style: {style}
                Include hashtags: {include_hashtags}
                Character limit: {char_limit}
                
                Post:""",
                input_variables=["platform", "topic", "style", "include_hashtags", "char_limit"]
            )
        )
    
    def generate_blog_post(self, **kwargs):
        return self.blog_chain.run(**kwargs)
    
    def generate_social_post(self, **kwargs):
        return self.social_chain.run(**kwargs)

# Usage
generator = ContentGenerator()
blog_post = generator.generate_blog_post(
    topic="artificial intelligence in healthcare",
    audience="healthcare professionals",
    tone="professional but accessible",
    length="800",
    requirements="statistics, real-world examples, future implications"
)
```

### Sequential Chains

```python
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate

class ContentPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        
        # Chain 1: Generate outline
        self.outline_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Create a detailed outline for a {content_type} about {topic}. Target audience: {audience}",
                input_variables=["content_type", "topic", "audience"]
            ),
            output_key="outline"
        )
        
        # Chain 2: Write content from outline
        self.content_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Based on this outline, write a complete {content_type}:
                
                Outline:
                {outline}
                
                Requirements:
                - Engaging introduction
                - Well-structured body
                - Strong conclusion
                - Appropriate tone for {audience}
                
                {content_type}:""",
                input_variables=["outline", "content_type", "audience"]
            ),
            output_key="content"
        )
        
        # Chain 3: Generate title and meta description
        self.meta_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Based on this content, generate:
                1. An engaging title (under 60 characters)
                2. A meta description (under 160 characters)
                3. 5 relevant tags
                
                Content:
                {content}
                
                Output format:
                Title: [title]
                Meta Description: [description]
                Tags: [tag1, tag2, tag3, tag4, tag5]""",
                input_variables=["content"]
            ),
            output_key="metadata"
        )
        
        # Combine into sequential chain
        self.pipeline = SequentialChain(
            chains=[self.outline_chain, self.content_chain, self.meta_chain],
            input_variables=["content_type", "topic", "audience"],
            output_variables=["outline", "content", "metadata"],
            verbose=True
        )
    
    def generate_complete_content(self, content_type: str, topic: str, audience: str):
        return self.pipeline({"content_type": content_type, "topic": topic, "audience": audience})

# Usage
pipeline = ContentPipeline()
result = pipeline.generate_complete_content(
    content_type="article",
    topic="sustainable living tips",
    audience="environmentally conscious consumers"
)

print("Outline:", result["outline"])
print("\nContent:", result["content"][:200] + "...")
print("\nMetadata:", result["metadata"])
```

### Router Chains

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

class IntelligentAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        
        # Define different prompt templates for different tasks
        self.prompt_infos = [
            {
                "name": "math",
                "description": "Good for solving mathematical problems, calculations, and equations",
                "prompt_template": """You are a mathematics expert. Solve this problem step by step:
                
                Problem: {input}
                
                Solution:"""
            },
            {
                "name": "creative_writing",
                "description": "Good for creative writing, storytelling, and literary tasks",
                "prompt_template": """You are a creative writer. Help with this creative writing task:
                
                Task: {input}
                
                Creative Response:"""
            },
            {
                "name": "coding",
                "description": "Good for programming, coding problems, and technical solutions",
                "prompt_template": """You are a senior software developer. Help with this coding task:
                
                Task: {input}
                
                Code Solution:"""
            },
            {
                "name": "general",
                "description": "Good for general questions and conversations",
                "prompt_template": """You are a helpful assistant. Answer this question:
                
                Question: {input}
                
                Answer:"""
            }
        ]
        
        # Create destination chains
        destination_chains = {}
        for p_info in self.prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain
        
        # Create default chain
        default_chain = ConversationChain(llm=self.llm, output_key="text")
        
        # Create router
        destinations = [f"{p['name']}: {p['description']}" for p in self.prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)
        
        # Create multi-prompt chain
        self.chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=True
        )
    
    def ask(self, question: str):
        return self.chain.run(question)

# Usage
assistant = IntelligentAssistant()

# Test different types of questions
questions = [
    "What is the integral of x^2 from 0 to 5?",
    "Write a short story about a robot learning to love",
    "How do I implement a binary search algorithm in Python?",
    "What's the weather like today?"
]

for question in questions:
    print(f"Question: {question}")
    response = assistant.ask(question)
    print(f"Response: {response}\n")
```

### Transform Chains

```python
from langchain.chains import TransformChain, SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
import re

def clean_text(inputs: dict) -> dict:
    """Transform function to clean text"""
    text = inputs["text"]
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Capitalize sentences
    sentences = text.split('.')
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    text = '. '.join(sentences)
    
    return {"cleaned_text": text}

def extract_keywords(inputs: dict) -> dict:
    """Transform function to extract keywords"""
    text = inputs["text"].lower()
    
    # Simple keyword extraction (remove stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    words = text.split()
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    
    return {"keywords": keywords[:10]}  # Top 10 keywords

class TextProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        
        # Create transform chains
        self.clean_chain = TransformChain(
            input_variables=["text"],
            output_variables=["cleaned_text"],
            transform=clean_text
        )
        
        self.keyword_chain = TransformChain(
            input_variables=["text"],
            output_variables=["keywords"],
            transform=extract_keywords
        )
        
        # Create LLM chain for analysis
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Analyze this cleaned text and provide insights:
                
                Text: {cleaned_text}
                Keywords: {keywords}
                
                Please provide:
                1. Summary (2-3 sentences)
                2. Main themes
                3. Sentiment analysis
                4. Readability assessment
                
                Analysis:""",
                input_variables=["cleaned_text", "keywords"]
            ),
            output_key="analysis"
        )
        
        # Combine into sequential chain
        self.pipeline = SequentialChain(
            chains=[self.clean_chain, self.keyword_chain, self.analysis_chain],
            input_variables=["text"],
            output_variables=["cleaned_text", "keywords", "analysis"],
            verbose=True
        )
    
    def process_text(self, text: str):
        return self.pipeline({"text": text})

# Usage
processor = TextProcessor()
raw_text = """
This is some  messy    text with extra spaces!!!
it has inconsistent capitalization and weird formatting.
But it contains important information about natural language processing and machine learning.
These technologies are revolutionizing how we interact with computers and process information.
"""

result = processor.process_text(raw_text)
print("Cleaned text:", result["cleaned_text"])
print("Keywords:", result["keywords"])
print("Analysis:", result["analysis"])
```

---

## 2.4 Projects

### Project 1: Conversational Chatbot with Memory

```python
import json
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class AdvancedChatbot:
    def __init__(self, bot_name="Assistant", personality="helpful and friendly"):
        self.bot_name = bot_name
        self.personality = personality
        self.llm = ChatOpenAI(temperature=0.7)
        
        # Use summary buffer memory for efficient memory management
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        # Custom prompt template
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=f"""You are {self.bot_name}, a {self.personality} AI assistant.
            
            Current conversation:
            {{history}}
            
            Human: {{input}}
            {self.bot_name}:"""
        )
        
        # Create conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )
        
        # Track conversation metadata
        self.conversation_start = datetime.now()
        self.message_count = 0
    
    def chat(self, message: str) -> str:
        """Send a message and get response"""
        self.message_count += 1
        response = self.conversation.predict(input=message)
        return response
    
    def get_conversation_summary(self) -> str:
        """Get summary of the conversation"""
        if hasattr(self.memory, 'moving_summary_buffer'):
            return self.memory.moving_summary_buffer
        return "No summary available yet."
    
    def get_stats(self) -> dict:
        """Get conversation statistics"""
        duration = datetime.now() - self.conversation_start
        return {
            "message_count": self.message_count,
            "duration_minutes": duration.total_seconds() / 60,
            "summary": self.get_conversation_summary()
        }
    
    def save_conversation(self, filename: str):
        """Save conversation to file"""
        data = {
            "bot_name": self.bot_name,
            "personality": self.personality,
            "start_time": self.conversation_start.isoformat(),
            "messages": [
                {
                    "type": "human" if i % 2 == 0 else "ai",
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                }
                for i, msg in enumerate(self.memory.chat_memory.messages)
            ],
            "summary": self.get_conversation_summary(),
            "stats": self.get_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.message_count = 0
        self.conversation_start = datetime.now()

# Usage example
def main():
    # Create chatbot with specific personality
    bot = AdvancedChatbot(
        bot_name="CodeMentor", 
        personality="experienced programming mentor who explains concepts clearly"
    )
    
    print(f"Chat with {bot.bot_name}! Type 'quit' to exit, 'stats' for statistics, 'save' to save conversation.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stats':
            stats = bot.get_stats()
            print(f"Statistics: {stats}")
            continue
        elif user_input.lower() == 'save':
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            bot.save_conversation(filename)
            print(f"Conversation saved to {filename}")
            continue
        
        response = bot.chat(user_input)
        print(f"{bot.bot_name}: {response}")

if __name__ == "__main__":
    main()
```

### Project 2: Multi-Step Reasoning System

```python
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

class ReasoningSystem:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1)  # Low temperature for consistency
        
        # Step 1: Problem decomposition
        self.decompose_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Break down this complex problem into smaller, manageable sub-problems:
                
                Problem: {problem}
                
                Sub-problems (list each with a number):""",
                input_variables=["problem"]
            ),
            output_key="sub_problems"
        )
        
        # Step 2: Gather information
        self.info_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""For each sub-problem, identify what information or data is needed:
                
                Sub-problems: {sub_problems}
                
                Required information for each sub-problem:""",
                input_variables=["sub_problems"]
            ),
            output_key="required_info"
        )
        
        # Step 3: Solve each sub-problem
        self.solve_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Solve each sub-problem step by step:
                
                Sub-problems: {sub_problems}
                Available information: {required_info}
                
                Solutions for each sub-problem:""",
                input_variables=["sub_problems", "required_info"]
            ),
            output_key="solutions"
        )
        
        # Step 4: Synthesize final answer
        self.synthesize_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Combine the solutions to provide a comprehensive answer to the original problem:
                
                Original problem: {problem}
                Sub-problem solutions: {solutions}
                
                Final comprehensive answer:""",
                input_variables=["problem", "solutions"]
            ),
            output_key="final_answer"
        )
        
        # Create sequential chain
        self.reasoning_pipeline = SequentialChain(
            chains=[self.decompose_chain, self.info_chain, self.solve_chain, self.synthesize_chain],
            input_variables=["problem"],
            output_variables=["sub_problems", "required_info", "solutions", "final_answer"],
            verbose=True
        )
    
    def reason_through_problem(self, problem: str) -> dict:
        """Process a complex problem through multi-step reasoning"""
        return self.reasoning_pipeline({"problem": problem})
    
    def explain_reasoning(self, result: dict) -> str:
        """Provide a clear explanation of the reasoning process"""
        explanation = f"""
        REASONING PROCESS EXPLANATION:
        
        1. PROBLEM DECOMPOSITION:
        {result['sub_problems']}
        
        2. INFORMATION REQUIREMENTS:
        {result['required_info']}
        
        3. SUB-PROBLEM SOLUTIONS:
        {result['solutions']}
        
        4. FINAL SYNTHESIS:
        {result['final_answer']}
        """
        return explanation

# Usage example
def test_reasoning_system():
    reasoner = ReasoningSystem()
    
    complex_problem = """
    A company wants to reduce its carbon footprint by 50% over the next 3 years while 
    maintaining profitability and employee satisfaction. They currently have 500 employees 
    across 3 office locations, use significant energy for data centers, and rely heavily 
    on business travel. What comprehensive strategy should they implement?
    """
    
    print("Processing complex problem...")
    result = reasoner.reason_through_problem(complex_problem)
    
    print("\n" + "="*80)
    print("REASONING SYSTEM OUTPUT")
    print("="*80)
    
    explanation = reasoner.explain_reasoning(result)
    print(explanation)

if __name__ == "__main__":
    test_reasoning_system()
```

### Project 3: Content Generation Pipeline

```python
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import json
from datetime import datetime

class ContentGenerationPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.8)  # Higher creativity for content
        self.memory = ConversationBufferMemory()
        
        # Research chain
        self.research_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Research the topic '{topic}' for {content_type} creation.
                Target audience: {audience}
                Content purpose: {purpose}
                
                Provide:
                1. Key facts and statistics
                2. Current trends
                3. Common questions/pain points
                4. Unique angles to explore
                5. Relevant examples or case studies
                
                Research findings:""",
                input_variables=["topic", "content_type", "audience", "purpose"]
            ),
            output_key="research"
        )
        
        # Outline chain  
        self.outline_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Create a detailed outline for a {content_type} about '{topic}'.
                
                Research context: {research}
                Target audience: {audience}
                Content purpose: {purpose}
                Desired length: {length}
                Tone: {tone}
                
                Create a comprehensive outline with:
                - Engaging title options (3-5 titles)
                - Introduction hook
                - Main sections with subsections
                - Key points for each section
                - Conclusion approach
                - Call-to-action ideas
                
                Detailed outline:""",
                input_variables=["topic", "content_type", "research", "audience", "purpose", "length", "tone"]
            ),
            output_key="outline"
        )
        
        # Content creation chain
        self.content_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Write a complete {content_type} based on this outline and research.
                
                Topic: {topic}
                Outline: {outline}
                Research: {research}
                Target audience: {audience}
                Tone: {tone}
                Length: {length}
                
                Requirements:
                - Follow the outline structure
                - Incorporate research findings naturally
                - Match the specified tone and audience
                - Include engaging transitions
                - Add relevant examples/anecdotes
                - End with strong conclusion and call-to-action
                
                Complete {content_type}:""",
                input_variables=["topic", "content_type", "outline", "research", "audience", "tone", "length"]
            ),
            output_key="content"
        )
        
        # SEO optimization chain
        self.seo_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Optimize this content for SEO and provide metadata:
                
                Content: {content}
                Primary topic: {topic}
                
                Generate:
                1. SEO-optimized title (under 60 characters)
                2. Meta description (under 160 characters)
                3. 10 relevant keywords/phrases
                4. 5 suggested internal link anchor texts
                5. Social media post variations (Twitter, LinkedIn, Facebook)
                6. Email subject line options
                
                SEO package:""",
                input_variables=["content", "topic"]
            ),
            output_key="seo_package"
        )
        
        # Quality review chain
        self.review_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Review this content and provide improvement suggestions:
                
                Content: {content}
                Original requirements:
                - Topic: {topic}
                - Audience: {audience}
                - Purpose: {purpose}
                - Tone: {tone}
                
                Evaluate:
                1. Clarity and readability (1-10 score)
                2. Audience appropriateness (1-10 score)  
                3. Engagement level (1-10 score)
                4. Goal achievement (1-10 score)
                5. Areas for improvement
                6. Strengths to maintain
                7. Specific revision suggestions
                
                Quality review:""",
                input_variables=["content", "topic", "audience", "purpose", "tone"]
            ),
            output_key="quality_review"
        )
        
        # Create the full pipeline
        self.pipeline = SequentialChain(
            chains=[
                self.research_chain,
                self.outline_chain, 
                self.content_chain,
                self.seo_chain,
                self.review_chain
            ],
            input_variables=["topic", "content_type", "audience", "purpose", "length", "tone"],
            output_variables=["research", "outline", "content", "seo_package", "quality_review"],
            verbose=True
        )
    
    def generate_content(self, **kwargs):
        """Generate complete content package"""
        return self.pipeline(kwargs)
    
    def save_content_package(self, result: dict, filename: str = None):
        """Save the complete content package"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"content_package_{timestamp}.json"
        
        # Add metadata
        package = {
            "generated_at": datetime.now().isoformat(),
            "content_package": result
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(package, f, indent=2, ensure_ascii=False)
        
        return filename

class ContentWorkflow:
    """Higher-level workflow manager for content creation"""
    
    def __init__(self):
        self.pipeline = ContentGenerationPipeline()
        self.content_templates = {
            "blog_post": {
                "length": "800-1200 words",
                "tone": "informative yet engaging",
                "purpose": "educate and inform readers"
            },
            "social_media": {
                "length": "50-100 words", 
                "tone": "casual and engaging",
                "purpose": "drive engagement and shares"
            },
            "email_newsletter": {
                "length": "300-500 words",
                "tone": "friendly and personal", 
                "purpose": "inform subscribers and drive action"
            },
            "white_paper": {
                "length": "2000-3000 words",
                "tone": "professional and authoritative",
                "purpose": "establish thought leadership"
            }
        }
    
    def create_content(self, topic: str, content_type: str, audience: str, custom_params: dict = None):
        """Create content with predefined templates"""
        
        if content_type not in self.content_templates:
            raise ValueError(f"Content type '{content_type}' not supported. Available: {list(self.content_templates.keys())}")
        
        # Get template defaults
        template = self.content_templates[content_type].copy()
        
        # Override with custom parameters
        if custom_params:
            template.update(custom_params)
        
        # Generate content
        result = self.pipeline.generate_content(
            topic=topic,
            content_type=content_type,
            audience=audience,
            **template
        )
        
        return result
    
    def batch_create_content(self, content_requests: list):
        """Create multiple pieces of content"""
        results = []
        
        for request in content_requests:
            print(f"Generating {request['content_type']} about '{request['topic']}'...")
            result = self.create_content(**request)
            results.append({
                "request": request,
                "result": result
            })
        
        return results

# Usage example
def demo_content_pipeline():
    workflow = ContentWorkflow()
    
    # Single content creation
    print("Creating blog post...")
    result = workflow.create_content(
        topic="The Future of Remote Work",
        content_type="blog_post", 
        audience="business professionals and managers",
        custom_params={
            "tone": "professional but conversational",
            "purpose": "help leaders understand remote work trends"
        }
    )
    
    # Save the result
    filename = workflow.pipeline.save_content_package(result)
    print(f"Content package saved to: {filename}")
    
    # Display key outputs
    print("\n" + "="*80)
    print("CONTENT GENERATION RESULTS")
    print("="*80)
    
    print("\nRESEARCH FINDINGS:")
    print(result["research"][:300] + "...")
    
    print("\nCONTENT OUTLINE:")
    print(result["outline"][:300] + "...")
    
    print("\nGENERATED CONTENT:")
    print(result["content"][:500] + "...")
    
    print("\nSEO PACKAGE:")
    print(result["seo_package"][:300] + "...")
    
    print("\nQUALITY REVIEW:")
    print(result["quality_review"][:300] + "...")

if __name__ == "__main__":
    demo_content_pipeline()
```

---

## Key Takeaways from Phase 2

### Advanced Skills Mastered
1. **Sophisticated Prompting**: Few-shot learning, chain-of-thought reasoning, and role-based prompts
2. **Memory Management**: Different memory types and their appropriate use cases
3. **Complex Chains**: Sequential processing, routing logic, and data transformation
4. **Production Patterns**: Robust error handling, logging, and structured workflows

### Best Practices Learned
- **Memory Selection**: Choose memory type based on use case (buffer for short conversations, summary for long ones)
- **Chain Design**: Keep chains focused on single responsibilities
- **Prompt Engineering**: Use examples and constraints to guide model behavior
- **Error Handling**: Always handle API failures and edge cases

### Common Patterns to Remember
- **Sequential Processing**: Research → Outline → Content → Review
- **Conditional Logic**: Route different types of requests to specialized chains
- **Data Transformation**: Clean and structure data between chain steps
- **Memory Integration**: Maintain context across multi-turn interactions

### Performance Considerations
- Monitor token usage across chains
- Use appropriate temperature settings for different tasks
- Implement caching for repeated operations
- Consider async processing for independent operations
