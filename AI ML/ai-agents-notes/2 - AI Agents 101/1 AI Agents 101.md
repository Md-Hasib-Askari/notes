## 🤖 **2. AI Agents 101**

---

### 🔹 **2.1 What are AI Agents?**

An **AI Agent** is an autonomous system built on top of a language model that can **perceive**, **reason**, **act**, and **learn** in a goal-directed manner.

#### Key Traits:

* **Autonomous**: Can take actions without human intervention.
* **Goal-Oriented**: Operates based on a task or objective.
* **Interactive**: Communicates with users and tools.
* **Adaptive**: Can use memory or reflection to improve over time.

> Think of an AI agent as an LLM "wrapped in a brain" that can interact with the world—tools, memory, APIs, and users.

---

### 🔹 **2.2 What are Tools?**

**Tools** (also called **actions** or **functions**) are external systems or utilities that the agent can invoke to achieve tasks beyond the LLM’s native capabilities.

#### Examples:

* Web search APIs
* Database queries
* Python code execution
* File system access
* Sending emails or Slack messages

> Tools let agents interact with the real world — compute results, retrieve data, or execute actions.

---

### 🔁 **2.3 The Agent Loop**

At the heart of every AI agent is the **Agent Loop**, a repeated cycle of interaction and reasoning.

#### 🧠 The 4 Stages:

1. **Perception / User Input**

   * Receives prompt, command, or environmental input
   * May include prior memory or history

2. **Reason and Plan**

   * The LLM interprets the task, breaks it into steps, and chooses the best action
   * May generate intermediate thoughts (e.g., “I should search this online first”)

3. **Acting / Tool Invocation**

   * Executes one or more tools
   * Passes the correct inputs and captures outputs

4. **Observation & Reflection**

   * Analyzes tool outputs and determines next step
   * May update memory or refine the plan

#### Optional:

* This loop continues until the task is complete or a stopping condition is met.

---

### 📦 **2.4 Example Use Cases**

#### 🔹 **Personal Assistant**

* Calendar management
* Email replies
* Scheduling meetings
* Task reminders

#### 🔹 **Code Generation**

* Takes user prompts to generate, edit, or debug code
* May include multi-step reasoning (ReAct) or testing

#### 🔹 **Data Analysis**

* Loads and analyzes CSVs, databases, or APIs
* Uses Python tools to generate plots, summaries, insights

#### 🔹 **Web Scraping / Crawling**

* Navigates websites
* Extracts structured information
* May build a knowledge base from content

#### 🔹 **NPC / Game AI**

* AI-powered characters that can interact dynamically with players
* Can use memory, context, and reasoning to simulate personalities
