# Installation & Basic Usage

## Overview
This guide will walk you through installing AutoGPT locally, configuring it properly, and running your first autonomous agent instances. By the end of this section, you'll have a working AutoGPT setup and experience with basic operations.

## Learning Objectives
By the end of this section, you should be able to:
- Install AutoGPT from official sources
- Configure environment variables and API keys securely
- Launch and interact with AutoGPT instances
- Navigate the web interface and understand command structure
- Execute basic tasks like web research and file operations

## Prerequisites
Before starting, ensure you have completed:
- [ ] Foundation Phase prerequisites
- [ ] Python 3.8+ installed and configured
- [ ] OpenAI API account with valid API key
- [ ] Basic command line familiarity
- [ ] Git installed (for cloning repository)

## 1. Install AutoGPT Locally

### Method 1: Git Clone (Recommended)
```bash
# Clone the official AutoGPT repository
git clone https://github.com/Significant-Gravitas/AutoGPT.git

# Navigate to the AutoGPT directory
cd AutoGPT

# Check available versions/branches
git branch -a
git tag --list | tail -10

# Switch to latest stable release (replace with actual version)
git checkout tags/v0.5.0
```

### Method 2: Download Release
1. Visit [AutoGPT Releases](https://github.com/Significant-Gravitas/AutoGPT/releases)
2. Download the latest stable release zip file
3. Extract to your desired directory
4. Navigate to the extracted folder

### Method 3: Docker Installation (Alternative)
```bash
# Pull the official Docker image
docker pull significantgravitas/autogpt

# Run with basic configuration
docker run -it --env-file=.env significantgravitas/autogpt
```

### Verify Installation
```bash
# Check directory structure
ls -la

# Verify key files exist
ls autogpt/
ls scripts/
ls requirements.txt
```

## 2. Configure Environment Variables and API Keys

### Create Environment File
```bash
# Copy the template environment file
cp .env.template .env

# Open for editing (use your preferred editor)
nano .env
# or
code .env
```

### Essential Environment Variables
Add the following to your `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_MODEL=gpt-4
TEMPERATURE=0

# AutoGPT Configuration
MEMORY_BACKEND=local
WORKSPACE_PATH=./workspace
RESTRICT_TO_WORKSPACE=True

# Browser Configuration
USE_WEB_BROWSER=selenium
HEADLESS_BROWSER=True

# Agent Configuration
AI_SETTINGS_FILE=ai_settings.yaml
```

### Security Best Practices
- **Never commit `.env` files to version control**
- **Use strong, unique API keys**
- **Regularly rotate API keys**
- **Set appropriate usage limits in OpenAI dashboard**
- **Monitor API usage and costs**

### API Key Management
```bash
# Verify API key is working
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Set environment variable for current session (Linux/Mac)
export OPENAI_API_KEY="your_key_here"

# Set environment variable for current session (Windows)
set OPENAI_API_KEY=your_key_here
```

## 3. Install Dependencies

### Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv autogpt_env

# Activate virtual environment
# Windows:
autogpt_env\Scripts\activate
# Linux/Mac:
source autogpt_env/bin/activate

# Install required packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(openai|requests|selenium)"
```

### Additional Dependencies
```bash
# For web browsing capabilities
pip install selenium webdriver-manager

# For enhanced memory backends
pip install redis  # if using Redis
pip install pinecone-client  # if using Pinecone

# For development and debugging
pip install pytest black flake8
```

## 4. Run Your First AutoGPT Instance

### Basic Launch
```bash
# Run AutoGPT with default settings
python -m autogpt

# Run with specific configuration
python -m autogpt --ai-settings ai_settings.yaml
```

### Launch with Docker
```bash
# Run with environment file
docker run -it --env-file=.env \
  -v $(pwd)/workspace:/app/workspace \
  significantgravitas/autogpt
```

### Configuration Options
```bash
# Available command line options
python -m autogpt --help

# Common options:
python -m autogpt \
  --continuous \
  --gpt3only \
  --gpt4only \
  --memory-type local \
  --workspace-path ./workspace
```

### First Run Setup
1. **Agent Name**: Choose a descriptive name for your agent
2. **Agent Role**: Define what your agent should do
3. **Goals**: Set 1-5 specific, measurable goals
4. **Constraints**: Review and adjust operational limits

### Example First Agent Setup
```yaml
ai_name: "ResearchAssistant"
ai_role: "An AI assistant specialized in conducting web research and summarizing findings"
ai_goals:
  - "Research the latest developments in artificial intelligence"
  - "Create a comprehensive summary of findings"
  - "Save the research report to a file"
ai_constraints:
  - "Only use reliable, well-known sources"
  - "Limit research to the last 6 months"
  - "Keep summaries concise and factual"
```

## 5. Understanding the Web Interface and Command Structure

### Web Interface Components
- **Dashboard**: Overview of agent status and activity
- **Chat Interface**: Direct communication with the agent
- **File Browser**: Access to workspace files
- **Logs**: Detailed execution logs and debugging info
- **Settings**: Configuration and preferences

### Command Structure
AutoGPT uses a combination of:
- **Natural language goals**: High-level objectives
- **System commands**: Direct operational instructions
- **Plugin commands**: Specific tool invocations
- **Memory commands**: Information storage and retrieval

### Basic Commands
```bash
# Agent control commands
y         # Authorize next action
n         # Deny next action
q         # Quit AutoGPT
c         # Continue without asking
s         # Skip current action

# Memory commands
/memory   # Show memory contents
/clear    # Clear memory
/save     # Save current state

# System commands
/help     # Show available commands
/status   # Show agent status
/restart  # Restart agent
```

### Interaction Modes
- **Manual Mode**: Approve each action individually
- **Continuous Mode**: Run autonomously with minimal interruption
- **Step Mode**: Execute one action at a time
- **Review Mode**: Review planned actions before execution

## 6. Practice with Basic Tasks

### Task 1: Web Research
**Objective**: Research a specific topic and create a summary

**Setup**:
```yaml
ai_name: "WebResearcher"
ai_role: "Research assistant for gathering and summarizing web information"
ai_goals:
  - "Research recent developments in renewable energy technology"
  - "Find at least 5 reliable sources"
  - "Create a structured summary with key findings"
  - "Save the summary as a markdown file"
```

**Expected Actions**:
1. Search for relevant information
2. Browse multiple sources
3. Extract key information
4. Organize findings
5. Create and save summary file

### Task 2: File Operations
**Objective**: Organize and manage files in the workspace

**Setup**:
```yaml
ai_name: "FileOrganizer"
ai_role: "Assistant for organizing and managing files and directories"
ai_goals:
  - "Create a structured directory for project files"
  - "Sort existing files by type and date"
  - "Create an index of all files with descriptions"
  - "Generate a folder structure report"
```

**Expected Actions**:
1. Analyze current file structure
2. Create appropriate directories
3. Move files to organized locations
4. Generate documentation
5. Create summary reports

### Task 3: Data Analysis
**Objective**: Analyze a simple dataset and provide insights

**Setup**:
```yaml
ai_name: "DataAnalyst"
ai_role: "Assistant for basic data analysis and visualization"
ai_goals:
  - "Load and examine a CSV dataset"
  - "Identify key patterns and trends"
  - "Create basic statistical summaries"
  - "Generate insights and recommendations"
```

**Expected Actions**:
1. Load and inspect data
2. Perform statistical analysis
3. Identify patterns and anomalies
4. Create visualizations
5. Write analysis report

## 7. Troubleshooting Common Issues

### Installation Issues
```bash
# Python version conflicts
python --version
python3 --version

# Missing dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Permission issues (Linux/Mac)
sudo chown -R $USER:$USER ./AutoGPT
chmod +x scripts/*.sh
```

### Configuration Issues
```bash
# API key not working
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Environment variables not loaded
source .env  # Linux/Mac
# Check if variables are set
echo $OPENAI_API_KEY
```

### Runtime Issues
- **High API costs**: Set usage limits in OpenAI dashboard
- **Slow performance**: Use GPT-3.5 instead of GPT-4 for testing
- **Memory issues**: Clear browser cache and restart
- **Network timeouts**: Check internet connection and firewall

### Common Error Messages
- `Invalid API key`: Check key format and permissions
- `Rate limit exceeded`: Wait or upgrade OpenAI plan
- `Module not found`: Reinstall requirements.txt
- `Permission denied`: Check file/directory permissions

## 8. Best Practices for Beginners

### Goal Setting
- **Be specific**: Vague goals lead to poor results
- **Set constraints**: Limit scope to prevent runaway behavior
- **Start small**: Begin with simple, achievable objectives
- **Iterate**: Refine goals based on initial results

### Monitoring and Control
- **Stay engaged**: Monitor agent actions, especially initially
- **Set timeouts**: Limit execution time for safety
- **Review logs**: Understand what the agent is doing
- **Use manual mode**: Approve actions until you're comfortable

### Cost Management
- **Monitor usage**: Check OpenAI dashboard regularly
- **Set limits**: Configure spending alerts
- **Use efficient models**: GPT-3.5 for testing, GPT-4 for production
- **Optimize prompts**: Reduce token usage where possible

### Security Considerations
- **Limit permissions**: Restrict file system access
- **Review actions**: Never blindly approve dangerous operations
- **Backup data**: Always backup important files
- **Use sandboxes**: Test in isolated environments

## 9. Next Steps

### Immediate Actions
1. **Complete all practice tasks** in this section
2. **Experiment with different goal types** and configurations
3. **Document your experiences** and lessons learned
4. **Join the AutoGPT community** for support and tips

### Prepare for Architecture Understanding
1. **Study the workspace directory** structure after runs
2. **Examine log files** to understand agent decision-making
3. **Review the codebase** structure in the AutoGPT directory
4. **Research agent memory systems** and plugin architecture

### Advanced Exploration
- **Try different AI models** and compare performance
- **Experiment with memory backends** (local vs. external)
- **Test plugin functionality** with simple use cases
- **Create custom agent configurations** for specific tasks

## Assessment Checklist
- [ ] AutoGPT successfully installed and configured
- [ ] Environment variables properly set
- [ ] API keys working and secure
- [ ] First agent run completed successfully
- [ ] Web interface navigation understood
- [ ] Basic command structure mastered
- [ ] Web research task completed
- [ ] File operations task completed
- [ ] Data analysis task attempted
- [ ] Troubleshooting skills developed
- [ ] Ready for architecture deep dive

## Resources for This Phase

### Official Documentation
- [AutoGPT Installation Guide](https://docs.agpt.co/setup/)
- [Configuration Reference](https://docs.agpt.co/configuration/)
- [API Documentation](https://docs.agpt.co/api/)

### Community Resources
- [AutoGPT Discord](https://discord.gg/autogpt)
- [GitHub Discussions](https://github.com/Significant-Gravitas/AutoGPT/discussions)
- [Reddit Community](https://www.reddit.com/r/AutoGPT/)

### Video Tutorials
- AutoGPT installation walkthroughs on YouTube
- Community-created setup guides
- Troubleshooting video series

### Practice Datasets
- [Kaggle Public Datasets](https://www.kaggle.com/datasets)
- [Government Open Data](https://www.data.gov/)
- [Sample CSV files for testing](https://people.sc.fsu.edu/~jburkardt/data/csv/csv.html)
