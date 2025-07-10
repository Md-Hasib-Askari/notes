# Week 1.1: Getting Started with Streamlit

## Installation and Setup

### Installing Streamlit
```bash
# Install Streamlit using pip
pip install streamlit

# Verify installation
streamlit --version

# Install additional useful packages
pip install pandas numpy matplotlib plotly
```

### Alternative Installation Methods
```bash
# Using conda
conda install -c conda-forge streamlit

# Install specific version
pip install streamlit==1.28.0

# Install from source (for developers)
pip install git+https://github.com/streamlit/streamlit.git
```

## Creating Your First App

### Hello World Application
```python
# app.py
import streamlit as st

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="My First App",
    page_icon="🚀",
    layout="wide"
)

# Main content
st.title("Hello, Streamlit! 🎉")
st.write("Welcome to my first Streamlit application!")

# Simple interaction
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! Nice to meet you! 👋")

# Display some data
st.subheader("Sample Data")
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Tokyo']
}
st.table(data)
```

### Basic App Structure
```python
# Standard Streamlit app structure
import streamlit as st
import pandas as pd
import numpy as np

# 1. Page configuration (always first)
st.set_page_config(page_title="App Title", page_icon="📊")

# 2. Title and description
st.title("My Streamlit Application")
st.markdown("Description of what this app does")

# 3. Sidebar (optional)
st.sidebar.header("Settings")

# 4. Main content area
# Your app logic here

# 5. Footer (optional)
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
```

## Running Streamlit Apps

### Basic Command
```bash
# Run the app
streamlit run app.py

# Run with specific port
streamlit run app.py --server.port 8080

# Run with custom host
streamlit run app.py --server.address 0.0.0.0

# Run in development mode (auto-reload)
streamlit run app.py --server.runOnSave true
```

### Advanced Running Options
```bash
# Run with different browser
streamlit run app.py --browser.serverAddress localhost

# Disable browser auto-opening
streamlit run app.py --server.headless true

# Set maximum upload size (in MB)
streamlit run app.py --server.maxUploadSize 200

# Run with specific theme
streamlit run app.py --theme.base dark
```

### Environment Configuration
```python
# .streamlit/config.toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## Understanding Streamlit's Magic

### Script Rerun Behavior
```python
import streamlit as st
import time

st.title("Understanding Streamlit Reruns")

# This will print every time the script runs
print("Script is running!")  # Visible in terminal

# Counter to track reruns
if 'counter' not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1
st.write(f"This script has run {st.session_state.counter} times")

# Trigger rerun with button
if st.button("Click me to rerun the script"):
    st.write("Button clicked! Script will rerun.")

# Show current time to see rerun effect
st.write(f"Current time: {time.strftime('%H:%M:%S')}")
```

### Widget State and Reruns
```python
import streamlit as st

st.title("Widget State Example")

# Each widget interaction triggers a rerun
text_input = st.text_input("Type something:")
number_input = st.number_input("Enter a number:", min_value=0, max_value=100)
slider_value = st.slider("Choose a value:", 0, 100, 50)

# Display current values
st.write("Current values:")
st.write(f"Text: {text_input}")
st.write(f"Number: {number_input}")
st.write(f"Slider: {slider_value}")

# Calculate based on inputs
if text_input and number_input:
    result = len(text_input) * number_input * slider_value
    st.success(f"Calculated result: {result}")
```

### Performance Considerations
```python
import streamlit as st
import pandas as pd
import time

st.title("Performance and Caching Demo")

# Without caching - runs every time
def load_data_slow():
    time.sleep(2)  # Simulate slow operation
    return pd.DataFrame({
        'A': range(1000),
        'B': range(1000, 2000)
    })

# With caching - runs only once
@st.cache_data
def load_data_fast():
    time.sleep(2)  # Simulate slow operation
    return pd.DataFrame({
        'A': range(1000),
        'B': range(1000, 2000)
    })

# Demo the difference
if st.button("Load data without caching (slow)"):
    with st.spinner("Loading..."):
        data = load_data_slow()
    st.success("Data loaded!")
    st.dataframe(data.head())

if st.button("Load data with caching (fast after first time)"):
    with st.spinner("Loading..."):
        data = load_data_fast()
    st.success("Data loaded!")
    st.dataframe(data.head())
```

## Development Workflow

### File Structure
```
my_streamlit_project/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── data/                 # Data files
├── utils/                # Utility functions
│   └── helpers.py
└── README.md            # Project documentation
```

### Best Practices for Development
```python
# app.py - Clean structure example
import streamlit as st
from utils.helpers import load_data, process_data

def main():
    """Main application function"""
    # Page config
    st.set_page_config(
        page_title="My App",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    create_header()
    
    # Sidebar
    settings = create_sidebar()
    
    # Main content
    create_main_content(settings)
    
    # Footer
    create_footer()

def create_header():
    """Create the app header"""
    st.title("📊 My Streamlit Dashboard")
    st.markdown("Welcome to my analytics dashboard")
    st.markdown("---")

def create_sidebar():
    """Create sidebar with settings"""
    st.sidebar.header("⚙️ Settings")
    
    settings = {
        'data_source': st.sidebar.selectbox(
            "Data Source",
            ["Local File", "Database", "API"]
        ),
        'refresh_rate': st.sidebar.slider(
            "Refresh Rate (seconds)",
            1, 60, 10
        )
    }
    
    return settings

def create_main_content(settings):
    """Create main content area"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Overview")
        # Add content here
    
    with col2:
        st.subheader("Visualizations")
        # Add content here

def create_footer():
    """Create app footer"""
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
```

## Debugging and Troubleshooting

### Common Issues and Solutions
```python
import streamlit as st

# Issue 1: Widget key conflicts
st.title("Debugging Common Issues")

# BAD: May cause key conflicts
# st.text_input("Name")  # in multiple places

# GOOD: Use unique keys
name1 = st.text_input("First Name", key="first_name")
name2 = st.text_input("Last Name", key="last_name")

# Issue 2: Handling exceptions
try:
    number = st.number_input("Enter a number:")
    result = 10 / number
    st.write(f"Result: {result}")
except ZeroDivisionError:
    st.error("Cannot divide by zero!")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Issue 3: Debug information
if st.checkbox("Show debug info"):
    st.write("Session state:", st.session_state)
    st.write("URL parameters:", st.experimental_get_query_params())
```

### Development Tips
```python
import streamlit as st
import sys

# Enable debug mode
if st.checkbox("Debug Mode"):
    st.write("Python version:", sys.version)
    st.write("Streamlit version:", st.__version__)
    
    # Show session state
    st.subheader("Session State")
    st.write(st.session_state)
    
    # Show current script run count
    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    st.session_state.run_count += 1
    st.write(f"Script runs: {st.session_state.run_count}")

# Use st.echo() to show code
with st.echo():
    # This code will be displayed
    x = 10
    y = 20
    result = x + y
    st.write(f"The sum is: {result}")
```

## Key Concepts Summary

### Streamlit Execution Model
1. **Top-to-bottom execution**: Script runs from top to bottom on every interaction
2. **Stateless by default**: Variables reset on each run (use session_state for persistence)
3. **Widget-driven**: User interactions trigger script reruns
4. **Automatic UI updates**: No need to manually refresh the interface

### Essential Commands
```python
# Must-know Streamlit commands for beginners
st.title()          # Main title
st.header()         # Section header
st.subheader()      # Subsection header
st.write()          # Universal display function
st.text()           # Plain text
st.markdown()       # Markdown text
st.success()        # Success message
st.error()          # Error message
st.warning()        # Warning message
st.info()           # Info message
```

### Development Checklist
- [ ] Install Streamlit and dependencies
- [ ] Create basic app structure
- [ ] Test local development server
- [ ] Understand rerun behavior
- [ ] Implement error handling
- [ ] Use unique widget keys
- [ ] Structure code in functions
- [ ] Add configuration file
- [ ] Test different browsers
- [ ] Document your code

## Next Steps
- Practice creating simple apps with different components
- Experiment with various widgets and layouts
- Learn about session state management
- Explore Streamlit's built-in themes and customization options
- Study the official documentation and examples
