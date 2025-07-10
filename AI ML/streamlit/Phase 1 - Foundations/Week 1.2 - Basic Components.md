# Week 1.2: Basic Components

## Text Elements

### Titles and Headers
```python
import streamlit as st

# Page title (appears in browser tab)
st.set_page_config(page_title="Text Elements Demo")

# Main title - largest text
st.title("🎯 Main Application Title")

# Headers - for section organization
st.header("📝 Section Header")
st.subheader("📄 Subsection Header")

# Alternative header styles
st.markdown("# Alternative Title Style")
st.markdown("## Alternative Header Style")
st.markdown("### Alternative Subheader Style")
```

### Text Display Methods
```python
import streamlit as st

# Plain text
st.text("This is plain text - monospace font")

# Simple write - most versatile
st.write("This is the most versatile display method")
st.write("You can mix", "multiple", "arguments")

# Markdown support
st.markdown("""
## Markdown Features
- **Bold text**
- *Italic text*
- `Code snippets`
- [Links](https://streamlit.io)
- Lists and more!
""")

# Formatted strings
name = "Alice"
age = 30
st.write(f"Hello {name}, you are {age} years old!")

# LaTeX support
st.latex(r'''
    E = mc^2
    ''')

# Code display
code = '''
def hello():
    print("Hello, Streamlit!")
'''
st.code(code, language='python')
```

### Advanced Text Formatting
```python
import streamlit as st

# Colored text with markdown
st.markdown("""
<div style='color: red; font-size: 20px;'>
This is red text
</div>
""", unsafe_allow_html=True)

# Success, error, warning, info messages
st.success("✅ Operation completed successfully!")
st.error("❌ Something went wrong")
st.warning("⚠️ Please check your input")
st.info("ℹ️ Additional information")

# Caption text (smaller, lighter)
st.caption("This is a caption - useful for additional context")

# Metric display
st.metric(
    label="Temperature",
    value="70°F",
    delta="1.2°F"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sales", "1,234", "12%")
with col2:
    st.metric("Users", "5,678", "-2%")
with col3:
    st.metric("Revenue", "$9,012", "8%")
```

## Display Data

### Basic Data Display
```python
import streamlit as st
import pandas as pd
import numpy as np

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Sydney'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)

# Display with st.write() - automatic formatting
st.subheader("Using st.write()")
st.write(df)

# Display as interactive dataframe
st.subheader("Using st.dataframe()")
st.dataframe(df)

# Display as static table
st.subheader("Using st.table()")
st.table(df)
```

### Advanced Dataframe Features
```python
import streamlit as st
import pandas as pd
import numpy as np

# Create larger dataset
np.random.seed(42)
large_df = pd.DataFrame({
    'Product': [f'Product_{i}' for i in range(100)],
    'Sales': np.random.randint(100, 1000, 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'Date': pd.date_range('2023-01-01', periods=100, freq='D')
})

# Interactive dataframe with customization
st.subheader("Enhanced Dataframe Display")
st.dataframe(
    large_df,
    use_container_width=True,  # Fill container width
    height=400,                # Set height
    hide_index=True           # Hide row indices
)

# Highlight specific data
st.subheader("Highlighted Data")
def highlight_high_sales(val):
    color = 'background-color: yellow' if val > 500 else ''
    return color

styled_df = large_df.style.applymap(
    highlight_high_sales, 
    subset=['Sales']
)
st.dataframe(styled_df)
```

### JSON and Dictionary Display
```python
import streamlit as st

# Dictionary display
person = {
    'name': 'John Doe',
    'age': 30,
    'skills': ['Python', 'Streamlit', 'Data Analysis'],
    'contact': {
        'email': 'john@example.com',
        'phone': '+1-234-567-8900'
    }
}

st.subheader("Dictionary Display")
st.write(person)

# JSON display
st.subheader("JSON Display")
st.json(person)

# Display with expandable structure
with st.expander("Click to see detailed person info"):
    st.json(person)
```

## Input Widgets

### Text Input Widgets
```python
import streamlit as st

st.header("Text Input Widgets")

# Basic text input
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

# Text input with default value
email = st.text_input(
    "Email address:",
    value="user@example.com",
    placeholder="Enter your email"
)

# Password input
password = st.text_input(
    "Password:",
    type="password",
    help="Enter a secure password"
)

# Text area for longer text
feedback = st.text_area(
    "Feedback:",
    height=100,
    placeholder="Share your thoughts..."
)

# Display inputs
if st.button("Show Inputs"):
    st.write("Name:", name)
    st.write("Email:", email)
    st.write("Password entered:", "Yes" if password else "No")
    st.write("Feedback:", feedback)
```

### Numeric Input Widgets
```python
import streamlit as st

st.header("Numeric Input Widgets")

# Number input
age = st.number_input(
    "Enter your age:",
    min_value=0,
    max_value=150,
    value=25,
    step=1
)

# Float input
price = st.number_input(
    "Product price:",
    min_value=0.0,
    max_value=1000.0,
    value=99.99,
    step=0.01,
    format="%.2f"
)

# Slider
rating = st.slider(
    "Rate this app:",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = Poor, 5 = Excellent"
)

# Range slider
price_range = st.slider(
    "Price range:",
    min_value=0,
    max_value=1000,
    value=(100, 500),  # Default range
    step=10
)

# Display results
st.write(f"Age: {age}")
st.write(f"Price: ${price:.2f}")
st.write(f"Rating: {rating}/5")
st.write(f"Price range: ${price_range[0]} - ${price_range[1]}")
```

### Selection Widgets
```python
import streamlit as st

st.header("Selection Widgets")

# Selectbox (dropdown)
favorite_color = st.selectbox(
    "Choose your favorite color:",
    ["Red", "Green", "Blue", "Yellow", "Purple"]
)

# Multiselect
hobbies = st.multiselect(
    "Select your hobbies:",
    ["Reading", "Sports", "Music", "Travel", "Cooking", "Gaming"],
    default=["Reading", "Music"]
)

# Radio buttons
education = st.radio(
    "Highest education level:",
    ["High School", "Bachelor's", "Master's", "PhD"],
    horizontal=True
)

# Checkbox
newsletter = st.checkbox("Subscribe to newsletter")
terms = st.checkbox("I agree to the terms and conditions")

# Select slider
priority = st.select_slider(
    "Priority level:",
    options=["Low", "Medium", "High", "Critical"],
    value="Medium"
)

# Display selections
if st.button("Show Selections"):
    st.write(f"Favorite color: {favorite_color}")
    st.write(f"Hobbies: {', '.join(hobbies)}")
    st.write(f"Education: {education}")
    st.write(f"Newsletter: {'Yes' if newsletter else 'No'}")
    st.write(f"Terms accepted: {'Yes' if terms else 'No'}")
    st.write(f"Priority: {priority}")
```

### Date and Time Widgets
```python
import streamlit as st
from datetime import datetime, date, time

st.header("Date and Time Widgets")

# Date input
birth_date = st.date_input(
    "Enter your birth date:",
    value=date(1990, 1, 1),
    min_value=date(1900, 1, 1),
    max_value=date.today()
)

# Time input
meeting_time = st.time_input(
    "Meeting time:",
    value=time(9, 30)
)

# Datetime input (using columns)
col1, col2 = st.columns(2)
with col1:
    event_date = st.date_input("Event date:")
with col2:
    event_time = st.time_input("Event time:")

# Calculate age
if birth_date:
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    st.write(f"You are {age} years old")

# Display datetime combination
if event_date and event_time:
    event_datetime = datetime.combine(event_date, event_time)
    st.write(f"Event scheduled for: {event_datetime.strftime('%Y-%m-%d %H:%M')}")
```

## Button Interactions

### Basic Button Usage
```python
import streamlit as st

st.header("Button Interactions")

# Basic button
if st.button("Click me!"):
    st.write("Button was clicked! 🎉")

# Button with custom styling
if st.button("🚀 Launch", type="primary"):
    st.balloons()  # Fun animation
    st.success("Launched successfully!")

# Button with help text
if st.button("Help Button", help="Click this for assistance"):
    st.info("This is the help information")

# Disabled button state
disabled = st.checkbox("Disable button")
if st.button("Maybe Disabled", disabled=disabled):
    st.write("This button was enabled and clicked!")
```

### Interactive Button Examples
```python
import streamlit as st
import random

st.header("Interactive Button Examples")

# Counter example
if 'counter' not in st.session_state:
    st.session_state.counter = 0

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("➕ Increment"):
        st.session_state.counter += 1

with col2:
    if st.button("➖ Decrement"):
        st.session_state.counter -= 1

with col3:
    if st.button("🔄 Reset"):
        st.session_state.counter = 0

st.write(f"Current count: {st.session_state.counter}")

# Random number generator
if st.button("🎲 Generate Random Number"):
    random_num = random.randint(1, 100)
    st.write(f"Random number: {random_num}")

# Form submission simulation
with st.form("user_form"):
    st.subheader("User Information Form")
    
    form_name = st.text_input("Name:")
    form_email = st.text_input("Email:")
    form_age = st.number_input("Age:", min_value=1, max_value=100)
    
    submitted = st.form_submit_button("Submit Form")
    
    if submitted:
        if form_name and form_email:
            st.success(f"Form submitted for {form_name}")
            st.write(f"Email: {form_email}")
            st.write(f"Age: {form_age}")
        else:
            st.error("Please fill in all required fields")
```

### Advanced Button Patterns
```python
import streamlit as st
import time

st.header("Advanced Button Patterns")

# Toggle button behavior
if 'toggle_state' not in st.session_state:
    st.session_state.toggle_state = False

if st.button("🔄 Toggle Mode"):
    st.session_state.toggle_state = not st.session_state.toggle_state

if st.session_state.toggle_state:
    st.success("Mode is ON")
else:
    st.info("Mode is OFF")

# Progress button
if st.button("⏳ Start Process"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f'Progress: {i+1}%')
        time.sleep(0.01)
    
    status_text.text('Process completed!')
    st.success("All done!")

# Confirmation pattern
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = False

if not st.session_state.confirm_delete:
    if st.button("🗑️ Delete Item"):
        st.session_state.confirm_delete = True
        st.rerun()
else:
    st.warning("Are you sure you want to delete this item?")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Yes, Delete"):
            st.success("Item deleted!")
            st.session_state.confirm_delete = False
    
    with col2:
        if st.button("❌ Cancel"):
            st.session_state.confirm_delete = False
```

## Practical Examples

### Simple Calculator
```python
import streamlit as st

st.title("🧮 Simple Calculator")

# Input numbers
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("First number:", value=0.0)
with col2:
    num2 = st.number_input("Second number:", value=0.0)

# Operation selection
operation = st.selectbox(
    "Choose operation:",
    ["Addition (+)", "Subtraction (-)", "Multiplication (×)", "Division (÷)"]
)

# Calculate button
if st.button("Calculate"):
    if operation == "Addition (+)":
        result = num1 + num2
        st.success(f"{num1} + {num2} = {result}")
    elif operation == "Subtraction (-)":
        result = num1 - num2
        st.success(f"{num1} - {num2} = {result}")
    elif operation == "Multiplication (×)":
        result = num1 * num2
        st.success(f"{num1} × {num2} = {result}")
    elif operation == "Division (÷)":
        if num2 != 0:
            result = num1 / num2
            st.success(f"{num1} ÷ {num2} = {result}")
        else:
            st.error("Cannot divide by zero!")
```

### Text Formatter Tool
```python
import streamlit as st

st.title("📝 Text Formatter Tool")

# Input text
user_text = st.text_area("Enter your text:", height=100)

if user_text:
    # Formatting options
    st.subheader("Formatting Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uppercase = st.checkbox("UPPERCASE")
        lowercase = st.checkbox("lowercase")
        title_case = st.checkbox("Title Case")
    
    with col2:
        remove_spaces = st.checkbox("Remove extra spaces")
        word_count = st.checkbox("Show word count")
        char_count = st.checkbox("Show character count")
    
    # Apply formatting
    formatted_text = user_text
    
    if uppercase:
        formatted_text = formatted_text.upper()
    elif lowercase:
        formatted_text = formatted_text.lower()
    elif title_case:
        formatted_text = formatted_text.title()
    
    if remove_spaces:
        formatted_text = " ".join(formatted_text.split())
    
    # Display results
    st.subheader("Formatted Text")
    st.text_area("Result:", formatted_text, height=100, disabled=True)
    
    # Show statistics
    if word_count or char_count:
        col1, col2 = st.columns(2)
        
        if word_count:
            with col1:
                words = len(formatted_text.split())
                st.metric("Word Count", words)
        
        if char_count:
            with col2:
                chars = len(formatted_text)
                st.metric("Character Count", chars)
```

## Key Concepts Summary

### Widget State Management
- All widgets return values immediately
- Widget values persist during script reruns
- Use unique keys for widgets to avoid conflicts
- Session state can store persistent data

### Best Practices
1. **Organize content**: Use headers and subheaders
2. **Validate inputs**: Check for empty or invalid values
3. **Provide feedback**: Use success/error messages
4. **Use help text**: Guide users with helpful descriptions
5. **Group related widgets**: Use columns and containers

### Common Patterns
- Form submission with validation
- Progressive disclosure with expanders
- Conditional display based on user input
- Real-time updates with automatic reruns

## Next Steps
- Experiment with different widget combinations
- Practice building interactive forms
- Learn about layout components (columns, containers)
- Explore advanced widget features and customization
