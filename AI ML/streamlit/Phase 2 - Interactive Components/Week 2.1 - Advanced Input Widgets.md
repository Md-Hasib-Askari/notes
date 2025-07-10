# Week 2.1: Advanced Input Widgets

## Selection Widgets

### Radio Buttons
```python
import streamlit as st

st.title("🔘 Radio Button Examples")

# Basic radio button
payment_method = st.radio(
    "Choose payment method:",
    ["Credit Card", "PayPal", "Bank Transfer", "Cash"]
)
st.write(f"Selected: {payment_method}")

# Horizontal radio buttons
st.subheader("Horizontal Layout")
difficulty = st.radio(
    "Select difficulty level:",
    ["Easy", "Medium", "Hard", "Expert"],
    horizontal=True
)

# Radio with custom key and help
priority = st.radio(
    "Task Priority:",
    ["Low", "Medium", "High", "Critical"],
    index=1,  # Default to "Medium"
    key="task_priority",
    help="Choose the urgency level for this task"
)

# Conditional content based on radio selection
if difficulty == "Expert":
    st.warning("⚠️ Expert mode requires advanced knowledge!")
elif difficulty == "Easy":
    st.success("✅ Perfect for beginners!")

# Radio with icons
transport = st.radio(
    "Transportation method:",
    ["🚗 Car", "🚌 Bus", "🚇 Train", "🚲 Bicycle", "🚶 Walking"]
)
```

### Checkboxes
```python
import streamlit as st

st.subheader("☑️ Checkbox Examples")

# Single checkboxes with different purposes
newsletter = st.checkbox("📧 Subscribe to newsletter")
terms = st.checkbox("📋 I agree to the terms and conditions")
marketing = st.checkbox("📢 Receive marketing emails", value=True)  # Default checked

# Multiple related checkboxes
st.write("**Select your interests:**")
col1, col2, col3 = st.columns(3)

with col1:
    tech = st.checkbox("💻 Technology")
    science = st.checkbox("🔬 Science")
    business = st.checkbox("💼 Business")

with col2:
    sports = st.checkbox("⚽ Sports")
    music = st.checkbox("🎵 Music")
    travel = st.checkbox("✈️ Travel")

with col3:
    food = st.checkbox("🍕 Food")
    art = st.checkbox("🎨 Art")
    books = st.checkbox("📚 Books")

# Collect all interests
interests = []
for interest, checked in [
    ("Technology", tech), ("Science", science), ("Business", business),
    ("Sports", sports), ("Music", music), ("Travel", travel),
    ("Food", food), ("Art", art), ("Books", books)
]:
    if checked:
        interests.append(interest)

if interests:
    st.success(f"Your interests: {', '.join(interests)}")

# Checkbox for showing/hiding content
show_advanced = st.checkbox("🔧 Show advanced options")
if show_advanced:
    st.subheader("Advanced Settings")
    debug_mode = st.checkbox("Enable debug mode")
    verbose_logging = st.checkbox("Verbose logging")
    auto_save = st.checkbox("Auto-save every 5 minutes", value=True)
```

### Multiselect
```python
import streamlit as st

st.subheader("📋 Multiselect Examples")

# Basic multiselect
programming_languages = st.multiselect(
    "Select programming languages you know:",
    ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "Ruby", "PHP"],
    default=["Python"]  # Default selection
)

# Multiselect with more options
frameworks = st.multiselect(
    "Web frameworks experience:",
    [
        "React", "Vue.js", "Angular", "Django", "Flask", 
        "Express.js", "Spring Boot", "Laravel", "Rails"
    ],
    help="Select all frameworks you have worked with"
)

# Multiselect for data analysis tools
data_tools = st.multiselect(
    "Data analysis tools:",
    [
        "pandas", "NumPy", "Matplotlib", "Plotly", "Seaborn",
        "Scikit-learn", "TensorFlow", "PyTorch", "R", "SQL"
    ],
    placeholder="Choose your tools..."
)

# Dynamic content based on selections
if programming_languages:
    st.write(f"**Selected languages:** {', '.join(programming_languages)}")
    
    if "Python" in programming_languages and data_tools:
        st.info("🐍 Great! Python with data tools makes a powerful combination!")

# Multiselect for filtering
countries = [
    "United States", "Canada", "United Kingdom", "Germany", 
    "France", "Japan", "Australia", "Brazil", "India", "China"
]

selected_countries = st.multiselect(
    "Filter by countries:",
    countries,
    default=countries[:3]
)

if selected_countries:
    st.write(f"Showing data for: {len(selected_countries)} countries")
    for country in selected_countries:
        st.write(f"• {country}")
```

## Sliders

### Number Sliders
```python
import streamlit as st
import numpy as np

st.subheader("🎚️ Slider Examples")

# Basic number slider
age = st.slider(
    "Select your age:",
    min_value=0,
    max_value=100,
    value=25,
    step=1
)
st.write(f"Age: {age} years old")

# Float slider with custom format
price = st.slider(
    "Product price:",
    min_value=0.0,
    max_value=1000.0,
    value=99.99,
    step=0.01,
    format="$%.2f"
)

# Range slider (two values)
price_range = st.slider(
    "Price range filter:",
    min_value=0,
    max_value=1000,
    value=(100, 500),  # Default range
    step=10
)
st.write(f"Price range: ${price_range[0]} - ${price_range[1]}")

# Time-based slider
import datetime
start_time = datetime.time(9, 0)  # 9:00 AM
end_time = datetime.time(17, 0)   # 5:00 PM

meeting_time = st.slider(
    "Select meeting time:",
    min_value=start_time,
    max_value=end_time,
    value=datetime.time(14, 30),  # 2:30 PM
    step=datetime.timedelta(minutes=30),
    format="HH:mm"
)

# Interactive visualization with slider
st.subheader("📊 Interactive Chart")
num_points = st.slider("Number of data points:", 10, 200, 50)
noise_level = st.slider("Noise level:", 0.0, 2.0, 0.5, 0.1)

# Generate data based on slider values
x = np.linspace(0, 10, num_points)
y = np.sin(x) + np.random.normal(0, noise_level, num_points)

# Simple line chart
import pandas as pd
chart_data = pd.DataFrame({"x": x, "y": y})
st.line_chart(chart_data.set_index("x"))
```

### Select Slider
```python
import streamlit as st

st.subheader("🎛️ Select Slider Examples")

# Priority levels
priority = st.select_slider(
    "Task priority:",
    options=["Very Low", "Low", "Medium", "High", "Very High", "Critical"],
    value="Medium"
)

# Size options
size = st.select_slider(
    "T-shirt size:",
    options=["XS", "S", "M", "L", "XL", "XXL"],
    value="M"
)

# Custom scale
satisfaction = st.select_slider(
    "How satisfied are you?",
    options=[
        "😤 Very Dissatisfied",
        "😞 Dissatisfied", 
        "😐 Neutral",
        "😊 Satisfied",
        "😍 Very Satisfied"
    ],
    value="😐 Neutral"
)

# Educational levels
education = st.select_slider(
    "Education level:",
    options=[
        "Elementary", "Middle School", "High School",
        "Associate", "Bachelor's", "Master's", "PhD"
    ]
)

# Color intensity
color_intensity = st.select_slider(
    "Color intensity:",
    options=["Light", "Medium", "Dark", "Very Dark"],
    value="Medium"
)

# Display results
st.write("**Your Selections:**")
st.write(f"Priority: {priority}")
st.write(f"Size: {size}")
st.write(f"Satisfaction: {satisfaction}")
st.write(f"Education: {education}")
st.write(f"Color: {color_intensity}")
```

## File Uploads

### Basic File Upload
```python
import streamlit as st
import pandas as pd
import io

st.subheader("📁 File Upload Examples")

# Single file upload
uploaded_file = st.file_uploader(
    "Upload a file:",
    type=["txt", "csv", "xlsx", "json"],
    help="Supported formats: TXT, CSV, Excel, JSON"
)

if uploaded_file is not None:
    # File details
    st.success("✅ File uploaded successfully!")
    st.write("**File Details:**")
    st.write(f"• Name: {uploaded_file.name}")
    st.write(f"• Type: {uploaded_file.type}")
    st.write(f"• Size: {uploaded_file.size} bytes")
    
    # Process different file types
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
    elif uploaded_file.name.endswith('.txt'):
        content = str(uploaded_file.read(), "utf-8")
        st.subheader("📝 Text Content")
        st.text_area("File content:", content, height=200)
        
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
        st.subheader("📊 Excel Data")
        st.dataframe(df.head())
        
    elif uploaded_file.name.endswith('.json'):
        import json
        json_data = json.load(uploaded_file)
        st.subheader("🔧 JSON Data")
        st.json(json_data)

# Multiple file upload
st.subheader("📁 Multiple File Upload")
uploaded_files = st.file_uploader(
    "Upload multiple images:",
    type=["png", "jpg", "jpeg", "gif"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} files:")
    
    # Display images in columns
    cols = st.columns(min(len(uploaded_files), 3))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        col_idx = idx % 3
        with cols[col_idx]:
            st.image(uploaded_file, caption=uploaded_file.name, width=200)

# File upload with processing
st.subheader("⚙️ File Processing")
data_file = st.file_uploader(
    "Upload CSV for analysis:",
    type="csv",
    key="analysis_file"
)

if data_file is not None:
    df = pd.read_csv(data_file)
    
    # Basic analysis options
    analysis_type = st.selectbox(
        "Choose analysis:",
        ["Summary Statistics", "Missing Values", "Data Types", "Sample Data"]
    )
    
    if analysis_type == "Summary Statistics":
        st.write(df.describe())
    elif analysis_type == "Missing Values":
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
    elif analysis_type == "Data Types":
        st.write(df.dtypes)
    elif analysis_type == "Sample Data":
        sample_size = st.slider("Sample size:", 5, min(50, len(df)), 10)
        st.write(df.sample(sample_size))
```

## Date and Time Widgets

### Date Input
```python
import streamlit as st
from datetime import datetime, date, timedelta

st.subheader("📅 Date Input Examples")

# Basic date input
birth_date = st.date_input(
    "Enter your birth date:",
    value=date(1990, 1, 1),
    min_value=date(1900, 1, 1),
    max_value=date.today()
)

# Calculate and display age
if birth_date:
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    st.write(f"You are {age} years old")

# Date range selection
st.subheader("📅 Date Range")
date_range = st.date_input(
    "Select date range for report:",
    value=(date.today() - timedelta(days=7), date.today()),
    min_value=date(2020, 1, 1),
    max_value=date.today() + timedelta(days=365)
)

if len(date_range) == 2:
    start_date, end_date = date_range
    duration = (end_date - start_date).days
    st.write(f"Selected range: {duration} days")

# Project timeline
st.subheader("🗓️ Project Timeline")
project_start = st.date_input("Project start date:", date.today())
project_end = st.date_input(
    "Project end date:", 
    date.today() + timedelta(days=30),
    min_value=project_start
)

if project_end >= project_start:
    project_duration = (project_end - project_start).days
    st.info(f"Project duration: {project_duration} days")
else:
    st.error("End date must be after start date!")
```

### Time Input
```python
import streamlit as st
from datetime import time, datetime, timedelta

st.subheader("🕐 Time Input Examples")

# Basic time input
meeting_time = st.time_input(
    "Meeting time:",
    value=time(9, 30),
    help="Select your preferred meeting time"
)

# Work schedule
st.subheader("⏰ Work Schedule")
col1, col2 = st.columns(2)

with col1:
    work_start = st.time_input("Work start time:", time(9, 0))
with col2:
    work_end = st.time_input("Work end time:", time(17, 0))

# Calculate work hours
if work_end > work_start:
    work_duration = datetime.combine(date.today(), work_end) - datetime.combine(date.today(), work_start)
    hours = work_duration.total_seconds() / 3600
    st.write(f"Work hours: {hours:.1f} hours")

# Appointment scheduling
st.subheader("📋 Appointment Scheduling")
appointment_date = st.date_input("Appointment date:")
appointment_time = st.time_input("Appointment time:")

if appointment_date and appointment_time:
    appointment_datetime = datetime.combine(appointment_date, appointment_time)
    now = datetime.now()
    
    if appointment_datetime > now:
        time_until = appointment_datetime - now
        st.success(f"Appointment scheduled for {appointment_datetime.strftime('%Y-%m-%d %H:%M')}")
        st.info(f"Time until appointment: {time_until}")
    else:
        st.warning("Please select a future date and time")

# Time zone consideration
st.subheader("🌍 Time Zone Example")
local_time = st.time_input("Local time:", time(12, 0))
timezone_offset = st.selectbox(
    "Your timezone:",
    [
        ("UTC-8 (PST)", -8),
        ("UTC-5 (EST)", -5),
        ("UTC+0 (GMT)", 0),
        ("UTC+1 (CET)", 1),
        ("UTC+9 (JST)", 9)
    ],
    format_func=lambda x: x[0]
)

if local_time and timezone_offset:
    # Convert to UTC
    local_datetime = datetime.combine(date.today(), local_time)
    utc_datetime = local_datetime - timedelta(hours=timezone_offset[1])
    st.write(f"UTC time: {utc_datetime.time()}")
```

## Advanced Widget Combinations

### Dynamic Form Builder
```python
import streamlit as st

st.subheader("🔧 Dynamic Form Example")

# Form configuration
form_type = st.selectbox(
    "Form type:",
    ["Contact Form", "Survey", "Registration", "Feedback"]
)

if form_type == "Contact Form":
    with st.form("contact_form"):
        st.write("📞 Contact Information")
        
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name*")
            email = st.text_input("Email*")
        with col2:
            last_name = st.text_input("Last Name*")
            phone = st.text_input("Phone")
        
        subject = st.selectbox("Subject:", ["General", "Support", "Sales", "Other"])
        message = st.text_area("Message*", height=100)
        
        priority = st.radio("Priority:", ["Low", "Medium", "High"], horizontal=True)
        contact_method = st.multiselect("Preferred contact method:", ["Email", "Phone", "SMS"])
        
        submitted = st.form_submit_button("Submit Contact Form")
        
        if submitted:
            if first_name and last_name and email and message:
                st.success("✅ Contact form submitted successfully!")
            else:
                st.error("❌ Please fill in all required fields")

elif form_type == "Survey":
    with st.form("survey_form"):
        st.write("📊 Customer Satisfaction Survey")
        
        # Rating questions
        overall_rating = st.select_slider(
            "Overall satisfaction:",
            options=["Very Poor", "Poor", "Fair", "Good", "Excellent"]
        )
        
        service_rating = st.slider("Service quality (1-10):", 1, 10, 5)
        
        # Multiple choice
        recommend = st.radio(
            "Would you recommend us?",
            ["Definitely", "Probably", "Maybe", "Probably Not", "Definitely Not"]
        )
        
        # Checkboxes for features
        st.write("Which features do you use?")
        features = ["Feature A", "Feature B", "Feature C", "Feature D"]
        selected_features = []
        
        for feature in features:
            if st.checkbox(feature, key=f"feature_{feature}"):
                selected_features.append(feature)
        
        comments = st.text_area("Additional comments:", height=80)
        
        submitted = st.form_submit_button("Submit Survey")
        
        if submitted:
            st.success("📊 Survey submitted! Thank you for your feedback.")
            st.write(f"Overall rating: {overall_rating}")
            st.write(f"Service rating: {service_rating}/10")
            st.write(f"Features used: {', '.join(selected_features) if selected_features else 'None'}")
```

## Best Practices

### Widget Organization
```python
import streamlit as st

# Use consistent naming and organization
st.title("📋 Best Practices Demo")

# Group related widgets
with st.expander("👤 Personal Information"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        age = st.number_input("Age", 18, 100, 25)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        country = st.selectbox("Country", ["USA", "Canada", "UK", "Other"])

with st.expander("⚙️ Preferences"):
    notifications = st.checkbox("Enable notifications")
    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)
    language = st.selectbox("Language", ["English", "Spanish", "French"])

# Validation and feedback
if st.button("Save Preferences"):
    if name and age:
        st.success("✅ Preferences saved successfully!")
        # Show summary
        st.json({
            "name": name,
            "age": age,
            "gender": gender,
            "country": country,
            "notifications": notifications,
            "theme": theme,
            "language": language
        })
    else:
        st.error("❌ Please fill in required fields")
```

## Key Concepts

### Widget State Management
- All widgets maintain their state automatically
- Use unique `key` parameters to avoid conflicts
- Widget values persist across script reruns
- Form widgets only update on form submission

### User Experience Tips
1. **Group related widgets** using containers and expanders
2. **Provide clear labels** and help text
3. **Set sensible defaults** for better UX
4. **Validate inputs** and provide feedback
5. **Use consistent spacing** and layout

### Performance Considerations
- Widgets trigger script reruns on interaction
- Use forms to batch multiple inputs
- Consider using session state for complex interactions
- Cache expensive operations that depend on widget values

## Next Steps
- Practice combining multiple widget types
- Learn about layout containers and organization
- Explore form submission and validation patterns
- Build more complex interactive applications
