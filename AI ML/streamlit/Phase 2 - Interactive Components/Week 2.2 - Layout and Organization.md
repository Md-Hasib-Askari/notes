# Week 2.2: Layout and Organization

## Columns

### Basic Column Layouts
```python
import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 Column Layout Examples")

# Two equal columns
st.subheader("Two Equal Columns")
col1, col2 = st.columns(2)

with col1:
    st.write("**Left Column**")
    st.write("This content is in the left column")
    age = st.slider("Age", 18, 65, 25)

with col2:
    st.write("**Right Column**")
    st.write("This content is in the right column")
    name = st.text_input("Name")

# Three columns with different widths
st.subheader("Custom Width Columns")
col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is twice as wide

with col1:
    st.button("🔴 Red", use_container_width=True)

with col2:
    st.selectbox("Choose option:", ["Option A", "Option B", "Option C"])

with col3:
    st.button("🟢 Green", use_container_width=True)

# Four columns for metrics
st.subheader("Metrics Dashboard")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Sales", "$12,345", "+5%")

with col2:
    st.metric("Users", "1,234", "+12%")

with col3:
    st.metric("Revenue", "$45,678", "+8%")

with col4:
    st.metric("Growth", "23%", "+2%")
```

### Advanced Column Usage
```python
import streamlit as st
import plotly.express as px

st.subheader("📈 Data Visualization Layout")

# Sample data
df = px.data.iris()

# Left column for controls, right for visualization
control_col, viz_col = st.columns([1, 3])

with control_col:
    st.write("**Chart Controls**")
    
    x_axis = st.selectbox(
        "X-axis:",
        df.columns[:-1]  # Exclude species column
    )
    
    y_axis = st.selectbox(
        "Y-axis:",
        df.columns[:-1],
        index=1
    )
    
    color_by = st.selectbox(
        "Color by:",
        ["species", "None"]
    )
    
    chart_type = st.radio(
        "Chart type:",
        ["Scatter", "Line", "Bar"]
    )
    
    show_trend = st.checkbox("Show trendline")

with viz_col:
    st.write("**Visualization**")
    
    # Create chart based on selections
    if chart_type == "Scatter":
        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            color="species" if color_by == "species" else None,
            trendline="ols" if show_trend else None
        )
    elif chart_type == "Line":
        fig = px.line(df.sort_values(x_axis), x=x_axis, y=y_axis)
    else:  # Bar
        fig = px.bar(df.groupby("species")[y_axis].mean().reset_index(), 
                    x="species", y=y_axis)
    
    st.plotly_chart(fig, use_container_width=True)

# Image gallery with columns
st.subheader("🖼️ Image Gallery Layout")
image_cols = st.columns(3)

# Placeholder images
for i, col in enumerate(image_cols):
    with col:
        st.image(f"https://via.placeholder.com/300x200/{'FF6B6B' if i==0 else '4ECDC4' if i==1 else 'FFE66D'}/FFFFFF?text=Image+{i+1}")
        st.caption(f"Image {i+1} caption")
        if st.button(f"View Details", key=f"img_{i}"):
            st.info(f"Showing details for Image {i+1}")
```

## Containers

### Basic Containers
```python
import streamlit as st
import time

st.title("📦 Container Examples")

# Basic container
st.subheader("Basic Container")
container = st.container()
container.write("This content is inside a container")
container.button("Container Button")

# You can add content to container later
st.write("This is outside the container")
container.write("This is added to the container later")

# Container with border (using custom CSS)
st.subheader("Styled Container")
with st.container():
    st.markdown("""
    <div style='padding: 20px; border: 2px solid #1f77b4; border-radius: 10px; background-color: #f0f8ff;'>
        <h4>📝 Important Information</h4>
        <p>This container has custom styling with border and background color.</p>
    </div>
    """, unsafe_allow_html=True)

# Multiple containers for organization
st.subheader("Organized Content")

# Header container
header_container = st.container()
with header_container:
    st.markdown("### 🏠 Dashboard Header")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Users", "1,234")
    with col2:
        st.metric("Revenue", "$56,789")
    with col3:
        st.metric("Orders", "890")

# Content container
content_container = st.container()
with content_container:
    st.markdown("### 📊 Main Content")
    
    tab1, tab2 = st.tabs(["Sales", "Analytics"])
    
    with tab1:
        st.line_chart(np.random.randn(20, 3))
    
    with tab2:
        st.bar_chart(np.random.randn(20, 3))

# Footer container
footer_container = st.container()
with footer_container:
    st.markdown("---")
    st.markdown("*Dashboard footer - Last updated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "*")
```

### Empty Containers for Dynamic Content
```python
import streamlit as st
import time

st.subheader("🔄 Dynamic Content with Empty Containers")

# Empty container for dynamic updates
placeholder = st.empty()

# Control buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Show Chart"):
        with placeholder.container():
            st.subheader("📈 Sales Chart")
            chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns=['A', 'B', 'C']
            )
            st.line_chart(chart_data)

with col2:
    if st.button("Show Table"):
        with placeholder.container():
            st.subheader("📋 Data Table")
            df = pd.DataFrame({
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['NY', 'LA', 'Chicago']
            })
            st.dataframe(df)

with col3:
    if st.button("Clear"):
        placeholder.empty()

# Progress bar example with empty container
if st.button("Start Process"):
    progress_placeholder = st.empty()
    
    for i in range(100):
        progress_placeholder.progress(i + 1)
        time.sleep(0.01)
    
    progress_placeholder.success("✅ Process completed!")
```

## Sidebar

### Basic Sidebar Usage
```python
import streamlit as st
import plotly.express as px

st.title("🗂️ Sidebar Navigation Example")

# Sidebar header
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Use these controls to customize the view")

# Navigation
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Home", "Data Analysis", "Settings", "About"]
)

# Filters in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

# Data filters
data_source = st.sidebar.radio(
    "Data source:",
    ["Sample Data", "Upload File", "Database"]
)

date_range = st.sidebar.date_input(
    "Date range:",
    value=[pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31')]
)

show_raw_data = st.sidebar.checkbox("Show raw data")

# Sidebar with file upload
if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV:",
        type="csv"
    )

# Visualization controls
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chart Options")

chart_height = st.sidebar.slider("Chart height:", 300, 800, 400)
show_legend = st.sidebar.checkbox("Show legend", value=True)

# Main content based on sidebar selections
if page == "Home":
    st.header("🏠 Welcome to the Dashboard")
    st.write("Use the sidebar to navigate and configure options.")
    
    if show_raw_data:
        st.subheader("📋 Sample Data")
        df = px.data.iris()
        st.dataframe(df)

elif page == "Data Analysis":
    st.header("📈 Data Analysis")
    
    # Use sidebar selections
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    fig.update_layout(height=chart_height, showlegend=show_legend)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Settings":
    st.header("⚙️ Settings")
    st.write("Configuration options will go here.")

elif page == "About":
    st.header("ℹ️ About")
    st.write("This is a demo of sidebar navigation and controls.")
```

### Advanced Sidebar Features
```python
import streamlit as st

# Collapsible sidebar sections
with st.sidebar:
    st.title("🎛️ Advanced Controls")
    
    # User profile section
    with st.expander("👤 User Profile"):
        user_name = st.text_input("Username")
        user_role = st.selectbox("Role", ["Admin", "User", "Guest"])
        st.button("Update Profile")
    
    # Theme settings
    with st.expander("🎨 Theme Settings"):
        theme = st.radio("Theme:", ["Light", "Dark"])
        accent_color = st.color_picker("Accent color:", "#FF6B6B")
        font_size = st.slider("Font size:", 12, 20, 14)
    
    # Advanced filters
    with st.expander("🔍 Advanced Filters"):
        price_range = st.slider("Price range:", 0, 1000, (100, 500))
        categories = st.multiselect(
            "Categories:",
            ["Electronics", "Clothing", "Books", "Home"]
        )
        in_stock = st.checkbox("In stock only")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Export Data")
    export_format = st.selectbox("Format:", ["CSV", "JSON", "Excel"])
    if st.button("Download", use_container_width=True):
        st.success("Download started!")
```

## Tabs

### Basic Tab Usage
```python
import streamlit as st
import pandas as pd
import numpy as np

st.title("📑 Tab Examples")

# Basic tabs
tab1, tab2, tab3 = st.tabs(["📈 Charts", "📋 Data", "⚙️ Settings"])

with tab1:
    st.header("Charts and Visualizations")
    
    chart_type = st.selectbox("Chart type:", ["Line", "Bar", "Area"])
    
    # Generate sample data
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Series A', 'Series B', 'Series C']
    )
    
    if chart_type == "Line":
        st.line_chart(chart_data)
    elif chart_type == "Bar":
        st.bar_chart(chart_data)
    else:  # Area
        st.area_chart(chart_data)

with tab2:
    st.header("Data Tables")
    
    # Data options
    col1, col2 = st.columns(2)
    with col1:
        rows = st.slider("Number of rows:", 5, 50, 20)
    with col2:
        columns = st.multiselect(
            "Show columns:",
            ['Series A', 'Series B', 'Series C'],
            default=['Series A', 'Series B', 'Series C']
        )
    
    # Display filtered data
    display_data = chart_data[columns].head(rows)
    st.dataframe(display_data, use_container_width=True)
    
    # Summary statistics
    if st.checkbox("Show statistics"):
        st.write(display_data.describe())

with tab3:
    st.header("Application Settings")
    
    # Settings form
    with st.form("settings_form"):
        auto_refresh = st.checkbox("Auto-refresh data")
        refresh_interval = st.slider("Refresh interval (seconds):", 5, 60, 10)
        enable_notifications = st.checkbox("Enable notifications")
        
        # Theme settings
        st.subheader("Display Settings")
        dark_mode = st.checkbox("Dark mode")
        show_grid = st.checkbox("Show grid lines", value=True)
        
        if st.form_submit_button("Save Settings"):
            st.success("Settings saved successfully!")
```

### Dynamic Tabs
```python
import streamlit as st

st.subheader("🔄 Dynamic Tab Content")

# Tab configuration
num_tabs = st.slider("Number of tabs:", 2, 6, 3)

# Create dynamic tabs
tab_names = [f"Tab {i+1}" for i in range(num_tabs)]
tabs = st.tabs(tab_names)

# Populate tabs with content
for i, tab in enumerate(tabs):
    with tab:
        st.write(f"**Content for Tab {i+1}**")
        
        # Different content types for different tabs
        if i % 3 == 0:
            # Chart content
            data = np.random.randn(10, 2)
            df = pd.DataFrame(data, columns=['X', 'Y'])
            st.line_chart(df)
            
        elif i % 3 == 1:
            # Form content
            with st.form(f"form_{i}"):
                name = st.text_input(f"Name for tab {i+1}:")
                value = st.number_input(f"Value for tab {i+1}:", 0, 100, 50)
                submitted = st.form_submit_button(f"Submit Tab {i+1}")
                
                if submitted:
                    st.success(f"Submitted: {name} = {value}")
        else:
            # Info content
            st.info(f"This is informational content for Tab {i+1}")
            st.write("Some additional text and information goes here.")
            
            if st.button(f"Action for Tab {i+1}"):
                st.balloons()
```

## Expanders

### Basic Expanders
```python
import streamlit as st

st.title("🔽 Expander Examples")

# Basic expander
with st.expander("📋 Click to see more details"):
    st.write("This content is hidden by default and revealed when expanded.")
    st.write("You can put any Streamlit components inside an expander.")
    
    # Components inside expander
    user_input = st.text_input("Enter some text:")
    if user_input:
        st.write(f"You entered: {user_input}")

# Expander with custom label
with st.expander("🔍 Advanced Search Options", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        search_type = st.selectbox("Search type:", ["Exact", "Fuzzy", "Regex"])
        case_sensitive = st.checkbox("Case sensitive")
    
    with col2:
        date_filter = st.date_input("Filter by date:")
        category_filter = st.multiselect("Categories:", ["A", "B", "C"])

# Multiple expanders for organization
st.subheader("📚 FAQ Section")

faqs = [
    ("How do I get started?", "Simply follow the installation guide and run your first app."),
    ("What are the system requirements?", "Python 3.7+ and a modern web browser."),
    ("Can I deploy my app?", "Yes! You can deploy to Streamlit Cloud, Heroku, or other platforms."),
    ("Is it free to use?", "Yes, Streamlit is open source and free to use.")
]

for question, answer in faqs:
    with st.expander(f"❓ {question}"):
        st.write(answer)
        if st.button("Was this helpful?", key=f"helpful_{question}"):
            st.success("Thank you for your feedback!")

# Expander with complex content
with st.expander("📊 Data Analysis Tools", expanded=True):
    st.write("**Available Analysis Tools:**")
    
    tools = {
        "Statistical Analysis": "Descriptive statistics, correlation analysis, hypothesis testing",
        "Data Visualization": "Charts, plots, interactive visualizations",
        "Machine Learning": "Classification, regression, clustering algorithms",
        "Time Series": "Trend analysis, forecasting, seasonality detection"
    }
    
    selected_tools = []
    
    for tool, description in tools.items():
        if st.checkbox(tool):
            selected_tools.append(tool)
            st.write(f"*{description}*")
    
    if selected_tools:
        st.success(f"Selected tools: {', '.join(selected_tools)}")
```

### Nested Expanders
```python
import streamlit as st

st.subheader("🗂️ Nested Expanders")

with st.expander("🏢 Company Information"):
    st.write("**ABC Corporation**")
    
    with st.expander("📍 Locations"):
        locations = ["New York", "London", "Tokyo", "Sydney"]
        for location in locations:
            with st.expander(f"🌍 {location} Office"):
                st.write(f"Address: 123 Main St, {location}")
                st.write(f"Phone: +1-555-{location[:2].upper()}-0000")
                st.write(f"Employees: {np.random.randint(50, 500)}")
    
    with st.expander("👥 Departments"):
        departments = ["Engineering", "Sales", "Marketing", "HR"]
        for dept in departments:
            with st.expander(f"🏛️ {dept}"):
                st.write(f"Head: John Doe")
                st.write(f"Team size: {np.random.randint(10, 100)}")
                st.write(f"Budget: ${np.random.randint(100, 1000)}k")

# Conditional expanders
st.subheader("🔧 Conditional Content")

user_level = st.selectbox("User level:", ["Beginner", "Intermediate", "Advanced"])

if user_level == "Beginner":
    with st.expander("📖 Getting Started Guide"):
        st.write("1. Install Streamlit")
        st.write("2. Create your first app")
        st.write("3. Learn basic components")
        
elif user_level == "Intermediate":
    with st.expander("🔧 Intermediate Features"):
        st.write("- Session state management")
        st.write("- Custom components")
        st.write("- Advanced layouts")
        
else:  # Advanced
    with st.expander("🚀 Advanced Topics"):
        st.write("- Performance optimization")
        st.write("- Custom theming")
        st.write("- Deployment strategies")
```

## Layout Best Practices

### Responsive Design Patterns
```python
import streamlit as st

st.title("📱 Responsive Layout Examples")

# Mobile-friendly layout
st.subheader("Mobile-Friendly Design")

# Use columns that work well on mobile
if st.checkbox("Mobile view simulation"):
    # Single column layout for mobile
    st.write("**Mobile Layout (Single Column)**")
    st.metric("Sales", "$12,345", "+5%")
    st.metric("Users", "1,234", "+12%")
    st.metric("Revenue", "$45,678", "+8%")
else:
    # Multi-column layout for desktop
    st.write("**Desktop Layout (Multi-Column)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sales", "$12,345", "+5%")
    with col2:
        st.metric("Users", "1,234", "+12%")
    with col3:
        st.metric("Revenue", "$45,678", "+8%")

# Progressive disclosure
st.subheader("📋 Progressive Disclosure")

basic_info = st.container()
with basic_info:
    st.write("**Basic Information**")
    name = st.text_input("Name")
    email = st.text_input("Email")

if name and email:  # Only show advanced options when basic info is filled
    with st.expander("🔧 Advanced Options"):
        notifications = st.checkbox("Enable notifications")
        theme = st.selectbox("Theme", ["Light", "Dark"])
        language = st.selectbox("Language", ["English", "Spanish", "French"])

# Contextual layout
st.subheader("🎯 Contextual Layout")

task_type = st.selectbox("Task type:", ["Data Entry", "Analysis", "Reporting"])

if task_type == "Data Entry":
    # Form-heavy layout
    with st.form("data_entry"):
        col1, col2 = st.columns(2)
        with col1:
            product = st.text_input("Product")
            quantity = st.number_input("Quantity", min_value=1)
        with col2:
            price = st.number_input("Price", min_value=0.01, format="%.2f")
            category = st.selectbox("Category", ["A", "B", "C"])
        
        submitted = st.form_submit_button("Add Entry")

elif task_type == "Analysis":
    # Chart-heavy layout
    chart_col, control_col = st.columns([3, 1])
    
    with control_col:
        st.write("**Controls**")
        metric = st.selectbox("Metric", ["Sales", "Users", "Revenue"])
        period = st.selectbox("Period", ["Daily", "Weekly", "Monthly"])
    
    with chart_col:
        st.write("**Analysis Chart**")
        # Placeholder for chart
        st.line_chart(np.random.randn(30, 1))

else:  # Reporting
    # Document-style layout
    st.write("**📊 Executive Report**")
    
    # Report sections in expanders
    with st.expander("📈 Sales Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sales", "$125,000")
        with col2:
            st.metric("Growth", "+15%")
        with col3:
            st.metric("Target", "85%")
    
    with st.expander("📋 Key Insights"):
        st.write("• Sales increased by 15% compared to last quarter")
        st.write("• Customer acquisition rate improved by 8%")
        st.write("• Product A remains the top performer")
```

## Key Concepts Summary

### Layout Hierarchy
1. **Page Level**: Main content area
2. **Containers**: Group related content
3. **Columns**: Horizontal organization
4. **Tabs**: Separate views/sections
5. **Expanders**: Collapsible content
6. **Sidebar**: Persistent navigation/controls

### Best Practices
- **Progressive Disclosure**: Show information as needed
- **Consistent Spacing**: Use columns for alignment
- **Logical Grouping**: Related items together
- **Mobile Consideration**: Test on different screen sizes
- **Clear Navigation**: Intuitive information architecture

### Performance Tips
- Use containers to organize content logically
- Avoid excessive nesting of layout components
- Consider using tabs for content that doesn't need to be visible simultaneously
- Use expanders to reduce initial page load visual complexity

## Next Steps
- Practice combining different layout components
- Experiment with responsive design patterns
- Learn about form handling and validation
- Explore advanced styling and theming options
