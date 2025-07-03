# Week 1.3: Simple Projects

## Project 1: Personal Portfolio Page

### Basic Portfolio Structure
```python
import streamlit as st
from datetime import date
import base64

# Page configuration
st.set_page_config(
    page_title="John Doe - Portfolio",
    page_icon="👨‍💻",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 0;
}
.sub-header {
    font-size: 1.2rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.8rem;
    color: #1e40af;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}
.skill-tag {
    background-color: #eff6ff;
    color: #1e40af;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    margin: 0.25rem;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

def create_portfolio():
    # Header Section
    st.markdown('<h1 class="main-header">John Doe</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Full Stack Developer & Data Scientist</p>', unsafe_allow_html=True)
    
    # Profile image placeholder (you can add actual image)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("https://via.placeholder.com/200x200/1e40af/ffffff?text=JD", width=200)
    
    # Contact Information
    st.markdown('<h2 class="section-header">📞 Contact Information</h2>', unsafe_allow_html=True)
    
    contact_col1, contact_col2 = st.columns(2)
    with contact_col1:
        st.write("📧 **Email:** john.doe@example.com")
        st.write("📱 **Phone:** +1 (555) 123-4567")
    with contact_col2:
        st.write("🌐 **LinkedIn:** [linkedin.com/in/johndoe](https://linkedin.com/in/johndoe)")
        st.write("💻 **GitHub:** [github.com/johndoe](https://github.com/johndoe)")
    
    # About Me Section
    st.markdown('<h2 class="section-header">👨‍💻 About Me</h2>', unsafe_allow_html=True)
    st.write("""
    Passionate full-stack developer with 5+ years of experience in building scalable web applications 
    and data-driven solutions. Expertise in Python, JavaScript, and modern frameworks. Strong background 
    in machine learning and data analysis.
    """)
    
    # Skills Section
    st.markdown('<h2 class="section-header">🛠️ Technical Skills</h2>', unsafe_allow_html=True)
    
    skills_data = {
        "Programming Languages": ["Python", "JavaScript", "TypeScript", "Java", "SQL"],
        "Web Frameworks": ["React", "Node.js", "Express", "Django", "Flask"],
        "Data Science": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch"],
        "Databases": ["PostgreSQL", "MongoDB", "Redis", "MySQL"],
        "Tools & Technologies": ["Docker", "AWS", "Git", "Jenkins", "Kubernetes"]
    }
    
    for category, skills in skills_data.items():
        st.write(f"**{category}:**")
        skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
        st.markdown(skill_html, unsafe_allow_html=True)
        st.write("")
    
    # Experience Section
    st.markdown('<h2 class="section-header">💼 Work Experience</h2>', unsafe_allow_html=True)
    
    experiences = [
        {
            "title": "Senior Software Engineer",
            "company": "Tech Innovations Inc.",
            "period": "2022 - Present",
            "description": [
                "Led development of microservices architecture serving 1M+ users",
                "Implemented ML models for recommendation system improving engagement by 25%",
                "Mentored junior developers and conducted code reviews"
            ]
        },
        {
            "title": "Full Stack Developer",
            "company": "StartupXYZ",
            "period": "2020 - 2022",
            "description": [
                "Built responsive web applications using React and Node.js",
                "Designed and implemented RESTful APIs",
                "Optimized database queries reducing response time by 40%"
            ]
        }
    ]
    
    for exp in experiences:
        with st.expander(f"{exp['title']} @ {exp['company']} ({exp['period']})"):
            for desc in exp['description']:
                st.write(f"• {desc}")
    
    # Projects Section
    st.markdown('<h2 class="section-header">🚀 Featured Projects</h2>', unsafe_allow_html=True)
    
    project_col1, project_col2 = st.columns(2)
    
    with project_col1:
        st.subheader("📊 Data Analytics Dashboard")
        st.write("Interactive dashboard for business intelligence")
        st.write("**Tech Stack:** Python, Streamlit, Plotly, PostgreSQL")
        if st.button("View Project 1", key="proj1"):
            st.success("Opening project demo...")
    
    with project_col2:
        st.subheader("🤖 ML Prediction App")
        st.write("Machine learning model for sales forecasting")
        st.write("**Tech Stack:** Python, Scikit-learn, Flask, Docker")
        if st.button("View Project 2", key="proj2"):
            st.success("Opening project demo...")
    
    # Education Section
    st.markdown('<h2 class="section-header">🎓 Education</h2>', unsafe_allow_html=True)
    
    education_col1, education_col2 = st.columns(2)
    with education_col1:
        st.write("**Master of Science in Computer Science**")
        st.write("University of Technology • 2018-2020")
    with education_col2:
        st.write("**Bachelor of Science in Software Engineering**")
        st.write("State University • 2014-2018")

def create_interactive_features():
    st.markdown('<h2 class="section-header">💬 Get in Touch</h2>', unsafe_allow_html=True)
    
    # Contact form
    with st.form("contact_form"):
        st.write("Send me a message:")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name*")
            email = st.text_input("Your Email*")
        with col2:
            subject = st.text_input("Subject*")
            company = st.text_input("Company (Optional)")
        
        message = st.text_area("Message*", height=100)
        
        submitted = st.form_submit_button("Send Message", type="primary")
        
        if submitted:
            if name and email and subject and message:
                st.success("Thank you for your message! I'll get back to you soon.")
                # In a real app, you would send the email here
            else:
                st.error("Please fill in all required fields.")

# Main app
create_portfolio()
create_interactive_features()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b;'>Built with ❤️ using Streamlit</div>", 
    unsafe_allow_html=True
)
```

## Project 2: Simple Calculator

### Enhanced Calculator with History
```python
import streamlit as st
import math
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart Calculator",
    page_icon="🧮",
    layout="wide"
)

# Initialize session state for calculation history
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(operation, result):
    """Add calculation to history"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({
        'time': timestamp,
        'operation': operation,
        'result': result
    })

def create_basic_calculator():
    st.title("🧮 Smart Calculator")
    
    # Calculator tabs
    tab1, tab2, tab3 = st.tabs(["Basic Calculator", "Scientific", "History"])
    
    with tab1:
        st.subheader("Basic Operations")
        
        # Input section
        col1, col2 = st.columns(2)
        with col1:
            num1 = st.number_input("First number:", value=0.0, format="%.6f")
        with col2:
            num2 = st.number_input("Second number:", value=0.0, format="%.6f")
        
        # Operation buttons
        st.write("Choose operation:")
        op_col1, op_col2, op_col3, op_col4 = st.columns(4)
        
        with op_col1:
            if st.button("➕ Add", use_container_width=True):
                result = num1 + num2
                operation = f"{num1} + {num2}"
                st.success(f"Result: {result}")
                add_to_history(operation, result)
        
        with op_col2:
            if st.button("➖ Subtract", use_container_width=True):
                result = num1 - num2
                operation = f"{num1} - {num2}"
                st.success(f"Result: {result}")
                add_to_history(operation, result)
        
        with op_col3:
            if st.button("✖️ Multiply", use_container_width=True):
                result = num1 * num2
                operation = f"{num1} × {num2}"
                st.success(f"Result: {result}")
                add_to_history(operation, result)
        
        with op_col4:
            if st.button("➗ Divide", use_container_width=True):
                if num2 != 0:
                    result = num1 / num2
                    operation = f"{num1} ÷ {num2}"
                    st.success(f"Result: {result}")
                    add_to_history(operation, result)
                else:
                    st.error("Cannot divide by zero!")
    
    with tab2:
        st.subheader("Scientific Operations")
        
        # Single number operations
        number = st.number_input("Enter number:", value=0.0, key="sci_num")
        
        sci_col1, sci_col2, sci_col3 = st.columns(3)
        
        with sci_col1:
            if st.button("√ Square Root"):
                if number >= 0:
                    result = math.sqrt(number)
                    operation = f"√{number}"
                    st.success(f"Result: {result}")
                    add_to_history(operation, result)
                else:
                    st.error("Cannot calculate square root of negative number!")
            
            if st.button("x² Square"):
                result = number ** 2
                operation = f"{number}²"
                st.success(f"Result: {result}")
                add_to_history(operation, result)
        
        with sci_col2:
            if st.button("sin(x)"):
                result = math.sin(math.radians(number))
                operation = f"sin({number}°)"
                st.success(f"Result: {result:.6f}")
                add_to_history(operation, result)
            
            if st.button("cos(x)"):
                result = math.cos(math.radians(number))
                operation = f"cos({number}°)"
                st.success(f"Result: {result:.6f}")
                add_to_history(operation, result)
        
        with sci_col3:
            if st.button("log(x)"):
                if number > 0:
                    result = math.log10(number)
                    operation = f"log({number})"
                    st.success(f"Result: {result:.6f}")
                    add_to_history(operation, result)
                else:
                    st.error("Logarithm undefined for non-positive numbers!")
            
            if st.button("ln(x)"):
                if number > 0:
                    result = math.log(number)
                    operation = f"ln({number})"
                    st.success(f"Result: {result:.6f}")
                    add_to_history(operation, result)
                else:
                    st.error("Natural logarithm undefined for non-positive numbers!")
        
        # Power operation
        st.write("---")
        power_col1, power_col2 = st.columns(2)
        with power_col1:
            base = st.number_input("Base:", value=2.0, key="base")
        with power_col2:
            exponent = st.number_input("Exponent:", value=3.0, key="exp")
        
        if st.button("Calculate x^y"):
            result = base ** exponent
            operation = f"{base}^{exponent}"
            st.success(f"Result: {result}")
            add_to_history(operation, result)
    
    with tab3:
        st.subheader("Calculation History")
        
        if st.session_state.history:
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
            
            # Display history
            for i, calc in enumerate(reversed(st.session_state.history[-10:])):  # Show last 10
                with st.container():
                    col1, col2, col3 = st.columns([2, 3, 1])
                    with col1:
                        st.write(calc['time'])
                    with col2:
                        st.write(f"{calc['operation']} = {calc['result']}")
                    with col3:
                        if st.button("📋", key=f"copy_{i}", help="Copy result"):
                            st.success(f"Copied: {calc['result']}")
        else:
            st.info("No calculations yet. Start calculating to see history!")

# Run the calculator
create_basic_calculator()
```

## Project 3: Basic Data Display App

### Data Explorer Application
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def generate_sample_data():
    """Generate sample dataset for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate sample data
    data = {
        'Date': dates,
        'Sales': np.random.normal(1000, 200, len(dates)),
        'Customers': np.random.poisson(50, len(dates)),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], len(dates)),
        'Temperature': np.random.normal(20, 10, len(dates)),
        'Satisfaction': np.random.uniform(1, 5, len(dates))
    }
    
    df = pd.DataFrame(data)
    
    # Add some seasonality to sales
    df['Sales'] = df['Sales'] + 200 * np.sin(2 * np.pi * df.index / 365.25) + \
                  100 * np.sin(4 * np.pi * df.index / 365.25)
    
    # Ensure positive values
    df['Sales'] = np.abs(df['Sales'])
    df['Customers'] = np.abs(df['Customers'])
    
    return df

def create_data_explorer():
    st.title("📊 Data Explorer Dashboard")
    st.markdown("Explore and analyze your data with interactive visualizations")
    
    # Load data
    df = generate_sample_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Data Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions:",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    # Product filter
    products = st.sidebar.multiselect(
        "Select Products:",
        options=df['Product'].unique(),
        default=df['Product'].unique()
    )
    
    # Apply filters
    if len(date_range) == 2:
        mask = (df['Date'] >= pd.to_datetime(date_range[0])) & \
               (df['Date'] <= pd.to_datetime(date_range[1])) & \
               (df['Region'].isin(regions)) & \
               (df['Product'].isin(products))
        filtered_df = df[mask]
    else:
        filtered_df = df[df['Region'].isin(regions) & df['Product'].isin(products)]
    
    # Main dashboard
    if not filtered_df.empty:
        # Key metrics
        st.subheader("📈 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sales = filtered_df['Sales'].mean()
            st.metric(
                "Average Sales",
                f"${avg_sales:,.0f}",
                delta=f"{(avg_sales - df['Sales'].mean()):.0f}"
            )
        
        with col2:
            total_customers = filtered_df['Customers'].sum()
            st.metric(
                "Total Customers",
                f"{total_customers:,}",
                delta=f"{total_customers - df['Customers'].sum()}"
            )
        
        with col3:
            avg_satisfaction = filtered_df['Satisfaction'].mean()
            st.metric(
                "Avg Satisfaction",
                f"{avg_satisfaction:.2f}/5",
                delta=f"{(avg_satisfaction - df['Satisfaction'].mean()):.2f}"
            )
        
        with col4:
            data_points = len(filtered_df)
            st.metric(
                "Data Points",
                f"{data_points:,}",
                delta=f"{data_points - len(df)}"
            )
        
        # Charts section
        st.subheader("📊 Visualizations")
        
        # Chart selection
        chart_type = st.selectbox(
            "Choose Chart Type:",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot"]
        )
        
        if chart_type == "Line Chart":
            # Time series chart
            fig = px.line(
                filtered_df, 
                x='Date', 
                y='Sales', 
                color='Region',
                title="Sales Over Time by Region"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Bar Chart":
            # Sales by product
            product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()
            fig = px.bar(
                product_sales,
                x='Product',
                y='Sales',
                title="Total Sales by Product"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Scatter Plot":
            # Sales vs Customers
            fig = px.scatter(
                filtered_df,
                x='Customers',
                y='Sales',
                color='Region',
                size='Satisfaction',
                title="Sales vs Customers (Size = Satisfaction)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Histogram":
            # Sales distribution
            fig = px.histogram(
                filtered_df,
                x='Sales',
                nbins=30,
                title="Sales Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Box Plot":
            # Sales by region
            fig = px.box(
                filtered_df,
                x='Region',
                y='Sales',
                title="Sales Distribution by Region"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Raw Data")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Show all data")
            n_rows = len(filtered_df) if show_all else st.slider("Number of rows:", 5, 100, 10)
        
        with col2:
            columns_to_show = st.multiselect(
                "Select columns:",
                options=filtered_df.columns.tolist(),
                default=filtered_df.columns.tolist()
            )
        
        # Display filtered data
        if columns_to_show:
            display_df = filtered_df[columns_to_show].head(n_rows)
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Summary statistics
        with st.expander("📊 Summary Statistics"):
            st.write("Numerical columns summary:")
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            st.dataframe(filtered_df[numeric_cols].describe())
    
    else:
        st.warning("No data matches the selected filters. Please adjust your selection.")

# Run the data explorer
create_data_explorer()
```

## Project 4: Text Formatter Tool

### Advanced Text Processing Tool
```python
import streamlit as st
import re
import string
from collections import Counter
import unicodedata

# Page configuration
st.set_page_config(
    page_title="Text Formatter Pro",
    page_icon="📝",
    layout="wide"
)

def create_text_formatter():
    st.title("📝 Text Formatter Pro")
    st.markdown("Advanced text processing and formatting tool")
    
    # Input section
    st.subheader("📄 Input Text")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Type/Paste Text", "Upload File"],
        horizontal=True
    )
    
    user_text = ""
    
    if input_method == "Type/Paste Text":
        user_text = st.text_area(
            "Enter your text:",
            height=200,
            placeholder="Type or paste your text here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt', 'md', 'py', 'js', 'html', 'css']
        )
        if uploaded_file is not None:
            user_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded content:", user_text, height=100, disabled=True)
    
    if user_text:
        # Processing options
        st.subheader("⚙️ Processing Options")
        
        # Layout with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Basic Formatting", "Advanced", "Analysis", "Export"])
        
        with tab1:
            st.write("**Text Transformations:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                uppercase = st.checkbox("UPPERCASE")
                lowercase = st.checkbox("lowercase")
                title_case = st.checkbox("Title Case")
                sentence_case = st.checkbox("Sentence case")
            
            with col2:
                remove_extra_spaces = st.checkbox("Remove extra spaces")
                remove_empty_lines = st.checkbox("Remove empty lines")
                trim_lines = st.checkbox("Trim line spaces")
                add_line_numbers = st.checkbox("Add line numbers")
            
            with col3:
                reverse_text = st.checkbox("Reverse text")
                sort_lines = st.checkbox("Sort lines alphabetically")
                remove_duplicates = st.checkbox("Remove duplicate lines")
                shuffle_lines = st.checkbox("Shuffle lines")
        
        with tab2:
            st.write("**Advanced Processing:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                remove_punctuation = st.checkbox("Remove punctuation")
                remove_numbers = st.checkbox("Remove numbers")
                remove_special_chars = st.checkbox("Remove special characters")
                normalize_unicode = st.checkbox("Normalize Unicode")
            
            with col2:
                # Find and replace
                st.write("**Find & Replace:**")
                find_text = st.text_input("Find:")
                replace_text = st.text_input("Replace with:")
                case_sensitive = st.checkbox("Case sensitive")
                use_regex = st.checkbox("Use regular expressions")
        
        with tab3:
            # Text analysis
            st.write("**Text Analysis:**")
            
            # Basic statistics
            words = user_text.split()
            sentences = re.split(r'[.!?]+', user_text)
            paragraphs = user_text.split('\n\n')
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Characters", len(user_text))
                st.metric("Characters (no spaces)", len(user_text.replace(' ', '')))
            
            with stat_col2:
                st.metric("Words", len(words))
                st.metric("Unique Words", len(set(word.lower() for word in words if word)))
            
            with stat_col3:
                st.metric("Sentences", len([s for s in sentences if s.strip()]))
                st.metric("Paragraphs", len([p for p in paragraphs if p.strip()]))
            
            with stat_col4:
                avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
                st.metric("Avg Word Length", f"{avg_word_length:.1f}")
                lines = user_text.split('\n')
                st.metric("Lines", len(lines))
            
            # Word frequency
            if st.checkbox("Show word frequency"):
                # Clean words for frequency analysis
                clean_words = [
                    word.lower().strip(string.punctuation) 
                    for word in words 
                    if word.strip(string.punctuation)
                ]
                word_freq = Counter(clean_words)
                
                st.write("**Most common words:**")
                top_words = word_freq.most_common(10)
                
                freq_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                st.dataframe(freq_df, use_container_width=True)
        
        with tab4:
            # Export options
            st.write("**Export Options:**")
            
            export_format = st.selectbox(
                "Export format:",
                ["Plain Text", "Markdown", "HTML", "JSON"]
            )
            
            include_stats = st.checkbox("Include statistics")
        
        # Apply transformations
        processed_text = user_text
        
        # Basic transformations
        if uppercase:
            processed_text = processed_text.upper()
        elif lowercase:
            processed_text = processed_text.lower()
        elif title_case:
            processed_text = processed_text.title()
        elif sentence_case:
            processed_text = '. '.join(
                sentence.strip().capitalize() 
                for sentence in processed_text.split('.') 
                if sentence.strip()
            )
        
        if remove_extra_spaces:
            processed_text = re.sub(r'\s+', ' ', processed_text)
        
        if remove_empty_lines:
            lines = processed_text.split('\n')
            processed_text = '\n'.join(line for line in lines if line.strip())
        
        if trim_lines:
            lines = processed_text.split('\n')
            processed_text = '\n'.join(line.strip() for line in lines)
        
        if remove_punctuation:
            processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
        
        if remove_numbers:
            processed_text = re.sub(r'\d+', '', processed_text)
        
        if remove_special_chars:
            processed_text = re.sub(r'[^a-zA-Z0-9\s]', '', processed_text)
        
        if normalize_unicode:
            processed_text = unicodedata.normalize('NFKD', processed_text)
        
        if reverse_text:
            processed_text = processed_text[::-1]
        
        if sort_lines:
            lines = processed_text.split('\n')
            lines.sort()
            processed_text = '\n'.join(lines)
        
        if remove_duplicates:
            lines = processed_text.split('\n')
            seen = set()
            unique_lines = []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
            processed_text = '\n'.join(unique_lines)
        
        if shuffle_lines:
            import random
            lines = processed_text.split('\n')
            random.shuffle(lines)
            processed_text = '\n'.join(lines)
        
        if add_line_numbers:
            lines = processed_text.split('\n')
            numbered_lines = [f"{i+1:3d}: {line}" for i, line in enumerate(lines)]
            processed_text = '\n'.join(numbered_lines)
        
        # Find and replace
        if find_text:
            if use_regex:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    processed_text = re.sub(find_text, replace_text, processed_text, flags=flags)
                except re.error as e:
                    st.error(f"Regex error: {e}")
            else:
                if case_sensitive:
                    processed_text = processed_text.replace(find_text, replace_text)
                else:
                    # Case insensitive replace
                    pattern = re.compile(re.escape(find_text), re.IGNORECASE)
                    processed_text = pattern.sub(replace_text, processed_text)
        
        # Display results
        st.subheader("✨ Processed Text")
        
        # Show before/after comparison
        if st.checkbox("Show side-by-side comparison"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original:**")
                st.text_area("", user_text, height=300, disabled=True, key="original")
            with col2:
                st.write("**Processed:**")
                st.text_area("", processed_text, height=300, disabled=True, key="processed")
        else:
            st.text_area(
                "Result:",
                processed_text,
                height=300,
                key="result"
            )
        
        # Download processed text
        st.download_button(
            label="📥 Download Processed Text",
            data=processed_text,
            file_name=f"processed_text_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Import pandas for timestamp
import pandas as pd

# Run the text formatter
create_text_formatter()
```

## Key Learning Points

### Project Development Best Practices
1. **Modular Code Structure**: Break functionality into functions
2. **User Experience**: Clear navigation and intuitive interface
3. **Error Handling**: Graceful handling of edge cases
4. **Performance**: Use caching for expensive operations
5. **Responsive Design**: Adapt to different screen sizes

### Common Patterns Used
- **Multi-tab interfaces** for organizing content
- **Session state** for maintaining data across reruns
- **Form validation** with user feedback
- **Progressive enhancement** with optional features
- **Data export capabilities** for user convenience

### Technical Skills Practiced
- Complex layout design with columns and containers
- File upload and processing
- Data visualization with Plotly
- Text processing and regex operations
- User input validation and sanitization

## Next Steps
- Add more interactive features to existing projects
- Implement data persistence (database integration)
- Create multi-page applications
- Add user authentication
- Deploy projects to Streamlit Cloud
