# Week 2.3: Projects

## Project 1: Multi-Page Form Application

### Customer Registration System
```python
import streamlit as st
from datetime import datetime, date
import json

# Page configuration
st.set_page_config(
    page_title="Customer Registration System",
    page_icon="📝",
    layout="wide"
)

# Initialize session state for form data
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

def save_form_data(page_data):
    """Save current page data to session state"""
    st.session_state.form_data.update(page_data)

def create_navigation():
    """Create page navigation"""
    st.sidebar.title("🧭 Registration Steps")
    
    pages = [
        "1️⃣ Personal Info",
        "2️⃣ Contact Details", 
        "3️⃣ Preferences",
        "4️⃣ Review & Submit"
    ]
    
    # Show progress
    progress = (st.session_state.current_page - 1) / (len(pages) - 1)
    st.sidebar.progress(progress)
    
    # Page navigation
    for i, page_name in enumerate(pages, 1):
        if st.sidebar.button(page_name, key=f"nav_{i}"):
            st.session_state.current_page = i

def page_personal_info():
    """Page 1: Personal Information"""
    st.header("👤 Personal Information")
    
    with st.form("personal_info"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name*", 
                value=st.session_state.form_data.get('first_name', ''))
            last_name = st.text_input("Last Name*",
                value=st.session_state.form_data.get('last_name', ''))
            birth_date = st.date_input("Date of Birth*",
                value=st.session_state.form_data.get('birth_date', date(1990, 1, 1)),
                max_value=date.today())
        
        with col2:
            gender = st.selectbox("Gender",
                ["Select", "Male", "Female", "Other"],
                index=["Select", "Male", "Female", "Other"].index(
                    st.session_state.form_data.get('gender', 'Select')))
            nationality = st.text_input("Nationality",
                value=st.session_state.form_data.get('nationality', ''))
            id_number = st.text_input("ID Number",
                value=st.session_state.form_data.get('id_number', ''))
        
        submitted = st.form_submit_button("Next Step →", type="primary")
        
        if submitted:
            if first_name and last_name and birth_date and gender != "Select":
                save_form_data({
                    'first_name': first_name,
                    'last_name': last_name,
                    'birth_date': birth_date,
                    'gender': gender,
                    'nationality': nationality,
                    'id_number': id_number
                })
                st.session_state.current_page = 2
                st.rerun()
            else:
                st.error("Please fill in all required fields!")

def page_contact_details():
    """Page 2: Contact Details"""
    st.header("📞 Contact Information")
    
    with st.form("contact_info"):
        # Address section
        st.subheader("🏠 Address")
        address = st.text_area("Street Address*",
            value=st.session_state.form_data.get('address', ''))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.text_input("City*",
                value=st.session_state.form_data.get('city', ''))
        with col2:
            state = st.text_input("State/Province",
                value=st.session_state.form_data.get('state', ''))
        with col3:
            postal_code = st.text_input("Postal Code*",
                value=st.session_state.form_data.get('postal_code', ''))
        
        country = st.selectbox("Country*",
            ["Select", "USA", "Canada", "UK", "Australia", "Other"],
            index=["Select", "USA", "Canada", "UK", "Australia", "Other"].index(
                st.session_state.form_data.get('country', 'Select')))
        
        # Contact methods
        st.subheader("📱 Contact Methods")
        col1, col2 = st.columns(2)
        with col1:
            email = st.text_input("Email Address*",
                value=st.session_state.form_data.get('email', ''))
            phone = st.text_input("Phone Number*",
                value=st.session_state.form_data.get('phone', ''))
        with col2:
            alt_email = st.text_input("Alternative Email",
                value=st.session_state.form_data.get('alt_email', ''))
            alt_phone = st.text_input("Alternative Phone",
                value=st.session_state.form_data.get('alt_phone', ''))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("← Previous"):
                st.session_state.current_page = 1
                st.rerun()
        with col2:
            if st.form_submit_button("Next Step →", type="primary"):
                if address and city and postal_code and country != "Select" and email and phone:
                    save_form_data({
                        'address': address, 'city': city, 'state': state,
                        'postal_code': postal_code, 'country': country,
                        'email': email, 'phone': phone,
                        'alt_email': alt_email, 'alt_phone': alt_phone
                    })
                    st.session_state.current_page = 3
                    st.rerun()
                else:
                    st.error("Please fill in all required fields!")

def page_preferences():
    """Page 3: Preferences"""
    st.header("⚙️ Preferences & Settings")
    
    with st.form("preferences"):
        # Communication preferences
        st.subheader("📧 Communication Preferences")
        newsletter = st.checkbox("Subscribe to newsletter",
            value=st.session_state.form_data.get('newsletter', False))
        marketing = st.checkbox("Receive marketing emails",
            value=st.session_state.form_data.get('marketing', False))
        sms_updates = st.checkbox("SMS updates",
            value=st.session_state.form_data.get('sms_updates', False))
        
        # Interests
        st.subheader("🎯 Interests")
        interests = st.multiselect("Select your interests:",
            ["Technology", "Sports", "Music", "Travel", "Food", "Art", "Business"],
            default=st.session_state.form_data.get('interests', []))
        
        # Account settings
        st.subheader("🔒 Account Settings")
        col1, col2 = st.columns(2)
        with col1:
            account_type = st.selectbox("Account Type",
                ["Basic", "Premium", "Enterprise"],
                index=["Basic", "Premium", "Enterprise"].index(
                    st.session_state.form_data.get('account_type', 'Basic')))
        with col2:
            language = st.selectbox("Preferred Language",
                ["English", "Spanish", "French", "German"],
                index=["English", "Spanish", "French", "German"].index(
                    st.session_state.form_data.get('language', 'English')))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("← Previous"):
                st.session_state.current_page = 2
                st.rerun()
        with col2:
            if st.form_submit_button("Review →", type="primary"):
                save_form_data({
                    'newsletter': newsletter, 'marketing': marketing,
                    'sms_updates': sms_updates, 'interests': interests,
                    'account_type': account_type, 'language': language
                })
                st.session_state.current_page = 4
                st.rerun()

def page_review_submit():
    """Page 4: Review and Submit"""
    st.header("📋 Review Your Information")
    
    # Display all collected data
    data = st.session_state.form_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        st.write(f"**Name:** {data.get('first_name', '')} {data.get('last_name', '')}")
        st.write(f"**Date of Birth:** {data.get('birth_date', '')}")
        st.write(f"**Gender:** {data.get('gender', '')}")
        st.write(f"**Nationality:** {data.get('nationality', 'Not provided')}")
        
        st.subheader("📞 Contact Information")
        st.write(f"**Email:** {data.get('email', '')}")
        st.write(f"**Phone:** {data.get('phone', '')}")
        st.write(f"**Address:** {data.get('address', '')}, {data.get('city', '')}")
        st.write(f"**Country:** {data.get('country', '')}")
    
    with col2:
        st.subheader("⚙️ Preferences")
        st.write(f"**Account Type:** {data.get('account_type', '')}")
        st.write(f"**Language:** {data.get('language', '')}")
        st.write(f"**Newsletter:** {'Yes' if data.get('newsletter') else 'No'}")
        st.write(f"**Marketing:** {'Yes' if data.get('marketing') else 'No'}")
        
        interests = data.get('interests', [])
        if interests:
            st.write(f"**Interests:** {', '.join(interests)}")
    
    # Terms and conditions
    st.subheader("📄 Terms and Conditions")
    agree_terms = st.checkbox("I agree to the Terms and Conditions*")
    agree_privacy = st.checkbox("I agree to the Privacy Policy*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Edit Information"):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        if st.button("📥 Download Data"):
            # Create downloadable JSON
            json_data = json.dumps(data, indent=2, default=str)
            st.download_button(
                "💾 Download as JSON",
                json_data,
                "registration_data.json",
                "application/json"
            )
    
    with col3:
        if st.button("✅ Submit Registration", type="primary"):
            if agree_terms and agree_privacy:
                st.success("🎉 Registration submitted successfully!")
                st.balloons()
                # Clear form data
                st.session_state.form_data = {}
                st.session_state.current_page = 1
            else:
                st.error("Please agree to the terms and conditions!")

# Main application
def main():
    st.title("📝 Customer Registration System")
    
    create_navigation()
    
    # Route to appropriate page
    if st.session_state.current_page == 1:
        page_personal_info()
    elif st.session_state.current_page == 2:
        page_contact_details()
    elif st.session_state.current_page == 3:
        page_preferences()
    elif st.session_state.current_page == 4:
        page_review_submit()

if __name__ == "__main__":
    main()
```

## Project 2: Data Filtering Dashboard

### Sales Analytics Dashboard
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_sample_data():
    """Generate comprehensive sample sales data"""
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate data
    n_records = len(dates) * 10  # 10 sales per day on average
    
    data = {
        'Date': np.random.choice(dates, n_records),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'], n_records),
        'Category': np.random.choice(['Electronics', 'Accessories', 'Computing'], n_records),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'Salesperson': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], n_records),
        'Customer_Type': np.random.choice(['New', 'Returning', 'VIP'], n_records),
        'Sales_Amount': np.random.normal(500, 200, n_records),
        'Quantity': np.random.poisson(2, n_records) + 1,
        'Discount': np.random.uniform(0, 0.3, n_records),
    }
    
    df = pd.DataFrame(data)
    df['Sales_Amount'] = np.abs(df['Sales_Amount'])  # Ensure positive values
    df['Net_Amount'] = df['Sales_Amount'] * (1 - df['Discount'])
    df['Month'] = df['Date'].dt.month_name()
    df['Quarter'] = df['Date'].dt.quarter
    
    return df

def create_filters(df):
    """Create sidebar filters"""
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Product filters
    products = st.sidebar.multiselect(
        "Products:",
        options=sorted(df['Product'].unique()),
        default=sorted(df['Product'].unique())
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Regions:",
        options=sorted(df['Region'].unique()),
        default=sorted(df['Region'].unique())
    )
    
    # Salesperson filter
    salespeople = st.sidebar.multiselect(
        "Salespeople:",
        options=sorted(df['Salesperson'].unique()),
        default=sorted(df['Salesperson'].unique())
    )
    
    # Customer type filter
    customer_types = st.sidebar.multiselect(
        "Customer Types:",
        options=sorted(df['Customer_Type'].unique()),
        default=sorted(df['Customer_Type'].unique())
    )
    
    # Sales amount range
    min_amount, max_amount = float(df['Sales_Amount'].min()), float(df['Sales_Amount'].max())
    amount_range = st.sidebar.slider(
        "Sales Amount Range:",
        min_value=min_amount,
        max_value=max_amount,
        value=(min_amount, max_amount),
        format="$%.0f"
    )
    
    return {
        'date_range': date_range,
        'products': products,
        'regions': regions,
        'salespeople': salespeople,
        'customer_types': customer_types,
        'amount_range': amount_range
    }

def apply_filters(df, filters):
    """Apply all filters to the dataframe"""
    filtered_df = df.copy()
    
    # Date filter
    if len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) &
            (filtered_df['Date'].dt.date <= end_date)
        ]
    
    # Other filters
    if filters['products']:
        filtered_df = filtered_df[filtered_df['Product'].isin(filters['products'])]
    
    if filters['regions']:
        filtered_df = filtered_df[filtered_df['Region'].isin(filters['regions'])]
    
    if filters['salespeople']:
        filtered_df = filtered_df[filtered_df['Salesperson'].isin(filters['salespeople'])]
    
    if filters['customer_types']:
        filtered_df = filtered_df[filtered_df['Customer_Type'].isin(filters['customer_types'])]
    
    # Amount range filter
    filtered_df = filtered_df[
        (filtered_df['Sales_Amount'] >= filters['amount_range'][0]) &
        (filtered_df['Sales_Amount'] <= filters['amount_range'][1])
    ]
    
    return filtered_df

def create_kpi_metrics(df):
    """Create KPI metrics display"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sales = df['Net_Amount'].sum()
    total_orders = len(df)
    avg_order_value = df['Net_Amount'].mean()
    total_quantity = df['Quantity'].sum()
    avg_discount = df['Discount'].mean()
    
    with col1:
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col3:
        st.metric("Avg Order Value", f"${avg_order_value:.0f}")
    
    with col4:
        st.metric("Total Quantity", f"{total_quantity:,}")
    
    with col5:
        st.metric("Avg Discount", f"{avg_discount:.1%}")

def create_visualizations(df):
    """Create dashboard visualizations"""
    # Sales over time
    st.subheader("📈 Sales Trends")
    
    time_grouping = st.selectbox("Group by:", ["Day", "Week", "Month"])
    
    if time_grouping == "Day":
        time_df = df.groupby(df['Date'].dt.date)['Net_Amount'].sum().reset_index()
        time_df.columns = ['Date', 'Sales']
    elif time_grouping == "Week":
        time_df = df.groupby(df['Date'].dt.to_period('W'))['Net_Amount'].sum().reset_index()
        time_df['Date'] = time_df['Date'].dt.start_time
        time_df.columns = ['Date', 'Sales']
    else:  # Month
        time_df = df.groupby(df['Date'].dt.to_period('M'))['Net_Amount'].sum().reset_index()
        time_df['Date'] = time_df['Date'].dt.start_time
        time_df.columns = ['Date', 'Sales']
    
    fig_time = px.line(time_df, x='Date', y='Sales', title=f"Sales by {time_grouping}")
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by product
        product_sales = df.groupby('Product')['Net_Amount'].sum().sort_values(ascending=True)
        fig_product = px.bar(x=product_sales.values, y=product_sales.index, 
                           orientation='h', title="Sales by Product")
        st.plotly_chart(fig_product, use_container_width=True)
        
        # Sales by region
        region_sales = df.groupby('Region')['Net_Amount'].sum()
        fig_region = px.pie(values=region_sales.values, names=region_sales.index,
                          title="Sales Distribution by Region")
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        # Sales by salesperson
        salesperson_sales = df.groupby('Salesperson')['Net_Amount'].sum().sort_values(ascending=False)
        fig_salesperson = px.bar(x=salesperson_sales.index, y=salesperson_sales.values,
                                title="Sales by Salesperson")
        st.plotly_chart(fig_salesperson, use_container_width=True)
        
        # Customer type analysis
        customer_analysis = df.groupby('Customer_Type').agg({
            'Net_Amount': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        fig_customer = px.scatter(customer_analysis, x='Quantity', y='Net_Amount',
                                size='Net_Amount', color='Customer_Type',
                                title="Customer Type Analysis")
        st.plotly_chart(fig_customer, use_container_width=True)

def main():
    st.title("📊 Sales Analytics Dashboard")
    st.markdown("Interactive dashboard for sales data analysis and filtering")
    
    # Load data
    df = load_sample_data()
    
    # Create filters
    filters = create_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Show filtered data info
    st.info(f"Showing {len(filtered_df):,} records out of {len(df):,} total records")
    
    if not filtered_df.empty:
        # KPI metrics
        create_kpi_metrics(filtered_df)
        st.markdown("---")
        
        # Visualizations
        create_visualizations(filtered_df)
        
        # Data table
        if st.checkbox("Show detailed data"):
            st.subheader("📋 Detailed Sales Data")
            
            # Additional table filters
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by:", 
                    ['Date', 'Net_Amount', 'Product', 'Salesperson'])
            with col2:
                sort_order = st.selectbox("Sort order:", ['Descending', 'Ascending'])
            
            # Sort data
            ascending = sort_order == 'Ascending'
            display_df = filtered_df.sort_values(sort_by, ascending=ascending)
            
            # Show top N records
            n_records = st.slider("Number of records to display:", 10, 100, 50)
            st.dataframe(display_df.head(n_records), use_container_width=True)
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download filtered data",
                csv,
                "sales_data_filtered.csv",
                "text/csv"
            )
    else:
        st.warning("No data matches the current filters. Please adjust your selection.")

if __name__ == "__main__":
    main()
```

## Project 3: File Processing Tool

### Document Analyzer and Processor
```python
import streamlit as st
import pandas as pd
import json
import io
import zipfile
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="Document Processor",
    page_icon="📄",
    layout="wide"
)

def process_text_file(file):
    """Process text files"""
    content = str(file.read(), "utf-8")
    
    # Basic text analysis
    words = content.split()
    lines = content.split('\n')
    
    analysis = {
        'total_characters': len(content),
        'total_characters_no_spaces': len(content.replace(' ', '')),
        'total_words': len(words),
        'total_lines': len(lines),
        'average_words_per_line': len(words) / len(lines) if lines else 0,
        'unique_words': len(set(word.lower().strip('.,!?";') for word in words)),
    }
    
    return content, analysis

def process_csv_file(file):
    """Process CSV files"""
    df = pd.read_csv(file)
    
    analysis = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().sum(),
        'data_types': dict(df.dtypes),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }
    
    return df, analysis

def process_json_file(file):
    """Process JSON files"""
    content = json.load(file)
    
    def count_keys(obj, level=0):
        count = 0
        max_depth = level
        if isinstance(obj, dict):
            count += len(obj)
            for value in obj.values():
                sub_count, sub_depth = count_keys(value, level + 1)
                count += sub_count
                max_depth = max(max_depth, sub_depth)
        elif isinstance(obj, list):
            for item in obj:
                sub_count, sub_depth = count_keys(item, level + 1)
                count += sub_count
                max_depth = max(max_depth, sub_depth)
        return count, max_depth
    
    total_keys, max_depth = count_keys(content)
    
    analysis = {
        'type': type(content).__name__,
        'total_keys': total_keys,
        'max_depth': max_depth,
        'size_estimate': len(str(content)),
    }
    
    return content, analysis

def create_file_processor():
    """Main file processing interface"""
    st.title("📄 Document Processor & Analyzer")
    st.markdown("Upload and process various file types with detailed analysis")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload files for processing:",
        type=['txt', 'csv', 'json', 'xlsx', 'pdf'],
        accept_multiple_files=True,
        help="Supported formats: TXT, CSV, JSON, Excel, PDF"
    )
    
    if uploaded_files:
        # Processing options
        st.sidebar.header("⚙️ Processing Options")
        
        # Global options
        show_preview = st.sidebar.checkbox("Show file preview", value=True)
        show_analysis = st.sidebar.checkbox("Show detailed analysis", value=True)
        batch_process = st.sidebar.checkbox("Batch process all files")
        
        # Create tabs for different files
        if len(uploaded_files) == 1:
            process_single_file(uploaded_files[0], show_preview, show_analysis)
        else:
            if batch_process:
                process_multiple_files(uploaded_files, show_preview, show_analysis)
            else:
                # Create tabs for individual files
                tab_names = [f"📄 {file.name}" for file in uploaded_files]
                tabs = st.tabs(tab_names)
                
                for tab, file in zip(tabs, uploaded_files):
                    with tab:
                        process_single_file(file, show_preview, show_analysis)

def process_single_file(file, show_preview, show_analysis):
    """Process a single file"""
    st.subheader(f"📄 Processing: {file.name}")
    
    # File information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Size", f"{file.size:,} bytes")
    with col2:
        st.metric("File Type", file.type or "Unknown")
    with col3:
        st.metric("Upload Time", datetime.now().strftime("%H:%M:%S"))
    
    try:
        # Process based on file extension
        file_ext = file.name.lower().split('.')[-1]
        
        if file_ext == 'txt':
            content, analysis = process_text_file(file)
            
            if show_analysis:
                st.subheader("📊 Text Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Characters", f"{analysis['total_characters']:,}")
                    st.metric("Total Words", f"{analysis['total_words']:,}")
                    st.metric("Unique Words", f"{analysis['unique_words']:,}")
                
                with col2:
                    st.metric("Total Lines", f"{analysis['total_lines']:,}")
                    st.metric("Avg Words/Line", f"{analysis['average_words_per_line']:.1f}")
                    st.metric("Characters (no spaces)", f"{analysis['total_characters_no_spaces']:,}")
            
            if show_preview:
                st.subheader("📖 Content Preview")
                st.text_area("File content:", content, height=300)
            
            # Text processing options
            st.subheader("⚙️ Text Processing")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔤 Convert to Uppercase"):
                    processed_content = content.upper()
                    st.text_area("Processed content:", processed_content, height=200)
                
                if st.button("📊 Word Frequency"):
                    words = content.lower().split()
                    word_freq = pd.Series(words).value_counts().head(10)
                    st.bar_chart(word_freq)
            
            with col2:
                if st.button("🔡 Convert to Lowercase"):
                    processed_content = content.lower()
                    st.text_area("Processed content:", processed_content, height=200)
                
                if st.button("📈 Line Length Analysis"):
                    lines = content.split('\n')
                    line_lengths = [len(line) for line in lines]
                    st.line_chart(pd.DataFrame({'Line Length': line_lengths}))
        
        elif file_ext == 'csv':
            df, analysis = process_csv_file(file)
            
            if show_analysis:
                st.subheader("📊 CSV Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", f"{analysis['rows']:,}")
                with col2:
                    st.metric("Columns", f"{analysis['columns']:,}")
                with col3:
                    st.metric("Missing Values", f"{analysis['missing_values']:,}")
                with col4:
                    st.metric("Memory Usage", f"{analysis['memory_usage']:,} bytes")
                
                # Column information
                st.subheader("📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': [str(dtype) for dtype in df.dtypes],
                    'Non-Null Count': [df[col].count() for col in df.columns],
                    'Null Count': [df[col].isnull().sum() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
            
            if show_preview:
                st.subheader("📖 Data Preview")
                preview_rows = st.slider("Rows to preview:", 5, min(50, len(df)), 10)
                st.dataframe(df.head(preview_rows), use_container_width=True)
            
            # Data processing options
            st.subheader("⚙️ Data Processing")
            
            # Column selection for analysis
            numeric_cols = analysis['numeric_columns']
            if numeric_cols:
                selected_col = st.selectbox("Select numeric column for analysis:", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Summary Statistics"):
                        st.write(df[selected_col].describe())
                
                with col2:
                    if st.button("📈 Distribution Plot"):
                        st.bar_chart(df[selected_col].value_counts().head(20))
        
        elif file_ext == 'json':
            content, analysis = process_json_file(file)
            
            if show_analysis:
                st.subheader("📊 JSON Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Data Type", analysis['type'])
                with col2:
                    st.metric("Total Keys", f"{analysis['total_keys']:,}")
                with col3:
                    st.metric("Max Depth", analysis['max_depth'])
            
            if show_preview:
                st.subheader("📖 JSON Content")
                st.json(content)
        
        # Download processed file
        st.subheader("💾 Download Options")
        if st.button("📥 Download Analysis Report"):
            # Create analysis report
            report = {
                'file_name': file.name,
                'file_size': file.size,
                'analysis_time': datetime.now().isoformat(),
                'analysis': analysis if 'analysis' in locals() else {}
            }
            
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                "Download Report",
                report_json,
                f"analysis_report_{file.name}.json",
                "application/json"
            )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

def process_multiple_files(files, show_preview, show_analysis):
    """Batch process multiple files"""
    st.subheader("🔄 Batch Processing Results")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        try:
            file_ext = file.name.lower().split('.')[-1]
            
            if file_ext == 'txt':
                _, analysis = process_text_file(file)
            elif file_ext == 'csv':
                _, analysis = process_csv_file(file)
            elif file_ext == 'json':
                _, analysis = process_json_file(file)
            else:
                analysis = {'error': 'Unsupported file type'}
            
            results.append({
                'file_name': file.name,
                'file_size': file.size,
                'file_type': file_ext,
                'status': 'Success',
                **analysis
            })
        
        except Exception as e:
            results.append({
                'file_name': file.name,
                'file_size': file.size,
                'file_type': file.name.split('.')[-1],
                'status': 'Error',
                'error': str(e)
            })
        
        progress_bar.progress((i + 1) / len(files))
    
    # Display results
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Batch Processing Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Files", len(files))
    with col2:
        success_count = len(results_df[results_df['status'] == 'Success'])
        st.metric("Successful", success_count)
    with col3:
        error_count = len(results_df[results_df['status'] == 'Error'])
        st.metric("Errors", error_count)

if __name__ == "__main__":
    create_file_processor()
```

## Project 4: Interactive Survey App

### Customer Satisfaction Survey
```python
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Interactive Survey App",
    page_icon="📋",
    layout="wide"
)

# Initialize session state
if 'survey_responses' not in st.session_state:
    st.session_state.survey_responses = []
if 'current_survey' not in st.session_state:
    st.session_state.current_survey = {}

def create_survey_form():
    """Create the main survey form"""
    st.title("📋 Customer Satisfaction Survey")
    st.markdown("Help us improve our services by sharing your feedback!")
    
    with st.form("satisfaction_survey"):
        # Basic information
        st.subheader("👤 About You")
        col1, col2 = st.columns(2)
        
        with col1:
            age_group = st.selectbox(
                "Age Group:",
                ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
            )
            
            customer_type = st.radio(
                "Customer Type:",
                ["New Customer", "Returning Customer", "VIP Customer"],
                horizontal=True
            )
        
        with col2:
            location = st.selectbox(
                "Location:",
                ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
            )
            
            how_heard = st.selectbox(
                "How did you hear about us?",
                ["Social Media", "Search Engine", "Friend/Family", "Advertisement", "Other"]
            )
        
        # Service ratings
        st.subheader("⭐ Service Ratings")
        st.markdown("Please rate the following aspects (1 = Very Poor, 5 = Excellent):")
        
        col1, col2 = st.columns(2)
        
        with col1:
            overall_satisfaction = st.select_slider(
                "Overall Satisfaction:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
            
            product_quality = st.select_slider(
                "Product Quality:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
            
            value_for_money = st.select_slider(
                "Value for Money:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
        
        with col2:
            customer_service = st.select_slider(
                "Customer Service:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
            
            delivery_speed = st.select_slider(
                "Delivery Speed:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
            
            website_experience = st.select_slider(
                "Website Experience:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
        
        # Product usage
        st.subheader("🛍️ Product Usage")
        
        products_used = st.multiselect(
            "Which products/services have you used?",
            ["Product A", "Product B", "Product C", "Service X", "Service Y", "Service Z"]
        )
        
        usage_frequency = st.radio(
            "How often do you use our products/services?",
            ["Daily", "Weekly", "Monthly", "Occasionally", "First time"],
            horizontal=True
        )
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommend_likelihood = st.slider(
            "How likely are you to recommend us to others? (0-10)",
            min_value=0,
            max_value=10,
            value=7
        )
        
        improvements = st.multiselect(
            "What areas need improvement?",
            ["Product Quality", "Customer Service", "Pricing", "Delivery", 
             "Website", "Product Range", "Communication", "Support"]
        )
        
        # Open feedback
        st.subheader("💬 Additional Feedback")
        
        best_experience = st.text_area(
            "What was the best part of your experience?",
            height=80
        )
        
        suggestions = st.text_area(
            "Any suggestions for improvement?",
            height=80
        )
        
        additional_comments = st.text_area(
            "Additional comments:",
            height=80
        )
        
        # Contact preferences
        st.subheader("📞 Contact Preferences")
        
        col1, col2 = st.columns(2)
        with col1:
            follow_up = st.checkbox("I'm open to follow-up contact")
            newsletter = st.checkbox("Subscribe to newsletter")
        
        with col2:
            email_updates = st.checkbox("Receive email updates")
            survey_participation = st.checkbox("Participate in future surveys")
        
        # Submit button
        submitted = st.form_submit_button("📤 Submit Survey", type="primary")
        
        if submitted:
            # Collect all response data
            response_data = {
                'timestamp': datetime.now(),
                'age_group': age_group,
                'customer_type': customer_type,
                'location': location,
                'how_heard': how_heard,
                'overall_satisfaction': overall_satisfaction,
                'product_quality': product_quality,
                'value_for_money': value_for_money,
                'customer_service': customer_service,
                'delivery_speed': delivery_speed,
                'website_experience': website_experience,
                'products_used': products_used,
                'usage_frequency': usage_frequency,
                'recommend_likelihood': recommend_likelihood,
                'improvements': improvements,
                'best_experience': best_experience,
                'suggestions': suggestions,
                'additional_comments': additional_comments,
                'follow_up': follow_up,
                'newsletter': newsletter,
                'email_updates': email_updates,
                'survey_participation': survey_participation
            }
            
            # Save response
            st.session_state.survey_responses.append(response_data)
            st.success("🎉 Thank you for your feedback! Your response has been recorded.")
            st.balloons()
            
            # Show summary
            with st.expander("📊 Your Response Summary"):
                st.json(response_data, expanded=False)

def show_analytics():
    """Display survey analytics dashboard"""
    st.title("📊 Survey Analytics Dashboard")
    
    responses = st.session_state.survey_responses
    
    if not responses:
        st.info("No survey responses yet. Complete a survey to see analytics!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(responses)
    
    # Key metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Responses", len(df))
    
    with col2:
        avg_satisfaction = df['overall_satisfaction'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
    
    with col3:
        avg_nps = df['recommend_likelihood'].mean()
        st.metric("Avg NPS Score", f"{avg_nps:.1f}/10")
    
    with col4:
        response_rate = len(df) / (len(df) + 10) * 100  # Simulated
        st.metric("Response Rate", f"{response_rate:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction distribution
        satisfaction_counts = df['overall_satisfaction'].value_counts().sort_index()
        fig_sat = px.bar(
            x=satisfaction_counts.index,
            y=satisfaction_counts.values,
            title="Overall Satisfaction Distribution",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        st.plotly_chart(fig_sat, use_container_width=True)
        
        # Customer type breakdown
        customer_type_counts = df['customer_type'].value_counts()
        fig_customer = px.pie(
            values=customer_type_counts.values,
            names=customer_type_counts.index,
            title="Customer Type Distribution"
        )
        st.plotly_chart(fig_customer, use_container_width=True)
    
    with col2:
        # NPS distribution
        nps_counts = df['recommend_likelihood'].value_counts().sort_index()
        fig_nps = px.bar(
            x=nps_counts.index,
            y=nps_counts.values,
            title="NPS Score Distribution",
            labels={'x': 'Score (0-10)', 'y': 'Count'}
        )
        st.plotly_chart(fig_nps, use_container_width=True)
        
        # Location breakdown
        location_counts = df['location'].value_counts()
        fig_location = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            title="Responses by Location"
        )
        st.plotly_chart(fig_location, use_container_width=True)
    
    # Detailed analysis
    if st.checkbox("Show detailed analysis"):
        st.subheader("🔍 Detailed Analysis")
        
        # Rating correlations
        rating_cols = ['overall_satisfaction', 'product_quality', 'value_for_money', 
                      'customer_service', 'delivery_speed', 'website_experience']
        
        if len(df) > 1:
            correlation_matrix = df[rating_cols].corr()
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                title="Rating Correlations"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Raw data
        st.subheader("📋 Raw Survey Data")
        st.dataframe(df, use_container_width=True)
        
        # Download data
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Survey Data",
            csv,
            "survey_responses.csv",
            "text/csv"
        )

def main():
    """Main application"""
    # Navigation
    st.sidebar.title("📋 Survey App")
    page = st.sidebar.selectbox(
        "Navigate:",
        ["Take Survey", "View Analytics", "Survey Management"]
    )
    
    if page == "Take Survey":
        create_survey_form()
    elif page == "View Analytics":
        show_analytics()
    elif page == "Survey Management":
        st.title("⚙️ Survey Management")
        st.write("Survey configuration and management options")
        
        if st.button("🗑️ Clear All Responses"):
            st.session_state.survey_responses = []
            st.success("All responses cleared!")
        
        if st.session_state.survey_responses:
            st.write(f"Current responses: {len(st.session_state.survey_responses)}")

if __name__ == "__main__":
    main()
```

## Key Learning Points

### Advanced Streamlit Features Demonstrated
1. **Complex Form Handling**: Multi-step forms with validation
2. **Session State Management**: Persistent data across page interactions
3. **Dynamic Content**: Conditional rendering based on user input
4. **File Processing**: Multiple file type handling and analysis
5. **Data Visualization**: Interactive charts with Plotly
6. **Layout Optimization**: Professional dashboard layouts

### Best Practices Applied
- **Modular Code Structure**: Functions for different components
- **Error Handling**: Graceful handling of file processing errors
- **User Experience**: Clear navigation and progress indicators
- **Data Export**: Download capabilities for processed data
- **Responsive Design**: Mobile-friendly layouts

### Project Complexity Progression
- **Forms**: From simple to multi-step with validation
- **Data Handling**: From basic display to advanced filtering
- **File Processing**: Multiple formats with analysis capabilities
- **Interactive Features**: Real-time updates and dynamic content

## Next Steps
- Add database integration for persistent storage
- Implement user authentication and role-based access
- Create more advanced visualizations
- Add real-time data updates and notifications
- Develop custom components for enhanced functionality
