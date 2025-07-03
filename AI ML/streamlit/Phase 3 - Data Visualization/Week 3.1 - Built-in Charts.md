# Week 3.1: Built-in Charts

## Overview
Streamlit provides simple, built-in charting functions that create beautiful visualizations with minimal code. Perfect for quick data exploration and prototyping.

---

## Line Charts: `st.line_chart()`

### Basic Usage
```python
import streamlit as st
import pandas as pd
import numpy as np

# Generate sample time series data
dates = pd.date_range('2024-01-01', periods=100)
data = pd.DataFrame({
    'Sales': np.cumsum(np.random.randn(100)) + 100,
    'Marketing': np.cumsum(np.random.randn(100)) + 80,
    'Support': np.cumsum(np.random.randn(100)) + 60
}, index=dates)

st.title("📈 Line Charts")
st.line_chart(data)
```

### Advanced Line Chart Examples
```python
# Single column line chart
st.subheader("Single Metric Over Time")
single_metric = pd.DataFrame({
    'Revenue': np.cumsum(np.random.randn(50)) + 1000
}, index=pd.date_range('2024-01-01', periods=50))
st.line_chart(single_metric)

# Custom height and use_container_width
st.subheader("Customized Line Chart")
st.line_chart(data, height=400, use_container_width=True)

# Interactive filtering with line charts
st.subheader("Interactive Line Chart")
metrics = st.multiselect("Select metrics to display:", 
                        options=data.columns, 
                        default=data.columns.tolist())
if metrics:
    st.line_chart(data[metrics])
```

### Best Practices
- Use for time series data and trends
- Keep lines to 5 or fewer for readability
- Ensure data index is datetime for time series
- Use meaningful column names (they become legend labels)

---

## Bar Charts: `st.bar_chart()`

### Basic Usage
```python
# Sample categorical data
categories = ['Q1', 'Q2', 'Q3', 'Q4']
data = pd.DataFrame({
    'Revenue': [120, 150, 180, 200],
    'Expenses': [80, 90, 100, 110],
    'Profit': [40, 60, 80, 90]
}, index=categories)

st.title("📊 Bar Charts")
st.bar_chart(data)
```

### Advanced Bar Chart Examples
```python
# Horizontal layout with columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Quarterly Performance")
    quarterly_data = pd.DataFrame({
        'Sales': [100, 120, 140, 160],
        'Target': [110, 115, 135, 150]
    }, index=['Q1', 'Q2', 'Q3', 'Q4'])
    st.bar_chart(quarterly_data)

with col2:
    st.subheader("Department Budgets")
    dept_data = pd.DataFrame({
        'Budget': [50000, 75000, 60000, 40000, 30000]
    }, index=['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'])
    st.bar_chart(dept_data)

# Dynamic bar chart with user input
st.subheader("Interactive Bar Chart")
num_categories = st.slider("Number of categories", 3, 10, 5)
random_data = pd.DataFrame({
    'Values': np.random.randint(10, 100, num_categories)
}, index=[f'Category {i+1}' for i in range(num_categories)])
st.bar_chart(random_data)
```

### Best Practices
- Use for comparing categorical data
- Sort data meaningfully (ascending, descending, or logical order)
- Limit to 10-15 categories for readability
- Use consistent color schemes across related charts

---

## Area Charts: `st.area_chart()`

### Basic Usage
```python
# Stacked area chart showing cumulative values
st.title("📈 Area Charts")

# Generate sample data for different product lines
dates = pd.date_range('2024-01-01', periods=60)
area_data = pd.DataFrame({
    'Product A': np.random.randint(20, 50, 60),
    'Product B': np.random.randint(15, 40, 60),
    'Product C': np.random.randint(10, 30, 60)
}, index=dates)

st.area_chart(area_data)
```

### Advanced Area Chart Examples
```python
# Cumulative area chart
st.subheader("Cumulative Sales by Product")
cumulative_data = area_data.cumsum()
st.area_chart(cumulative_data)

# Interactive area chart with date range
st.subheader("Sales Trend Analysis")
start_date = st.date_input("Start date", value=dates[0].date())
end_date = st.date_input("End date", value=dates[-1].date())

# Filter data based on date range
mask = (area_data.index.date >= start_date) & (area_data.index.date <= end_date)
filtered_data = area_data.loc[mask]

if not filtered_data.empty:
    st.area_chart(filtered_data)
else:
    st.warning("No data available for selected date range")

# Percentage area chart
st.subheader("Market Share Over Time")
percentage_data = area_data.div(area_data.sum(axis=1), axis=0) * 100
st.area_chart(percentage_data)
```

### Best Practices
- Perfect for showing part-to-whole relationships over time
- Use when you want to emphasize the cumulative effect
- Limit to 3-5 series for clarity
- Consider the order of series (put largest at bottom)

---

## Maps: `st.map()`

### Basic Usage
```python
import numpy as np

st.title("🗺️ Maps")

# Generate sample location data
map_data = pd.DataFrame({
    'lat': np.random.randn(1000) * 0.1 + 37.76,  # San Francisco area
    'lon': np.random.randn(1000) * 0.1 + -122.4
})

st.map(map_data)
```

### Advanced Map Examples
```python
# Store locations with different zoom levels
st.subheader("Store Locations")

# Predefined store locations
stores = pd.DataFrame({
    'lat': [37.7749, 37.7849, 37.7649, 37.7549],
    'lon': [-122.4194, -122.4094, -122.4294, -122.4394],
    'size': [100, 150, 200, 120]  # This won't affect built-in map but good for data structure
})

st.map(stores)

# Interactive map with filtering
st.subheader("Crime Data Visualization")
num_points = st.slider("Number of data points", 100, 2000, 500)

# Generate crime data around major cities
cities = {
    'San Francisco': (37.7749, -122.4194),
    'New York': (40.7128, -74.0060),
    'Chicago': (41.8781, -87.6298),
    'Los Angeles': (34.0522, -118.2437)
}

selected_city = st.selectbox("Select city", list(cities.keys()))
lat_center, lon_center = cities[selected_city]

# Generate random points around selected city
crime_data = pd.DataFrame({
    'lat': np.random.normal(lat_center, 0.1, num_points),
    'lon': np.random.normal(lon_center, 0.1, num_points)
})

st.map(crime_data)

# Multi-layer map simulation (using different dataframes)
st.subheader("Multi-layer Analysis")
col1, col2 = st.columns(2)

with col1:
    show_restaurants = st.checkbox("Show Restaurants")
    show_hotels = st.checkbox("Show Hotels")

if show_restaurants or show_hotels:
    combined_data = pd.DataFrame()
    
    if show_restaurants:
        restaurants = pd.DataFrame({
            'lat': np.random.normal(37.7749, 0.05, 50),
            'lon': np.random.normal(-122.4194, 0.05, 50)
        })
        combined_data = pd.concat([combined_data, restaurants])
    
    if show_hotels:
        hotels = pd.DataFrame({
            'lat': np.random.normal(37.7749, 0.03, 30),
            'lon': np.random.normal(-122.4194, 0.03, 30)
        })
        combined_data = pd.concat([combined_data, hotels])
    
    if not combined_data.empty:
        st.map(combined_data)
```

### Best Practices
- Ensure lat/lon columns are named exactly 'lat' and 'lon'
- Keep data points reasonable (< 10,000 for performance)
- Consider data density - too many points can obscure patterns
- Use appropriate zoom level by centering data geographically

---

## Combining Multiple Chart Types

### Dashboard Example
```python
st.title("📊 Sales Dashboard")

# Generate sample sales data
dates = pd.date_range('2024-01-01', periods=30)
daily_sales = pd.DataFrame({
    'Online': np.random.randint(1000, 5000, 30),
    'Retail': np.random.randint(800, 4000, 30),
    'Wholesale': np.random.randint(500, 2000, 30)
}, index=dates)

# Create dashboard layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Sales Trend")
    st.line_chart(daily_sales)
    
    st.subheader("Sales by Channel")
    channel_totals = daily_sales.sum()
    st.bar_chart(channel_totals)

with col2:
    st.subheader("Cumulative Sales")
    st.area_chart(daily_sales.cumsum())
    
    # Summary metrics
    st.subheader("Summary")
    total_sales = daily_sales.sum().sum()
    st.metric("Total Sales", f"${total_sales:,}")
    
    best_day = daily_sales.sum(axis=1).max()
    st.metric("Best Day", f"${best_day:,}")
```

### Best Practices
- Use consistent color schemes across charts
- Provide clear titles and labels
- Consider chart hierarchy (most important first)
- Use appropriate chart types for data relationships
- Keep layouts clean and uncluttered

---

## Key Learning Points

### Technical Skills
- **Chart Selection**: Choose appropriate chart types for data
- **Data Preparation**: Format data correctly for each chart type
- **Interactivity**: Combine charts with widgets for dynamic visualization
- **Layout Design**: Organize multiple charts effectively

### Performance Tips
- Use `st.cache_data` for expensive data operations
- Limit data points for better performance
- Consider chart refresh frequency for real-time data

### Next Steps
- Learn advanced plotting with Matplotlib and Plotly
- Explore custom styling and theming
- Practice with real datasets
- Build complete dashboard applications
