# Week 3.2: Advanced Plotting

## Overview
Integrate powerful plotting libraries like Matplotlib, Plotly, and Altair for advanced, interactive, and customizable visualizations in Streamlit.

---

## Matplotlib Integration: `st.pyplot()`

### Basic Setup
```python
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.title("🎨 Advanced Plotting with Matplotlib")

# Basic matplotlib chart
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title("Sine Wave")
ax.set_xlabel("X values")
ax.set_ylabel("Y values")
st.pyplot(fig)
```

### Advanced Matplotlib Examples
```python
# Subplots with multiple visualizations
st.subheader("Multi-panel Visualization")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Panel 1: Line plot
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), label='sin(x)')
ax1.plot(x, np.cos(x), label='cos(x)')
ax1.legend()
ax1.set_title("Trigonometric Functions")

# Panel 2: Histogram
data = np.random.normal(100, 15, 1000)
ax2.hist(data, bins=30, alpha=0.7, color='skyblue')
ax2.set_title("Normal Distribution")
ax2.set_xlabel("Values")
ax2.set_ylabel("Frequency")

# Panel 3: Scatter plot
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100)
ax3.scatter(x_scatter, y_scatter, alpha=0.6)
ax3.set_title("Correlation Plot")
ax3.set_xlabel("X values")
ax3.set_ylabel("Y values")

# Panel 4: Box plot
box_data = [np.random.normal(0, 1, 100), 
           np.random.normal(1, 2, 100), 
           np.random.normal(-1, 0.5, 100)]
ax4.boxplot(box_data, labels=['Group A', 'Group B', 'Group C'])
ax4.set_title("Box Plot Comparison")

plt.tight_layout()
st.pyplot(fig)

# Interactive matplotlib with Streamlit widgets
st.subheader("Interactive Sine Wave")
frequency = st.slider("Frequency", 0.1, 5.0, 1.0)
amplitude = st.slider("Amplitude", 0.1, 3.0, 1.0)
phase = st.slider("Phase", 0, 2*np.pi, 0.0)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 4*np.pi, 1000)
y = amplitude * np.sin(frequency * x + phase)
ax.plot(x, y, linewidth=2)
ax.set_title(f"Interactive Sine Wave: A={amplitude}, f={frequency}, φ={phase:.2f}")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
```

### Styling and Customization
```python
# Custom styling
plt.style.use('seaborn-v0_8')  # Use seaborn style
fig, ax = plt.subplots(figsize=(10, 6))

# Sample data
categories = ['Product A', 'Product B', 'Product C', 'Product D']
values = [23, 45, 56, 78]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax.bar(categories, values, color=colors)
ax.set_title("Sales by Product", fontsize=16, fontweight='bold')
ax.set_ylabel("Sales (thousands)", fontsize=12)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height}k', ha='center', va='bottom')

st.pyplot(fig)
```

---

## Plotly Integration: `st.plotly_chart()`

### Basic Plotly Setup
```python
import plotly.express as px
import plotly.graph_objects as go

st.title("⚡ Interactive Plotting with Plotly")

# Basic scatter plot
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", 
                color="species", size="petal_length",
                title="Iris Dataset Visualization")
st.plotly_chart(fig, use_container_width=True)
```

### Advanced Plotly Examples
```python
# Interactive dashboard with multiple plots
st.subheader("Sales Dashboard")

# Generate sample sales data
dates = pd.date_range('2024-01-01', periods=365)
sales_data = pd.DataFrame({
    'date': dates,
    'sales': np.cumsum(np.random.randn(365)) + 1000,
    'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    'product': np.random.choice(['A', 'B', 'C'], 365)
})

# Time series plot
fig_time = px.line(sales_data, x='date', y='sales', 
                  color='region', title="Sales Trend by Region")
fig_time.update_layout(hovermode='x unified')
st.plotly_chart(fig_time, use_container_width=True)

# 3D scatter plot
st.subheader("3D Product Analysis")
product_analysis = sales_data.groupby(['region', 'product']).agg({
    'sales': ['mean', 'std', 'count']
}).round(2)
product_analysis.columns = ['avg_sales', 'std_sales', 'count']
product_analysis = product_analysis.reset_index()

fig_3d = px.scatter_3d(product_analysis, x='avg_sales', y='std_sales', z='count',
                      color='region', symbol='product', size='count',
                      title="3D Product Performance Analysis")
st.plotly_chart(fig_3d, use_container_width=True)

# Interactive heatmap
st.subheader("Sales Heatmap")
pivot_data = sales_data.pivot_table(values='sales', index='region', 
                                   columns='product', aggfunc='mean')
fig_heatmap = px.imshow(pivot_data, text_auto=True, aspect="auto",
                       title="Average Sales by Region and Product")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Animated bubble chart
st.subheader("Animated Sales Evolution")
if st.button("Generate Animation"):
    # Create time-based data for animation
    animation_data = []
    for month in range(1, 13):
        for region in ['North', 'South', 'East', 'West']:
            animation_data.append({
                'month': month,
                'region': region,
                'sales': np.random.normal(1000, 200),
                'market_share': np.random.uniform(15, 35)
            })
    
    anim_df = pd.DataFrame(animation_data)
    fig_anim = px.scatter(anim_df, x="sales", y="market_share", 
                         animation_frame="month", animation_group="region",
                         size="sales", color="region", hover_name="region",
                         title="Sales vs Market Share Over Time")
    st.plotly_chart(fig_anim, use_container_width=True)
```

### Custom Plotly Charts
```python
# Custom gauge chart
st.subheader("Performance Gauge")
performance_score = st.slider("Performance Score", 0, 100, 75)

fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = performance_score,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Performance Score"},
    delta = {'reference': 80},
    gauge = {'axis': {'range': [None, 100]},
             'bar': {'color': "darkblue"},
             'steps': [{'range': [0, 50], 'color': "lightgray"},
                      {'range': [50, 80], 'color': "gray"}],
             'threshold': {'line': {'color': "red", 'width': 4},
                          'thickness': 0.75, 'value': 90}}))
st.plotly_chart(fig_gauge, use_container_width=True)
```

---

## Altair Integration

### Basic Altair Usage
```python
import altair as alt

st.title("📊 Statistical Visualizations with Altair")

# Sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Basic scatter plot with regression line
chart = alt.Chart(data).mark_circle(size=60).encode(
    x='x:Q',
    y='y:Q',
    color='category:N',
    tooltip=['x:Q', 'y:Q', 'category:N']
).interactive()

regression = alt.Chart(data).mark_line(color='red').transform_regression('x', 'y').encode(
    x='x:Q',
    y='y:Q'
)

st.altair_chart(chart + regression, use_container_width=True)
```

### Advanced Altair Examples
```python
# Multi-view dashboard
st.subheader("Linked Visualizations")

# Cars dataset
cars = alt.data.cars()

# Selection that will be linked across charts
click = alt.selection_multi(encodings=['color'])

# Scatter plot
scatter = alt.Chart(cars).mark_circle().encode(
    x=alt.X('Horsepower:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('Miles_per_Gallon:Q', scale=alt.Scale(zero=False)),
    color=alt.condition(click, 'Origin:N', alt.value('lightgray'))
).add_selection(click)

# Histogram
hist = alt.Chart(cars).mark_bar().encode(
    x='count()',
    y='Origin:N',
    color=alt.condition(click, 'Origin:N', alt.value('lightgray'))
).transform_filter(click)

# Combine charts
combined = scatter | hist
st.altair_chart(combined, use_container_width=True)
```

---

## Custom Visualizations with Third-party Libraries

### Seaborn Integration
```python
import seaborn as sns

st.subheader("Statistical Plots with Seaborn")

# Generate sample data
tips = sns.load_dataset('tips')

# Seaborn plots in matplotlib figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill', ax=ax1)
ax1.set_title('Total Bill by Day')

# Heatmap
corr_matrix = tips.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2)
ax2.set_title('Correlation Matrix')

plt.tight_layout()
st.pyplot(fig)
```

### Wordcloud Example
```python
# Uncomment to install: pip install wordcloud
# from wordcloud import WordCloud

st.subheader("Word Cloud Visualization")
text_input = st.text_area("Enter text for word cloud:", 
                         "Streamlit is amazing for data visualization and interactive dashboards")

if text_input:
    # Note: This requires wordcloud library
    st.info("Install wordcloud library: pip install wordcloud")
    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_input)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.imshow(wordcloud, interpolation='bilinear')
    # ax.axis('off')
    # st.pyplot(fig)
```

---

## Best Practices for Advanced Plotting

### Performance Optimization
```python
# Use caching for expensive plot generation
@st.cache_data
def generate_complex_plot(data_size):
    """Generate complex visualization with caching"""
    data = np.random.randn(data_size, 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[:, 0], data[:, 1], alpha=0.6)
    ax.set_title(f"Scatter plot with {data_size} points")
    return fig

# Use cached function
plot_size = st.selectbox("Data size", [1000, 5000, 10000])
fig = generate_complex_plot(plot_size)
st.pyplot(fig)
```

### Responsive Design
```python
# Responsive plotting based on container width
use_container_width = st.checkbox("Use container width", True)
if use_container_width:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.plotly_chart(fig)
```

---

## Key Learning Points

### Library Comparison
- **Matplotlib**: Best for publication-quality static plots
- **Plotly**: Excellent for interactive, web-ready visualizations
- **Altair**: Great for statistical visualizations and grammar of graphics
- **Seaborn**: Perfect for statistical data exploration

### Integration Tips
- Always use `use_container_width=True` for responsive design
- Cache expensive plotting operations
- Consider plot complexity vs. performance
- Use appropriate chart types for your data story

### Next Steps
- Explore advanced Plotly features (animations, 3D plots)
- Learn custom Altair transformations
- Practice with real-world datasets
- Build interactive dashboard applications
