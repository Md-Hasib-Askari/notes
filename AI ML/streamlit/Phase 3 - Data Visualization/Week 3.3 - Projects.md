# Week 3.3: Data Visualization Projects

## Overview
Build four comprehensive projects that demonstrate mastery of Streamlit's visualization capabilities, combining built-in charts with advanced plotting libraries.

---

## Project 1: Sales Dashboard

### Features
- Real-time sales metrics and KPIs
- Interactive filtering and drill-down capabilities
- Multiple visualization types (line, bar, area, maps)
- Export functionality for reports

### Implementation
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Sales Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def generate_sales_data():
    """Generate comprehensive sales dataset"""
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate data
    data = []
    for date in dates:
        for region in ['North', 'South', 'East', 'West']:
            for product in ['Product A', 'Product B', 'Product C']:
                sales = np.random.normal(1000, 200) * (1 + 0.1 * np.sin(date.timetuple().tm_yday / 365 * 2 * np.pi))
                data.append({
                    'date': date,
                    'region': region,
                    'product': product,
                    'sales': max(0, sales),
                    'units_sold': int(sales / np.random.uniform(10, 50)),
                    'salesperson': np.random.choice([f'Rep {i}' for i in range(1, 6)])
                })
    
    return pd.DataFrame(data)

def main():
    st.title("📊 Sales Performance Dashboard")
    
    # Load data
    df = generate_sales_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    
    # Region filter
    regions = st.sidebar.multiselect("Regions:", 
                                   options=df['region'].unique(), 
                                   default=df['region'].unique())
    
    # Product filter
    products = st.sidebar.multiselect("Products:", 
                                    options=df['product'].unique(), 
                                    default=df['product'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1]) &
        (df['region'].isin(regions)) &
        (df['product'].isin(products))
    ]
    
    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = filtered_df['sales'].sum()
    total_units = filtered_df['units_sold'].sum()
    avg_daily_sales = filtered_df.groupby('date')['sales'].sum().mean()
    top_region = filtered_df.groupby('region')['sales'].sum().idxmax()
    
    with col1:
        st.metric("Total Sales", f"${total_sales:,.0f}")
    with col2:
        st.metric("Units Sold", f"{total_units:,}")
    with col3:
        st.metric("Avg Daily Sales", f"${avg_daily_sales:,.0f}")
    with col4:
        st.metric("Top Region", top_region)
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Comparison", "🗺️ Geographic", "👥 Performance"])
    
    with tab1:
        # Sales trend over time
        daily_sales = filtered_df.groupby('date')['sales'].sum().reset_index()
        fig = px.line(daily_sales, x='date', y='sales', title="Daily Sales Trend")
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by product over time (area chart)
        product_sales = filtered_df.groupby(['date', 'product'])['sales'].sum().reset_index()
        fig = px.area(product_sales, x='date', y='sales', color='product', 
                     title="Sales by Product Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by region
            region_sales = filtered_df.groupby('region')['sales'].sum().reset_index()
            fig = px.bar(region_sales, x='region', y='sales', title="Sales by Region")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sales by product
            product_sales = filtered_df.groupby('product')['sales'].sum().reset_index()
            fig = px.pie(product_sales, values='sales', names='product', 
                        title="Sales Distribution by Product")
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly comparison
        filtered_df['month'] = filtered_df['date'].dt.to_period('M')
        monthly_sales = filtered_df.groupby(['month', 'region'])['sales'].sum().reset_index()
        monthly_sales['month'] = monthly_sales['month'].astype(str)
        
        fig = px.bar(monthly_sales, x='month', y='sales', color='region',
                    title="Monthly Sales by Region", barmode='group')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Simulated geographic data
        region_coords = {
            'North': (45.0, -100.0),
            'South': (30.0, -95.0),
            'East': (40.0, -75.0),
            'West': (35.0, -120.0)
        }
        
        geo_data = []
        for region in filtered_df['region'].unique():
            if region in region_coords:
                lat, lon = region_coords[region]
                sales = filtered_df[filtered_df['region'] == region]['sales'].sum()
                geo_data.append({'region': region, 'lat': lat, 'lon': lon, 'sales': sales})
        
        geo_df = pd.DataFrame(geo_data)
        
        if not geo_df.empty:
            # Map visualization
            st.subheader("Sales by Geographic Region")
            st.map(geo_df[['lat', 'lon']])
            
            # Bubble map with Plotly
            fig = px.scatter_mapbox(geo_df, lat='lat', lon='lon', size='sales',
                                  hover_name='region', hover_data=['sales'],
                                  color='sales', size_max=50, zoom=3,
                                  mapbox_style='open-street-map',
                                  title="Sales Distribution Map")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Salesperson performance
        rep_performance = filtered_df.groupby('salesperson')['sales'].agg(['sum', 'mean', 'count']).reset_index()
        rep_performance.columns = ['salesperson', 'total_sales', 'avg_sales', 'transactions']
        rep_performance = rep_performance.sort_values('total_sales', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(rep_performance, x='salesperson', y='total_sales',
                        title="Total Sales by Salesperson")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(rep_performance, x='transactions', y='avg_sales',
                           size='total_sales', hover_name='salesperson',
                           title="Sales Efficiency Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("Detailed Performance Metrics")
        st.dataframe(rep_performance.round(2), use_container_width=True)
    
    # Export functionality
    st.sidebar.header("📥 Export Data")
    if st.sidebar.button("Generate Report"):
        # Create summary report
        summary = {
            'Total Sales': f"${total_sales:,.0f}",
            'Total Units': f"{total_units:,}",
            'Average Daily Sales': f"${avg_daily_sales:,.0f}",
            'Top Performing Region': top_region,
            'Date Range': f"{date_range[0]} to {date_range[1]}",
            'Products Analyzed': ', '.join(products),
            'Regions Analyzed': ', '.join(regions)
        }
        
        st.sidebar.success("Report generated!")
        for key, value in summary.items():
            st.sidebar.text(f"{key}: {value}")

if __name__ == "__main__":
    main()
```

---

## Project 2: Stock Price Analyzer

### Features
- Real-time stock price data visualization
- Technical indicators and analysis
- Portfolio tracking and comparison
- Interactive charts with zoom and pan

### Implementation
```python
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data, stock.info
    except:
        return None, None

def calculate_moving_averages(data, windows=[20, 50]):
    """Calculate moving averages"""
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    return data

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def main():
    st.title("📈 Stock Price Analyzer")
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Settings")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    
    # Time period selection
    period = st.sidebar.selectbox("Time Period", 
                                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                index=5)
    
    # Technical indicators
    show_ma = st.sidebar.checkbox("Moving Averages", True)
    show_volume = st.sidebar.checkbox("Volume", True)
    show_rsi = st.sidebar.checkbox("RSI", False)
    
    if symbol:
        data, info = get_stock_data(symbol, period)
        
        if data is not None and not data.empty:
            # Calculate indicators
            data = calculate_moving_averages(data)
            data = calculate_rsi(data)
            
            # Company info
            if info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Company", info.get('longName', symbol))
                with col2:
                    st.metric("Sector", info.get('sector', 'N/A'))
                with col3:
                    st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")
            
            # Price metrics
            current_price = data['Close'][-1]
            prev_close = data['Close'][-2] if len(data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
            with col3:
                st.metric("High", f"${data['High'].max():.2f}")
            with col4:
                st.metric("Low", f"${data['Low'].min():.2f}")
            
            # Create subplots
            rows = 1
            if show_volume:
                rows += 1
            if show_rsi:
                rows += 1
            
            subplot_titles = ["Price"]
            if show_volume:
                subplot_titles.append("Volume")
            if show_rsi:
                subplot_titles.append("RSI")
            
            fig = make_subplots(rows=rows, cols=1, 
                              shared_xaxes=True,
                              subplot_titles=subplot_titles,
                              vertical_spacing=0.05,
                              row_heights=[0.6] + [0.2] * (rows-1))
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(x=data.index,
                                       open=data['Open'],
                                       high=data['High'],
                                       low=data['Low'],
                                       close=data['Close'],
                                       name="Price"), row=1, col=1)
            
            # Moving averages
            if show_ma:
                fig.add_trace(go.Scatter(x=data.index, y=data['MA20'],
                                       name="MA20", line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['MA50'],
                                       name="MA50", line=dict(color='red')), row=1, col=1)
            
            current_row = 2
            
            # Volume
            if show_volume:
                colors = ['red' if close < open else 'green' 
                         for close, open in zip(data['Close'], data['Open'])]
                fig.add_trace(go.Bar(x=data.index, y=data['Volume'],
                                   name="Volume", marker_color=colors), 
                            row=current_row, col=1)
                current_row += 1
            
            # RSI
            if show_rsi:
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                                       name="RSI", line=dict(color='purple')), 
                            row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            
            fig.update_layout(title=f"{symbol} Stock Analysis",
                            xaxis_rangeslider_visible=False,
                            height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Statistical Analysis")
                stats = {
                    "Mean Price": f"${data['Close'].mean():.2f}",
                    "Standard Deviation": f"${data['Close'].std():.2f}",
                    "Volatility (30d)": f"{data['Close'].pct_change().rolling(30).std().iloc[-1]*100:.2f}%",
                    "Volume Average": f"{data['Volume'].mean():,.0f}"
                }
                for stat, value in stats.items():
                    st.text(f"{stat}: {value}")
            
            with col2:
                st.subheader("📈 Performance")
                if len(data) > 1:
                    returns = data['Close'].pct_change().dropna()
                    cumulative_return = (data['Close'][-1] / data['Close'][0] - 1) * 100
                    
                    perf = {
                        "Total Return": f"{cumulative_return:.2f}%",
                        "Best Day": f"{returns.max()*100:.2f}%",
                        "Worst Day": f"{returns.min()*100:.2f}%",
                        "Sharpe Ratio": f"{returns.mean()/returns.std():.2f}" if returns.std() > 0 else "N/A"
                    }
                    for metric, value in perf.items():
                        st.text(f"{metric}: {value}")
            
            # Download data
            if st.button("📥 Download Data"):
                csv = data.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{symbol}_stock_data.csv",
                    mime="text/csv"
                )
        
        else:
            st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")

if __name__ == "__main__":
    main()
```

---

## Key Learning Points

### Visualization Techniques
- **Dashboard Layout**: Multi-column and tabbed interfaces
- **Interactive Filtering**: Dynamic data exploration
- **Real-time Updates**: Caching strategies for live data
- **Multiple Chart Types**: Combining different visualization methods

### Technical Skills
- **Data Processing**: Aggregation and transformation
- **Performance Optimization**: Caching and efficient data handling
- **User Experience**: Intuitive navigation and clear metrics
- **Export Functionality**: Data download and reporting

### Best Practices
- Use appropriate chart types for different data relationships
- Implement responsive design with `use_container_width=True`
- Cache expensive operations for better performance
- Provide clear navigation and filtering options
- Include summary metrics and key insights

### Next Steps
- Add real-time data streaming capabilities
- Implement advanced statistical analysis
- Create custom plotting functions
- Build multi-page dashboard applications
- Integrate with external APIs and databases
