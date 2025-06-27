# Plotly Interactive Visualization

## Overview
Plotly is a powerful library for creating interactive visualizations that enhance data exploration and ML model interpretation. Essential for building engaging dashboards and presentations.

## Installation and Setup

```python
# Install required packages
pip install plotly dash kaleido

# Import libraries
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
```

## Basic Interactive Plots

### Scatter Plots with Hover Information
```python
# Basic interactive scatter plot
import plotly.express as px

# Sample ML dataset
df = px.data.iris()

fig = px.scatter(df, x="sepal_width", y="sepal_length", 
                 color="species", size="petal_length",
                 hover_data=['petal_width'],
                 title="Iris Dataset Interactive Scatter Plot")

# Customize hover template
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "Sepal Width: %{x}<br>" +
                  "Sepal Length: %{y}<br>" +
                  "Petal Width: %{customdata[0]}<br>" +
                  "<extra></extra>"
)

fig.show()
```

### Line Plots with Zoom and Pan
```python
# Time series with interactive features
dates = pd.date_range('2020-01-01', periods=365)
values = np.cumsum(np.random.randn(365)) + 100

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates, y=values,
    mode='lines+markers',
    name='Stock Price',
    line=dict(width=2),
    marker=dict(size=4),
    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    title="Interactive Time Series",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode='x unified'
)

fig.show()
```

## Machine Learning Visualizations

### Model Performance Dashboard
```python
# Classification results visualization
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, 
                          n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Interactive confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True, aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual", color="Count"))
fig.show()

# Feature importance visualization
importances = model.feature_importances_
features = [f'Feature_{i}' for i in range(len(importances))]

fig = px.bar(x=features, y=importances,
             title="Feature Importance",
             labels={'x': 'Features', 'y': 'Importance'})
fig.update_traces(marker_color='lightblue')
fig.show()
```

### Interactive Learning Curves
```python
from sklearn.model_selection import learning_curve

# Generate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42), X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Create interactive learning curve
fig = go.Figure()

# Training scores
fig.add_trace(go.Scatter(
    x=train_sizes, y=train_mean,
    mode='lines+markers',
    name='Training Score',
    line=dict(color='blue'),
    error_y=dict(type='data', array=train_std, visible=True)
))

# Validation scores
fig.add_trace(go.Scatter(
    x=train_sizes, y=val_mean,
    mode='lines+markers',
    name='Validation Score',
    line=dict(color='red'),
    error_y=dict(type='data', array=val_std, visible=True)
))

fig.update_layout(
    title="Learning Curves",
    xaxis_title="Training Set Size",
    yaxis_title="Accuracy Score",
    hovermode='x unified'
)

fig.show()
```

## 3D Visualizations

### 3D Scatter Plots for Clustering
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate 3D clustering data
X_3d, y_true = make_blobs(n_samples=300, centers=4, n_features=3, 
                          random_state=42, cluster_std=1.5)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X_3d)

# Create 3D scatter plot
fig = go.Figure()

# Plot data points
fig.add_trace(go.Scatter3d(
    x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=y_pred,
        colorscale='Viridis',
        showscale=True
    ),
    text=[f'Cluster: {cluster}' for cluster in y_pred],
    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>%{text}<extra></extra>'
))

# Plot cluster centers
centers = kmeans.cluster_centers_
fig.add_trace(go.Scatter3d(
    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
    mode='markers',
    marker=dict(size=15, color='red', symbol='diamond'),
    name='Cluster Centers'
))

fig.update_layout(
    title="3D K-Means Clustering Results",
    scene=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Feature 3'
    )
)

fig.show()
```

### Surface Plots for Function Visualization
```python
# Visualize loss function landscape
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Simple quadratic function

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])

fig.update_layout(
    title="Loss Function Landscape",
    scene=dict(
        xaxis_title='Parameter 1',
        yaxis_title='Parameter 2',
        zaxis_title='Loss'
    )
)

fig.show()
```

## Dashboard Creation with Plotly Dash

### Basic ML Dashboard
```python
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)

# Sample data
df = px.data.iris()

# Define layout
app.layout = html.Div([
    html.H1("ML Dashboard", style={'text-align': 'center'}),
    
    html.Div([
        html.Label("Select Feature for X-axis:"),
        dcc.Dropdown(
            id='x-feature',
            options=[{'label': col, 'value': col} for col in df.columns[:-1]],
            value='sepal_length'
        )
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Select Feature for Y-axis:"),
        dcc.Dropdown(
            id='y-feature',
            options=[{'label': col, 'value': col} for col in df.columns[:-1]],
            value='sepal_width'
        )
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='histogram')
])

# Callback for scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-feature', 'value'),
     Input('y-feature', 'value')]
)
def update_scatter(x_feature, y_feature):
    fig = px.scatter(df, x=x_feature, y=y_feature, color='species',
                     title=f'{x_feature} vs {y_feature}')
    return fig

# Callback for histogram
@app.callback(
    Output('histogram', 'figure'),
    [Input('x-feature', 'value')]
)
def update_histogram(feature):
    fig = px.histogram(df, x=feature, color='species',
                       title=f'Distribution of {feature}')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
```

### Real-time Model Monitoring Dashboard
```python
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import random
import datetime

app = dash.Dash(__name__)

# Simulate real-time data
def generate_data():
    return {
        'time': datetime.datetime.now(),
        'accuracy': random.uniform(0.8, 0.95),
        'loss': random.uniform(0.1, 0.5),
        'predictions': random.randint(100, 1000)
    }

app.layout = html.Div([
    html.H1("Real-time Model Monitoring"),
    
    html.Div([
        html.Div([
            html.H3("Model Accuracy"),
            dcc.Graph(id='accuracy-gauge')
        ], className='six columns'),
        
        html.Div([
            html.H3("Prediction Volume"),
            dcc.Graph(id='prediction-count')
        ], className='six columns')
    ], className='row'),
    
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    )
])

@app.callback(
    [Output('accuracy-gauge', 'figure'),
     Output('prediction-count', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    data = generate_data()
    
    # Accuracy gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=data['accuracy'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Accuracy"},
        delta={'reference': 0.9},
        gauge={'axis': {'range': [None, 1]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [0, 0.8], 'color': "lightgray"},
                        {'range': [0.8, 0.9], 'color': "yellow"},
                        {'range': [0.9, 1], 'color': "green"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 0.9}}))
    
    # Prediction count
    pred_fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=data['predictions'],
        title={'text': "Predictions/Hour"},
        delta={'reference': 500, 'relative': True},
        number={'suffix': " pred/hr"}))
    
    return gauge, pred_fig
```

## Advanced Interactive Features

### Crossfilter-style Interactions
```python
# Multiple linked plots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Scatter Plot', 'Histogram', 'Box Plot', 'Violin Plot'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

df = px.data.iris()

# Add scatter plot
fig.add_trace(
    go.Scatter(x=df['sepal_width'], y=df['sepal_length'], 
               mode='markers', name='Data Points'),
    row=1, col=1
)

# Add histogram
fig.add_trace(
    go.Histogram(x=df['petal_length'], name='Petal Length'),
    row=1, col=2
)

# Add box plot
fig.add_trace(
    go.Box(y=df['petal_width'], name='Petal Width'),
    row=2, col=1
)

# Add violin plot
fig.add_trace(
    go.Violin(y=df['sepal_length'], name='Sepal Length'),
    row=2, col=2
)

fig.update_layout(height=600, showlegend=False, 
                  title_text="Multi-Plot Dashboard")
fig.show()
```

### Animation for Model Training
```python
# Animate gradient descent
def gradient_descent_data():
    x = np.linspace(-10, 10, 100)
    frames = []
    
    for step in range(20):
        # Simulate parameter updates
        theta = 5 - step * 0.5
        y = (x - theta)**2 + 2
        
        frames.append(go.Frame(
            data=[go.Scatter(x=x, y=y, mode='lines', name=f'Step {step}'),
                  go.Scatter(x=[theta], y=[2], mode='markers', 
                           marker=dict(size=15, color='red'), name='Current Point')],
            name=str(step)
        ))
    
    return frames

frames = gradient_descent_data()

fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

fig.update_layout(
    title="Gradient Descent Animation",
    xaxis_title="Parameter Value",
    yaxis_title="Loss",
    updatemenus=[dict(type="buttons", 
                     buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": 500}}])])]
)

fig.show()
```

## Best Practices

### Performance Optimization
```python
# Use WebGL for large datasets
fig = go.Figure()
fig.add_trace(go.Scattergl(  # Note: Scattergl instead of Scatter
    x=np.random.randn(100000),
    y=np.random.randn(100000),
    mode='markers',
    marker=dict(size=2)
))

# Optimize for web deployment
fig.update_layout(
    title="Large Dataset Visualization",
    template="plotly_white"  # Clean template for web
)
```

### Responsive Design
```python
# Mobile-friendly configuration
fig.update_layout(
    autosize=True,
    margin=dict(l=0, r=0, t=30, b=0),
    font=dict(size=12),
    showlegend=False,
    # Responsive configuration
    responsive=True
)

# Export configurations
fig.write_html("dashboard.html", 
               config={'displayModeBar': False,  # Hide toolbar
                      'responsive': True})
```

## Integration with ML Workflows

### Model Comparison Dashboard
```python
# Compare multiple models
models_performance = {
    'Random Forest': {'accuracy': 0.92, 'precision': 0.89, 'recall': 0.94},
    'SVM': {'accuracy': 0.88, 'precision': 0.91, 'recall': 0.85},
    'Logistic Regression': {'accuracy': 0.85, 'precision': 0.87, 'recall': 0.83}
}

metrics = ['accuracy', 'precision', 'recall']
models = list(models_performance.keys())

fig = go.Figure()

for metric in metrics:
    values = [models_performance[model][metric] for model in models]
    fig.add_trace(go.Bar(
        name=metric.title(),
        x=models,
        y=values,
        text=[f'{v:.3f}' for v in values],
        textposition='auto'
    ))

fig.update_layout(
    title="Model Performance Comparison",
    xaxis_title="Models",
    yaxis_title="Score",
    barmode='group'
)

fig.show()
```

## Learning Objectives
- [x] Create interactive scatter plots with hover information
- [x] Build 3D visualizations for clustering and surfaces
- [x] Develop real-time dashboards with Plotly Dash
- [x] Implement animation for model training visualization
- [x] Create responsive ML dashboards
- [x] Optimize performance for large datasets
- [x] Integrate with ML model evaluation workflows

## Next Steps
1. Explore advanced Dash components (DataTable, etc.)
2. Learn deployment strategies (Heroku, AWS, etc.)
3. Study integration with Jupyter notebooks
4. Practice building domain-specific dashboards
5. Master advanced animation techniques