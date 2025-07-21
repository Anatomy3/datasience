"""
Charts Component Library
Reusable chart components with consistent styling
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Color scheme
COLORS = {
    'primary': '#10b981',
    'secondary': '#34d399',
    'accent': '#6ee7b7',
    'success': '#059669',
    'info': '#06b6d4',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'light': '#f0fdf4',
    'white': '#ffffff',
    'text': '#1f2937',
    'gray': '#6b7280'
}

# Chart color palettes
COLOR_PALETTES = {
    'green_theme': [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success'], COLORS['info']],
    'diverging': [COLORS['danger'], COLORS['warning'], COLORS['gray'], COLORS['info'], COLORS['primary']],
    'sequential': [COLORS['light'], COLORS['accent'], COLORS['secondary'], COLORS['primary'], COLORS['success']],
    'categorical': [COLORS['primary'], COLORS['info'], COLORS['warning'], COLORS['success'], COLORS['danger']]
}

# Default layout settings
DEFAULT_LAYOUT = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font_family': 'Inter, sans-serif',
    'font_color': COLORS['text'],
    'margin': dict(l=40, r=40, t=60, b=40),
    'showlegend': True,
    'legend': dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
}

def apply_theme(fig, title=None, theme="modern_green"):
    """
    Apply consistent theme to plotly figures
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure
        title (str): Chart title
        theme (str): Theme name
    
    Returns:
        plotly.graph_objects.Figure: Styled figure
    """
    
    # Update layout with theme
    fig.update_layout(**DEFAULT_LAYOUT)
    
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color=COLORS['text'], family='Inter'),
                x=0.02,
                xanchor='left'
            )
        )
    
    # Update axes
    fig.update_xaxes(
        gridcolor='rgba(16, 185, 129, 0.1)',
        linecolor=COLORS['gray'],
        tickcolor=COLORS['gray'],
        title_font=dict(color=COLORS['text'])
    )
    
    fig.update_yaxes(
        gridcolor='rgba(16, 185, 129, 0.1)',
        linecolor=COLORS['gray'],
        tickcolor=COLORS['gray'],
        title_font=dict(color=COLORS['text'])
    )
    
    return fig

def create_salary_distribution_histogram(df, column='salary_in_usd', title="Salary Distribution"):
    """
    Create salary distribution histogram
    
    Args:
        df (pd.DataFrame): Data
        column (str): Salary column name
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Histogram figure
    """
    
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=title,
        color_discrete_sequence=[COLORS['primary']]
    )
    
    # Add mean and median lines
    mean_val = df[column].mean()
    median_val = df[column].median()
    
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLORS['warning'],
        annotation_text=f"Mean: ${mean_val:,.0f}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=COLORS['info'],
        annotation_text=f"Median: ${median_val:,.0f}",
        annotation_position="bottom"
    )
    
    fig.update_traces(
        marker=dict(
            color=COLORS['primary'],
            opacity=0.7,
            line=dict(width=1, color=COLORS['success'])
        )
    )
    
    fig.update_layout(
        xaxis_title="Salary (USD)",
        yaxis_title="Frequency",
        bargap=0.1
    )
    
    return apply_theme(fig, title)

def create_experience_salary_boxplot(df, x_col='experience_level', y_col='salary_in_usd', 
                                   title="Salary by Experience Level"):
    """
    Create box plot for salary by experience level
    
    Args:
        df (pd.DataFrame): Data
        x_col (str): X-axis column (categorical)
        y_col (str): Y-axis column (numeric)
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Box plot figure
    """
    
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=x_col,
        color_discrete_sequence=COLOR_PALETTES['green_theme']
    )
    
    # Customize box plot appearance
    fig.update_traces(
        marker=dict(
            outliercolor="rgba(219, 64, 82, 0.6)",
            line=dict(outliercolor="rgba(219, 64, 82, 0.6)", outlierwidth=2)
        ),
        line_color=COLORS['text']
    )
    
    fig.update_layout(
        xaxis_title="Experience Level",
        yaxis_title="Salary (USD)",
        showlegend=False
    )
    
    return apply_theme(fig, title)

def create_correlation_heatmap(df, title="Correlation Matrix"):
    """
    Create correlation heatmap for numeric columns
    
    Args:
        df (pd.DataFrame): Data
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Heatmap figure
    """
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        # Create empty heatmap with message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient numeric data for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['gray'])
        )
        return apply_theme(fig, title)
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        title=title,
        color_continuous_scale=[
            [0, COLORS['danger']],
            [0.5, COLORS['white']],
            [1, COLORS['primary']]
        ],
        aspect="auto",
        text_auto=True
    )
    
    # Update text appearance
    fig.update_traces(
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 12, "color": COLORS['text']}
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Correlation",
            titleside="right"
        )
    )
    
    return apply_theme(fig, title)

def create_scatter_plot(df, x_col, y_col, color_col=None, size_col=None, 
                       title="Scatter Plot", hover_data=None):
    """
    Create interactive scatter plot
    
    Args:
        df (pd.DataFrame): Data
        x_col (str): X-axis column
        y_col (str): Y-axis column
        color_col (str): Column for color coding
        size_col (str): Column for size coding
        title (str): Chart title
        hover_data (list): Additional columns for hover info
    
    Returns:
        plotly.graph_objects.Figure: Scatter plot figure
    """
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        hover_data=hover_data,
        color_discrete_sequence=COLOR_PALETTES['green_theme']
    )
    
    # Customize markers
    fig.update_traces(
        marker=dict(
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    )
    
    # Add trend line if both columns are numeric
    if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
        # Calculate trend line
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=p(df[x_col]),
                mode='lines',
                name='Trend Line',
                line=dict(color=COLORS['danger'], width=2, dash='dash')
            )
        )
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )
    
    return apply_theme(fig, title)

def create_pie_chart(df, values_col, names_col, title="Distribution"):
    """
    Create modern pie chart
    
    Args:
        df (pd.DataFrame): Data
        values_col (str): Values column
        names_col (str): Names/labels column
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Pie chart figure
    """
    
    # Aggregate data if needed
    if len(df) > 10:
        # Show top 10 and group others
        top_data = df.nlargest(9, values_col)
        others_sum = df.iloc[9:][values_col].sum()
        
        if others_sum > 0:
            others_row = pd.DataFrame({
                names_col: ['Others'],
                values_col: [others_sum]
            })
            plot_data = pd.concat([top_data, others_row], ignore_index=True)
        else:
            plot_data = top_data
    else:
        plot_data = df
    
    fig = px.pie(
        plot_data,
        values=values_col,
        names=names_col,
        title=title,
        color_discrete_sequence=COLOR_PALETTES['green_theme']
    )
    
    # Customize pie chart
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>' +
                     'Value: %{value}<br>' +
                     'Percentage: %{percent}<br>' +
                     '<extra></extra>',
        marker=dict(
            line=dict(color='white', width=2)
        )
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return apply_theme(fig, title)

def create_bar_chart(df, x_col, y_col, color_col=None, orientation='v', 
                    title="Bar Chart", sort_values=True):
    """
    Create modern bar chart
    
    Args:
        df (pd.DataFrame): Data
        x_col (str): X-axis column
        y_col (str): Y-axis column
        color_col (str): Column for color coding
        orientation (str): 'v' for vertical, 'h' for horizontal
        title (str): Chart title
        sort_values (bool): Whether to sort by values
    
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    
    plot_data = df.copy()
    
    # Sort data if requested
    if sort_values:
        plot_data = plot_data.sort_values(y_col, ascending=(orientation == 'h'))
    
    if orientation == 'h':
        fig = px.bar(
            plot_data,
            x=y_col,
            y=x_col,
            color=color_col,
            orientation='h',
            title=title,
            color_discrete_sequence=COLOR_PALETTES['green_theme']
        )
    else:
        fig = px.bar(
            plot_data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            color_discrete_sequence=COLOR_PALETTES['green_theme']
        )
    
    # Customize bars
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        hovertemplate='<b>%{x}</b><br>' +
                     'Value: %{y}<br>' +
                     '<extra></extra>'
    )
    
    # Update layout based on orientation
    if orientation == 'h':
        fig.update_layout(
            xaxis_title=y_col.replace('_', ' ').title(),
            yaxis_title=x_col.replace('_', ' ').title(),
            yaxis={'categoryorder': 'total ascending'}
        )
    else:
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
    
    return apply_theme(fig, title)

def create_line_chart(df, x_col, y_col, color_col=None, title="Time Series"):
    """
    Create line chart for time series or continuous data
    
    Args:
        df (pd.DataFrame): Data
        x_col (str): X-axis column
        y_col (str): Y-axis column
        color_col (str): Column for multiple lines
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Line chart figure
    """
    
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_sequence=COLOR_PALETTES['green_theme']
    )
    
    # Customize lines
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=2, color='white'))
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     f'{x_col}: %{{x}}<br>' +
                     f'{y_col}: %{{y}}<br>' +
                     '<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        hovermode='x unified'
    )
    
    return apply_theme(fig, title)

def create_gauge_chart(value, title="Gauge", min_val=0, max_val=100, 
                      thresholds=None, unit=""):
    """
    Create gauge chart for KPI display
    
    Args:
        value (float): Current value
        title (str): Chart title
        min_val (float): Minimum value
        max_val (float): Maximum value
        thresholds (dict): Threshold values for color zones
        unit (str): Unit symbol
    
    Returns:
        plotly.graph_objects.Figure: Gauge chart figure
    """
    
    if thresholds is None:
        thresholds = {
            'low': max_val * 0.3,
            'medium': max_val * 0.7,
            'high': max_val
        }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title} ({unit})" if unit else title},
        delta={'reference': thresholds['medium']},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [min_val, thresholds['low']], 'color': COLORS['danger']},
                {'range': [thresholds['low'], thresholds['medium']], 'color': COLORS['warning']},
                {'range': [thresholds['medium'], thresholds['high']], 'color': COLORS['success']}
            ],
            'threshold': {
                'line': {'color': COLORS['text'], 'width': 4},
                'thickness': 0.75,
                'value': thresholds['high'] * 0.9
            }
        }
    ))
    
    return apply_theme(fig)

def create_sunburst_chart(df, path_cols, values_col, title="Hierarchical Data"):
    """
    Create sunburst chart for hierarchical data
    
    Args:
        df (pd.DataFrame): Data
        path_cols (list): List of columns for hierarchy path
        values_col (str): Values column
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Sunburst chart figure
    """
    
    fig = px.sunburst(
        df,
        path=path_cols,
        values=values_col,
        title=title,
        color_discrete_sequence=COLOR_PALETTES['green_theme']
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                     'Value: %{value}<br>' +
                     'Percentage: %{percentParent}<br>' +
                     '<extra></extra>',
        maxdepth=3
    )
    
    return apply_theme(fig, title)

def create_violin_plot(df, x_col, y_col, title="Distribution Comparison"):
    """
    Create violin plot for distribution comparison
    
    Args:
        df (pd.DataFrame): Data
        x_col (str): Categorical column
        y_col (str): Numeric column
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Violin plot figure
    """
    
    fig = px.violin(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=x_col,
        color_discrete_sequence=COLOR_PALETTES['green_theme'],
        box=True
    )
    
    fig.update_traces(
        meanline_visible=True,
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        showlegend=False
    )
    
    return apply_theme(fig, title)

# Export all functions
__all__ = [
    'apply_theme',
    'create_salary_distribution_histogram',
    'create_experience_salary_boxplot',
    'create_correlation_heatmap',
    'create_scatter_plot',
    'create_pie_chart',
    'create_bar_chart',
    'create_line_chart',
    'create_gauge_chart',
    'create_sunburst_chart',
    'create_violin_plot',
    'COLORS',
    'COLOR_PALETTES',
    'DEFAULT_LAYOUT'
]