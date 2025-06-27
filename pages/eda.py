"""
EDA & Visualisasi Page
Interactive exploratory data analysis with filters
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, clientside_callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def layout(df, colors):
    """Create EDA page layout"""
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-chart-line me-3", style={'color': colors['primary']}),
                            "EDA & Visualisasi"
                        ], className="display-5 fw-bold mb-3"),
                        html.P("Eksplorasi data interaktif dengan filter dan visualisasi mendalam",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Filter Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-filter me-2"),
                                "Filter Data"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Experience Level:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='eda-experience-filter',
                                        options=[{'label': 'All', 'value': 'all'}] + 
                                               [{'label': exp_mapping(x), 'value': x} 
                                                for x in sorted(df['experience_level'].unique())],
                                        value='all',
                                        clearable=False,
                                        style={'borderRadius': '8px'}
                                    )
                                ], md=3),
                                dbc.Col([
                                    html.Label("Company Size:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='eda-company-size-filter',
                                        options=[{'label': 'All', 'value': 'all'}] + 
                                               [{'label': size_mapping(x), 'value': x} 
                                                for x in sorted(df['company_size'].unique())],
                                        value='all',
                                        clearable=False,
                                        style={'borderRadius': '8px'}
                                    )
                                ], md=3),
                                dbc.Col([
                                    html.Label("Employment Type:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='eda-employment-filter',
                                        options=[{'label': 'All', 'value': 'all'}] + 
                                               [{'label': emp_mapping(x), 'value': x} 
                                                for x in sorted(df['employment_type'].unique())],
                                        value='all',
                                        clearable=False,
                                        style={'borderRadius': '8px'}
                                    )
                                ], md=3),
                                dbc.Col([
                                    html.Label("Salary Range (USD):", className="fw-bold mb-2"),
                                    dcc.RangeSlider(
                                        id='eda-salary-range-filter',
                                        min=df['salary_in_usd'].min(),
                                        max=df['salary_in_usd'].max(),
                                        value=[df['salary_in_usd'].min(), df['salary_in_usd'].max()],
                                        marks={
                                            df['salary_in_usd'].min(): f'${df["salary_in_usd"].min()/1000:.0f}K',
                                            df['salary_in_usd'].max(): f'${df["salary_in_usd"].max()/1000:.0f}K'
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=3)
                            ])
                        ])
                    ], className="shadow-sm border-0 mb-4")
                ])
            ]),
            
            # Summary Stats after filtering
            dbc.Row([
                dbc.Col([
                    html.Div(id='eda-filtered-stats', className="mb-4")
                ])
            ]),
            
            # Main Visualizations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Salary Distribution"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='eda-salary-histogram')
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-box me-2"),
                                "Salary by Experience Level"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='eda-salary-boxplot')
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-area me-2"),
                                "Scatter Plot: Remote Ratio vs Salary"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='eda-scatter-plot')
                        ])
                    ], className="shadow-sm border-0")
                ], md=8, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Top Job Titles"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='eda-job-titles-chart')
                        ])
                    ], className="shadow-sm border-0")
                ], md=4, className="mb-4")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-fire me-2"),
                                "Correlation Heatmap"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='eda-correlation-heatmap')
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-globe me-2"),
                                "Average Salary by Country"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='eda-country-salary-chart')
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ])
            
        ], fluid=True, className="py-4")
    ])

# Helper functions for mapping
def exp_mapping(exp):
    mapping = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
    return mapping.get(exp, exp)

def size_mapping(size):
    mapping = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    return mapping.get(size, size)

def emp_mapping(emp):
    mapping = {'FT': 'Full Time', 'PT': 'Part Time', 'CT': 'Contract', 'FL': 'Freelance'}
    return mapping.get(emp, emp)

# Callbacks for interactivity - Split into multiple callbacks to avoid duplicates
@callback(
    Output('eda-filtered-stats', 'children'),
    [Input('eda-experience-filter', 'value'),
     Input('eda-company-size-filter', 'value'),
     Input('eda-employment-filter', 'value'),
     Input('eda-salary-range-filter', 'value')]
)
def update_filtered_stats(exp_level, company_size, emp_type, salary_range):
    from utils.data_loader import load_data
    df = load_data()
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'light': '#a7f3d0',
        'dark': '#059669'
    }
    
    # Filter data
    filtered_df = df.copy()
    
    if exp_level != 'all':
        filtered_df = filtered_df[filtered_df['experience_level'] == exp_level]
    if company_size != 'all':
        filtered_df = filtered_df[filtered_df['company_size'] == company_size]
    if emp_type != 'all':
        filtered_df = filtered_df[filtered_df['employment_type'] == emp_type]
    
    filtered_df = filtered_df[
        (filtered_df['salary_in_usd'] >= salary_range[0]) & 
        (filtered_df['salary_in_usd'] <= salary_range[1])
    ]
    
    return create_filtered_stats(filtered_df, colors)

@callback(
    [Output('eda-salary-histogram', 'figure'),
     Output('eda-salary-boxplot', 'figure')],
    [Input('eda-experience-filter', 'value'),
     Input('eda-company-size-filter', 'value'),
     Input('eda-employment-filter', 'value'),
     Input('eda-salary-range-filter', 'value')]
)
def update_salary_charts(exp_level, company_size, emp_type, salary_range):
    from utils.data_loader import load_data
    df = load_data()
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'light': '#a7f3d0',
        'dark': '#059669'
    }
    
    # Filter data
    filtered_df = df.copy()
    
    if exp_level != 'all':
        filtered_df = filtered_df[filtered_df['experience_level'] == exp_level]
    if company_size != 'all':
        filtered_df = filtered_df[filtered_df['company_size'] == company_size]
    if emp_type != 'all':
        filtered_df = filtered_df[filtered_df['employment_type'] == emp_type]
    
    filtered_df = filtered_df[
        (filtered_df['salary_in_usd'] >= salary_range[0]) & 
        (filtered_df['salary_in_usd'] <= salary_range[1])
    ]
    
    histogram = create_histogram(filtered_df, colors)
    boxplot = create_boxplot(filtered_df, colors)
    
    return histogram, boxplot

@callback(
    [Output('eda-scatter-plot', 'figure'),
     Output('eda-job-titles-chart', 'figure')],
    [Input('eda-experience-filter', 'value'),
     Input('eda-company-size-filter', 'value'),
     Input('eda-employment-filter', 'value'),
     Input('eda-salary-range-filter', 'value')]
)
def update_scatter_and_jobs(exp_level, company_size, emp_type, salary_range):
    from utils.data_loader import load_data
    df = load_data()
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'light': '#a7f3d0',
        'dark': '#059669'
    }
    
    # Filter data
    filtered_df = df.copy()
    
    if exp_level != 'all':
        filtered_df = filtered_df[filtered_df['experience_level'] == exp_level]
    if company_size != 'all':
        filtered_df = filtered_df[filtered_df['company_size'] == company_size]
    if emp_type != 'all':
        filtered_df = filtered_df[filtered_df['employment_type'] == emp_type]
    
    filtered_df = filtered_df[
        (filtered_df['salary_in_usd'] >= salary_range[0]) & 
        (filtered_df['salary_in_usd'] <= salary_range[1])
    ]
    
    scatter = create_scatter(filtered_df, colors)
    job_chart = create_job_titles_chart(filtered_df, colors)
    
    return scatter, job_chart

@callback(
    [Output('eda-correlation-heatmap', 'figure'),
     Output('eda-country-salary-chart', 'figure')],
    [Input('eda-experience-filter', 'value'),
     Input('eda-company-size-filter', 'value'),
     Input('eda-employment-filter', 'value'),
     Input('eda-salary-range-filter', 'value')]
)
def update_correlation_and_country(exp_level, company_size, emp_type, salary_range):
    from utils.data_loader import load_data
    df = load_data()
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'light': '#a7f3d0',
        'dark': '#059669'
    }
    
    # Filter data
    filtered_df = df.copy()
    
    if exp_level != 'all':
        filtered_df = filtered_df[filtered_df['experience_level'] == exp_level]
    if company_size != 'all':
        filtered_df = filtered_df[filtered_df['company_size'] == company_size]
    if emp_type != 'all':
        filtered_df = filtered_df[filtered_df['employment_type'] == emp_type]
    
    filtered_df = filtered_df[
        (filtered_df['salary_in_usd'] >= salary_range[0]) & 
        (filtered_df['salary_in_usd'] <= salary_range[1])
    ]
    
    heatmap = create_correlation_heatmap(filtered_df, colors)
    country_chart = create_country_chart(filtered_df, colors)
    
    return heatmap, country_chart

def create_filtered_stats(df, colors):
    """Create filtered statistics cards"""
    return dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H6([
                    html.I(className="fas fa-filter me-2"),
                    f"Filtered Results: {len(df):,} records"
                ], className="mb-2"),
                html.P([
                    f"Average Salary: ${df['salary_in_usd'].mean():,.0f} | ",
                    f"Median: ${df['salary_in_usd'].median():,.0f} | ",
                    f"Range: ${df['salary_in_usd'].min():,.0f} - ${df['salary_in_usd'].max():,.0f}"
                ], className="mb-0")
            ], color="primary", className="border-0", 
               style={'backgroundColor': colors['light'] + '40'})
        ])
    ])

def create_histogram(df, colors):
    """Create salary distribution histogram"""
    fig = px.histogram(
        df, x='salary_in_usd', nbins=30,
        title='Salary Distribution',
        color_discrete_sequence=[colors['primary']]
    )
    
    fig.add_vline(
        x=df['salary_in_usd'].mean(), 
        line_dash="dash", 
        line_color=colors['dark'],
        annotation_text=f"Mean: ${df['salary_in_usd'].mean():,.0f}"
    )
    
    fig.update_layout(
        xaxis_title="Salary (USD)",
        yaxis_title="Frequency",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_boxplot(df, colors):
    """Create salary boxplot by experience level"""
    fig = px.box(
        df, x='experience_level', y='salary_in_usd',
        title='Salary Distribution by Experience Level',
        color='experience_level',
        color_discrete_sequence=[colors['primary'], colors['secondary'], 
                               colors['accent'], colors['dark']]
    )
    
    fig.update_xaxis(
        ticktext=['Entry Level', 'Mid Level', 'Senior Level', 'Executive Level'],
        tickvals=['EN', 'MI', 'SE', 'EX']
    )
    
    fig.update_layout(
        xaxis_title="Experience Level",
        yaxis_title="Salary (USD)",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_scatter(df, colors):
    """Create scatter plot of remote ratio vs salary"""
    fig = px.scatter(
        df, x='remote_ratio', y='salary_in_usd',
        color='experience_level', size='salary_in_usd',
        title='Remote Work vs Salary by Experience Level',
        color_discrete_sequence=[colors['primary'], colors['secondary'], 
                               colors['accent'], colors['dark']],
        hover_data=['job_title', 'company_location']
    )
    
    fig.update_layout(
        xaxis_title="Remote Work Ratio (%)",
        yaxis_title="Salary (USD)",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_job_titles_chart(df, colors):
    """Create top job titles chart"""
    top_jobs = df['job_title'].value_counts().head(8)
    
    fig = px.bar(
        x=top_jobs.values, y=top_jobs.index,
        orientation='h',
        title='Top Job Titles',
        color=top_jobs.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Job Title",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(df, colors):
    """Create correlation heatmap"""
    # Select numeric columns and encode categorical
    df_encoded = df.copy()
    df_encoded['experience_encoded'] = pd.Categorical(df['experience_level']).codes
    df_encoded['company_size_encoded'] = pd.Categorical(df['company_size']).codes
    df_encoded['employment_encoded'] = pd.Categorical(df['employment_type']).codes
    
    numeric_cols = ['work_year', 'salary_in_usd', 'remote_ratio', 
                   'experience_encoded', 'company_size_encoded', 'employment_encoded']
    corr_matrix = df_encoded[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale=[[0, '#ffffff'], [0.5, colors['light']], [1, colors['primary']]],
        title='Correlation Matrix',
        aspect='auto'
    )
    
    fig.update_layout(
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_country_chart(df, colors):
    """Create average salary by country chart"""
    country_salary = (df.groupby('company_location')['salary_in_usd']
                     .mean()
                     .sort_values(ascending=True)
                     .tail(10))
    
    fig = px.bar(
        x=country_salary.values, y=country_salary.index,
        orientation='h',
        title='Top 10 Countries by Average Salary',
        color=country_salary.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_layout(
        xaxis_title="Average Salary (USD)",
        yaxis_title="Country",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig