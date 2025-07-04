"""
EDA & Visualisasi Page
Static exploratory data analysis with comprehensive visualizations
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def layout(df, colors):
    """Create EDA page layout with static visualizations"""
    
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
                        html.P("Eksplorasi data komprehensif dengan visualisasi interaktif dan analisis mendalam",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Summary Statistics Cards
            dbc.Row([
                dbc.Col([
                    create_summary_card(
                        "📊", f"{len(df):,}",
                        "Total Records", colors['primary']
                    )
                ], md=3, className="mb-4"),
                dbc.Col([
                    create_summary_card(
                        "💰", f"${df['salary_in_usd'].mean():,.0f}",
                        "Average Salary", colors['secondary']
                    )
                ], md=3, className="mb-4"),
                dbc.Col([
                    create_summary_card(
                        "📈", f"${df['salary_in_usd'].median():,.0f}",
                        "Median Salary", colors['accent']
                    )
                ], md=3, className="mb-4"),
                dbc.Col([
                    create_summary_card(
                        "🌍", str(df['company_location'].nunique()),
                        "Countries", colors['dark']
                    )
                ], md=3, className="mb-4")
            ]),
            
            # First Row of Visualizations
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
                            dcc.Graph(
                                figure=create_salary_histogram(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Experience Level Distribution"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_experience_pie_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            # Second Row of Visualizations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-box me-2"),
                                "Salary by Experience Level"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_salary_boxplot(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=8, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-building me-2"),
                                "Company Size"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_company_size_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=4, className="mb-4")
            ]),
            
            # Third Row of Visualizations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-area me-2"),
                                "Remote Work vs Salary Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_remote_scatter_plot(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=8, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-briefcase me-2"),
                                "Employment Types"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_employment_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=4, className="mb-4")
            ]),
            
            # Fourth Row of Visualizations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-globe me-2"),
                                "Top 15 Countries by Average Salary"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_top_countries_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-user-tie me-2"),
                                "Top 10 Job Titles"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_top_jobs_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            # Fifth Row - Advanced Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-fire me-2"),
                                "Correlation Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_correlation_heatmap(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-line me-2"),
                                "Salary Trends by Year"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_salary_trends_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            # Key Insights Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-lightbulb me-2"),
                                "Key Insights from EDA"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_key_insights_section(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_summary_card(icon, value, label, color):
    """Create summary statistics card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, className="fs-2 mb-2"),
                html.H3(value, className="fw-bold mb-1", style={'color': color}),
                html.P(label, className="mb-0 text-muted fw-medium")
            ], className="text-center")
        ])
    ], className="h-100 border-0 shadow-sm card-hover",
       style={
           'background': f'linear-gradient(145deg, #ffffff, #f8fffe)',
           'borderLeft': f'4px solid {color}'
       })

def create_salary_histogram(df, colors):
    """Create salary distribution histogram"""
    fig = px.histogram(
        df, x='salary_in_usd', nbins=35,
        title='Distribution of Salaries (USD)',
        color_discrete_sequence=[colors['primary']]
    )
    
    # Add mean and median lines
    mean_salary = df['salary_in_usd'].mean()
    median_salary = df['salary_in_usd'].median()
    
    fig.add_vline(
        x=mean_salary, 
        line_dash="dash", 
        line_color=colors['dark'],
        annotation_text=f"Mean: ${mean_salary:,.0f}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=median_salary, 
        line_dash="dot", 
        line_color=colors['secondary'],
        annotation_text=f"Median: ${median_salary:,.0f}",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        xaxis_title="Salary (USD)",
        yaxis_title="Frequency",
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(
        hovertemplate='<b>Salary Range:</b> $%{x:,.0f}<br><b>Count:</b> %{y}<extra></extra>'
    )
    
    return fig

def create_experience_pie_chart(df, colors):
    """Create experience level pie chart"""
    exp_counts = df['experience_level'].value_counts()
    exp_labels = {
        'EN': 'Entry Level',
        'MI': 'Mid Level', 
        'SE': 'Senior Level',
        'EX': 'Executive Level'
    }
    
    fig = px.pie(
        values=exp_counts.values,
        names=[exp_labels.get(x, x) for x in exp_counts.index],
        title='Experience Level Distribution',
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], colors['dark']]
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_salary_boxplot(df, colors):
    """Create salary boxplot by experience level"""
    fig = px.box(
        df, x='experience_level', y='salary_in_usd',
        title='Salary Distribution by Experience Level',
        color='experience_level',
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], colors['dark']]
    )
    
    # Update x-axis labels - FIXED: using update_xaxes instead of update_xaxis
    fig.update_xaxes(
        ticktext=['Entry Level', 'Mid Level', 'Senior Level', 'Executive Level'],
        tickvals=['EN', 'MI', 'SE', 'EX']
    )
    
    fig.update_layout(
        xaxis_title="Experience Level",
        yaxis_title="Salary (USD)",
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_company_size_chart(df, colors):
    """Create company size distribution chart"""
    size_counts = df['company_size'].value_counts()
    size_labels = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    
    fig = px.bar(
        x=[size_labels.get(x, x) for x in size_counts.index],
        y=size_counts.values,
        title='Company Size Distribution',
        color=size_counts.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_layout(
        xaxis_title="Company Size",
        yaxis_title="Count",
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )
    
    return fig

def create_remote_scatter_plot(df, colors):
    """Create scatter plot of remote ratio vs salary"""
    fig = px.scatter(
        df, x='remote_ratio', y='salary_in_usd',
        color='experience_level', 
        size='salary_in_usd',
        size_max=15,
        title='Remote Work Ratio vs Salary by Experience Level',
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], colors['dark']],
        hover_data=['job_title', 'company_location']
    )
    
    fig.update_layout(
        xaxis_title="Remote Work Ratio (%)",
        yaxis_title="Salary (USD)",
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_employment_chart(df, colors):
    """Create employment type distribution chart"""
    emp_counts = df['employment_type'].value_counts()
    emp_labels = {'FT': 'Full Time', 'PT': 'Part Time', 'CT': 'Contract', 'FL': 'Freelance'}
    
    fig = px.bar(
        x=emp_counts.values,
        y=[emp_labels.get(x, x) for x in emp_counts.index],
        orientation='h',
        title='Employment Types',
        color=emp_counts.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Employment Type",
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_top_countries_chart(df, colors):
    """Create top countries by average salary chart"""
    country_salary = (df.groupby('company_location')['salary_in_usd']
                     .agg(['mean', 'count'])
                     .reset_index())
    
    # Filter countries with at least 5 records
    country_salary = country_salary[country_salary['count'] >= 5]
    country_salary = country_salary.sort_values('mean', ascending=True).tail(15)
    
    fig = px.bar(
        x=country_salary['mean'], 
        y=country_salary['company_location'],
        orientation='h',
        title='Top 15 Countries by Average Salary (min 5 records)',
        color=country_salary['mean'],
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_layout(
        xaxis_title="Average Salary (USD)",
        yaxis_title="Country",
        height=500,
        margin=dict(t=50, b=40, l=80, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Avg Salary: $%{x:,.0f}<extra></extra>'
    )
    
    return fig

def create_top_jobs_chart(df, colors):
    """Create top job titles chart"""
    top_jobs = df['job_title'].value_counts().head(10)
    
    fig = px.bar(
        x=top_jobs.values, 
        y=top_jobs.index,
        orientation='h',
        title='Top 10 Most Common Job Titles',
        color=top_jobs.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Job Title",
        height=450,
        margin=dict(t=50, b=40, l=200, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )
    
    return fig

def create_correlation_heatmap(df, colors):
    """Create correlation heatmap for numerical variables"""
    try:
        # Select and encode relevant columns
        df_encoded = df.copy()
        df_encoded['experience_encoded'] = pd.Categorical(df['experience_level'], 
                                                        categories=['EN', 'MI', 'SE', 'EX']).codes
        df_encoded['company_size_encoded'] = pd.Categorical(df['company_size'], 
                                                           categories=['S', 'M', 'L']).codes
        df_encoded['employment_encoded'] = pd.Categorical(df['employment_type']).codes
        
        # Select numerical columns for correlation
        numeric_cols = ['work_year', 'salary_in_usd', 'remote_ratio', 
                       'experience_encoded', 'company_size_encoded', 'employment_encoded']
        
        corr_matrix = df_encoded[numeric_cols].corr()
        
        # Create readable labels
        labels = {
            'work_year': 'Work Year',
            'salary_in_usd': 'Salary (USD)',
            'remote_ratio': 'Remote Ratio',
            'experience_encoded': 'Experience Level',
            'company_size_encoded': 'Company Size',
            'employment_encoded': 'Employment Type'
        }
        
        # Rename correlation matrix
        corr_matrix.index = [labels[col] for col in corr_matrix.index]
        corr_matrix.columns = [labels[col] for col in corr_matrix.columns]
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale=[[0, '#ffffff'], [0.5, colors['light']], [1, colors['primary']]],
            title='Correlation Matrix of Key Variables',
            aspect='auto',
            text_auto=True
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        # Fallback simple chart if correlation fails
        fig = px.bar(
            x=['Data Processing'], y=[1],
            title='Correlation Analysis',
            text=['Correlation matrix calculation in progress...']
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

def create_salary_trends_chart(df, colors):
    """Create salary trends by year and experience level"""
    try:
        yearly_salary = df.groupby(['work_year', 'experience_level'])['salary_in_usd'].mean().reset_index()
        
        exp_labels = {
            'EN': 'Entry Level',
            'MI': 'Mid Level', 
            'SE': 'Senior Level',
            'EX': 'Executive Level'
        }
        
        yearly_salary['experience_label'] = yearly_salary['experience_level'].map(exp_labels)
        
        fig = px.line(
            yearly_salary, 
            x='work_year', 
            y='salary_in_usd',
            color='experience_label',
            title='Average Salary Trends by Year and Experience Level',
            color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], colors['dark']],
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Salary (USD)",
            height=400,
            margin=dict(t=50, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend_title="Experience Level"
        )
        
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=8)
        )
        
        return fig
        
    except Exception as e:
        # Fallback simple chart
        fig = px.bar(
            x=df['work_year'].value_counts().index,
            y=df['work_year'].value_counts().values,
            title='Data Records by Year'
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

def create_key_insights_section(df, colors):
    """Create key insights from EDA"""
    # Calculate insights
    avg_salary = df['salary_in_usd'].mean()
    max_salary = df['salary_in_usd'].max()
    top_job = df['job_title'].value_counts().index[0]
    top_country = df['company_location'].value_counts().index[0]
    remote_avg = df['remote_ratio'].mean()
    
    # Experience level salary comparison
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean()
    senior_premium = exp_salary['SE'] / exp_salary['EN'] if 'EN' in exp_salary and 'SE' in exp_salary else 1
    
    insights = [
        {
            'icon': '💰',
            'title': 'Salary Range Analysis',
            'description': f'Average salary is ${avg_salary:,.0f} with maximum reaching ${max_salary:,.0f}. Clear salary progression across experience levels.',
            'highlight': f'${avg_salary:,.0f} average'
        },
        {
            'icon': '📈',
            'title': 'Experience Premium',
            'description': f'Senior level professionals earn {senior_premium:.1f}x more than entry level, showing strong career growth potential.',
            'highlight': f'{senior_premium:.1f}x multiplier'
        },
        {
            'icon': '🌟',
            'title': 'Most In-Demand Role',
            'description': f'{top_job} is the most common position with {df[df["job_title"] == top_job].shape[0]} openings, indicating high market demand.',
            'highlight': f'{df[df["job_title"] == top_job].shape[0]} positions'
        },
        {
            'icon': '🌍',
            'title': 'Geographic Distribution',
            'description': f'{top_country} leads with {df[df["company_location"] == top_country].shape[0]} companies, representing major tech hub concentration.',
            'highlight': f'{df[df["company_location"] == top_country].shape[0]} companies'
        },
        {
            'icon': '🏠',
            'title': 'Remote Work Trends',
            'description': f'Average remote work ratio is {remote_avg:.0f}%, indicating flexible work arrangements are common in data science.',
            'highlight': f'{remote_avg:.0f}% remote'
        },
        {
            'icon': '📊',
            'title': 'Market Diversity',
            'description': f'Dataset spans {df["company_location"].nunique()} countries and {df["job_title"].nunique()} unique job titles, showing global market diversity.',
            'highlight': f'{df["company_location"].nunique()} countries'
        }
    ]
    
    insight_cards = []
    for i, insight in enumerate(insights):
        insight_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span(insight['icon'], className="fs-2 me-3"),
                                html.Div([
                                    html.H6(insight['title'], className="fw-bold mb-2"),
                                    html.P(insight['description'], className="mb-2 small"),
                                    dbc.Badge(insight['highlight'], color="primary", className="px-3 py-2")
                                ])
                            ], className="d-flex align-items-start")
                        ])
                    ])
                ], className="h-100 shadow-sm border-0 card-hover",
                   style={'borderLeft': f'4px solid {colors["primary"]}'})
            ], md=6, className="mb-3")
        )
    
    return dbc.Row(insight_cards)