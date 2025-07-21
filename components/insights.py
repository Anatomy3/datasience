"""
Insights and Charts Components
Handles data visualization and insights sections
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go

def create_insights_section(df, colors):
    """Create data insights section with charts"""
    return html.Div([
        # Section Header
        html.Div([
            html.H3("Data Insights", className="text-center mb-2", style={'color': colors['darker']}),
            html.P("Visualisasi tren gaji data science global", 
                   className="text-center text-muted mb-4")
        ]),
        
        # Charts Grid - Compact Layout
        dbc.Row([
            # Chart 1: Experience Analysis
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-user-graduate me-2", style={'color': colors['primary']}),
                            "Experience Levels"
                        ], className="mb-2"),
                        dcc.Graph(
                            figure=create_experience_chart(df, colors),
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ], className="p-3")
                ], className="shadow-sm border-0 mb-3")
            ], md=4),
            
            # Chart 2: Global Salary 
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-globe-americas me-2", style={'color': colors['secondary']}),
                            "Top Countries"
                        ], className="mb-2"),
                        dcc.Graph(
                            figure=create_country_chart(df, colors),
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ], className="p-3")
                ], className="shadow-sm border-0 mb-3")
            ], md=4),
            
            # Chart 3: Job Trends
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-briefcase me-2", style={'color': colors['accent']}),
                            "Popular Jobs"
                        ], className="mb-2"),
                        dcc.Graph(
                            figure=create_job_trends_chart(df, colors),
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ], className="p-3")
                ], className="shadow-sm border-0 mb-3")
            ], md=4)
        ])
    ], className="py-4")

def create_experience_chart(df, colors):
    """Create experience level distribution chart"""
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
        color_discrete_sequence=[colors['primary'], colors['secondary'], 
                               colors['accent'], colors['dark']]
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10)
    )
    
    return fig

def create_country_chart(df, colors):
    """Create top countries by salary chart"""
    top_countries = (df.groupby('company_location')['salary_in_usd']
                    .mean()
                    .sort_values(ascending=True)
                    .tail(10))
    
    fig = px.bar(
        x=top_countries.values,
        y=top_countries.index,
        orientation='h',
        color=top_countries.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Avg Salary: $%{x:,.0f}<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
        font=dict(size=10)
    )
    
    return fig

def create_job_trends_chart(df, colors):
    """Create job trends/categories chart"""
    top_jobs = df['job_title'].value_counts().head(8)
    
    fig = px.bar(
        x=top_jobs.values,
        y=top_jobs.index,
        orientation='h',
        color=top_jobs.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['accent']]]
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
        font=dict(size=10)
    )
    
    return fig

def create_key_insights(df, colors):
    """Create key insights section"""
    avg_salary = df['salary_in_usd'].mean()
    max_salary = df['salary_in_usd'].max()
    min_salary = df['salary_in_usd'].min()
    
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean()
    top_job = df['job_title'].value_counts().index[0]
    top_country = df['company_location'].value_counts().index[0]
    
    insights = [
        {
            'icon': '💰',
            'title': 'Rentang Gaji',
            'text': f'Gaji berkisar dari ${min_salary:,.0f} hingga ${max_salary:,.0f} dengan rata-rata ${avg_salary:,.0f}'
        },
        {
            'icon': '📈',
            'title': 'Premium Pengalaman',
            'text': f'Level senior mendapat gaji {(exp_salary.get("SE", 0) / exp_salary.get("EN", 1)):.1f}x lebih tinggi dari level pemula'
        },
        {
            'icon': '🌟',
            'title': 'Jabatan Teratas',
            'text': f'{top_job} adalah posisi paling umum dengan {df[df["job_title"] == top_job].shape[0]} lowongan'
        },
        {
            'icon': '🌍',
            'title': 'Negara Terkemuka',
            'text': f'{top_country} memimpin dengan {df[df["company_location"] == top_country].shape[0]} perusahaan'
        }
    ]
    
    insight_cards = []
    for insight in insights:
        insight_cards.append(
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span(insight['icon'], className="fs-3 me-3"),
                        html.Div([
                            html.H6(insight['title'], className="fw-bold mb-1"),
                            html.P(insight['text'], className="mb-0 small")
                        ])
                    ], className="d-flex align-items-center")
                ], className="p-3 rounded border-start border-3",
                   style={'borderColor': colors['primary'] + '!important',
                          'backgroundColor': colors['light'] + '20'})
            ], md=6, className="mb-3")
        )
    
    return dbc.Row(insight_cards)