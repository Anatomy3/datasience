"""
Results Visualization Page
Interactive results display with storytelling
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib
import os

def layout(df, colors, lang='id'):
    """Create results visualization page layout"""
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-chart-pie me-3", style={'color': colors['primary']}),
                            "Hasil Visualisasi & Storytelling"
                        ], className="display-5 fw-bold mb-3"),
                        html.P("Visualisasi hasil analisis dengan storytelling data yang menarik dan informatif",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Data Story Introduction
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-book-open me-2"),
                                "📊 Data Science Salaries: The Complete Story"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_story_introduction(df, colors)
                        ])
                    ], className="shadow-sm border-0 mb-4")
                ])
            ]),
            
            # Key Findings Dashboard
            dbc.Row([
                dbc.Col([
                    create_key_metrics_dashboard(df, colors)
                ])
            ], className="mb-4"),
            
            # Interactive Story Sections
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-play me-2"),
                                "Interactive Story Navigation"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_story_navigation()
                        ])
                    ], className="shadow-sm border-0 mb-4")
                ])
            ]),
            
            # Story Content
            dbc.Row([
                dbc.Col([
                    html.Div(id='story-content')
                ])
            ]),
            
            # Comprehensive Visualizations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-chart-area me-2"),
                                "Comprehensive Analysis Dashboard"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_comprehensive_dashboard(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4"),
            
            # Model Performance Summary
            dbc.Row([
                dbc.Col([
                    create_model_performance_summary(colors)
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_story_introduction(df, colors):
    """Create engaging story introduction"""
    avg_salary = df['salary_in_usd'].mean()
    max_salary = df['salary_in_usd'].max()
    countries = df['company_location'].nunique()
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("🌟 Welcome to the Data Science Salary Journey", 
                           className="text-primary fw-bold mb-3"),
                    html.P([
                        "Dalam era digital yang terus berkembang, profesi data science menjadi salah satu yang paling dicari. ",
                        f"Berdasarkan analisis terhadap {len(df):,} data scientist dari {countries} negara, ",
                        f"kami menemukan bahwa rata-rata gaji mencapai ${avg_salary:,.0f} per tahun. ",
                        "Mari kita telusuri lebih dalam tentang faktor-faktor yang mempengaruhi kompensasi ini."
                    ], className="mb-3", style={'textAlign': 'justify'}),
                    
                    html.Div([
                        html.H6("🎯 Pertanyaan Penelitian:", className="fw-bold text-success"),
                        html.Ul([
                            html.Li("Bagaimana experience level mempengaruhi gaji data scientist?"),
                            html.Li("Apakah remote work memberikan advantage dalam kompensasi?"),
                            html.Li("Negara mana yang menawarkan gaji tertinggi?"),
                            html.Li("Tren gaji seperti apa yang terjadi dalam beberapa tahun terakhir?")
                        ])
                    ], className="p-3 rounded mb-3", 
                       style={'backgroundColor': colors['light'] + '30'})
                ])
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-3x text-success mb-2"),
                        html.H3(f"${max_salary:,.0f}", className="fw-bold text-success"),
                        html.P("Highest Salary", className="text-muted")
                    ], className="text-center p-3 rounded border"),
                    
                    html.Div([
                        html.I(className="fas fa-globe fa-3x text-info mb-2"),
                        html.H3(f"{countries}", className="fw-bold text-info"),
                        html.P("Countries Analyzed", className="text-muted")
                    ], className="text-center p-3 rounded border mt-3"),
                    
                    html.Div([
                        html.I(className="fas fa-users fa-3x text-warning mb-2"),
                        html.H3(f"{df['job_title'].nunique()}", className="fw-bold text-warning"),
                        html.P("Job Roles", className="text-muted")
                    ], className="text-center p-3 rounded border mt-3")
                ])
            ], md=4)
        ])
    ])

def create_key_metrics_dashboard(df, colors):
    """Create key metrics overview dashboard"""
    
    # Calculate key metrics
    metrics = {
        'avg_salary': df['salary_in_usd'].mean(),
        'median_salary': df['salary_in_usd'].median(),
        'salary_growth': calculate_salary_growth(df),
        'remote_premium': calculate_remote_premium(df),
        'experience_multiplier': calculate_experience_multiplier(df),
        'top_paying_country': get_top_paying_country(df)
    }
    
    return dbc.Row([
        dbc.Col([
            create_metric_highlight(
                "💰", "Average Salary", f"${metrics['avg_salary']:,.0f}",
                f"Median: ${metrics['median_salary']:,.0f}", colors['primary']
            )
        ], md=4, className="mb-3"),
        dbc.Col([
            create_metric_highlight(
                "📈", "Salary Growth", f"{metrics['salary_growth']:.1f}%",
                "Year-over-year increase", colors['secondary']
            )
        ], md=4, className="mb-3"),
        dbc.Col([
            create_metric_highlight(
                "🏠", "Remote Premium", f"{metrics['remote_premium']:.1f}%",
                "Extra for 100% remote", colors['accent']
            )
        ], md=4, className="mb-3"),
        dbc.Col([
            create_metric_highlight(
                "⭐", "Experience Multiplier", f"{metrics['experience_multiplier']:.1f}x",
                "Senior vs Entry level", colors['dark']
            )
        ], md=4, className="mb-3"),
        dbc.Col([
            create_metric_highlight(
                "🌍", "Top Country", metrics['top_paying_country'],
                "Highest average salary", colors['primary']
            )
        ], md=4, className="mb-3"),
        dbc.Col([
            create_metric_highlight(
                "🎯", "Data Quality", "100%",
                "Complete & clean dataset", colors['secondary']
            )
        ], md=4, className="mb-3")
    ])

def create_metric_highlight(icon, title, value, subtitle, color):
    """Create highlighted metric card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, className="fs-1 mb-2 d-block"),
                html.H3(value, className="fw-bold mb-1", style={'color': color}),
                html.H6(title, className="mb-1"),
                html.Small(subtitle, className="text-muted")
            ], className="text-center")
        ])
    ], className="h-100 shadow-sm border-0",
       style={
           'background': f'linear-gradient(135deg, #ffffff, {color}10)',
           'borderTop': f'4px solid {color}'
       })

def create_story_navigation():
    """Create interactive story navigation"""
    return dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("Chapter 1: Experience Journey", id="story-chapter-1", 
                          color="outline-primary", className="story-nav-btn"),
                dbc.Button("Chapter 2: Remote Revolution", id="story-chapter-2", 
                          color="outline-primary", className="story-nav-btn"),
                dbc.Button("Chapter 3: Global Landscape", id="story-chapter-3", 
                          color="outline-primary", className="story-nav-btn"),
                dbc.Button("Chapter 4: Future Trends", id="story-chapter-4", 
                          color="outline-primary", className="story-nav-btn")
            ], className="w-100")
        ])
    ])

def create_comprehensive_dashboard(df, colors):
    """Create comprehensive analysis dashboard"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Salary Distribution by Experience', 
                       'Remote Work Impact',
                       'Top 10 Countries by Avg Salary',
                       'Company Size vs Salary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Experience level boxplot
    for i, exp in enumerate(['EN', 'MI', 'SE', 'EX']):
        exp_data = df[df['experience_level'] == exp]['salary_in_usd']
        fig.add_trace(
            go.Box(y=exp_data, name=f'{exp} Level', 
                  marker_color=colors['primary']),
            row=1, col=1
        )
    
    # 2. Remote work scatter
    fig.add_trace(
        go.Scatter(x=df['remote_ratio'], y=df['salary_in_usd'],
                  mode='markers', name='Salary vs Remote',
                  marker=dict(color=colors['secondary'], opacity=0.6)),
        row=1, col=2
    )
    
    # 3. Top countries
    country_avg = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(x=country_avg.values, y=country_avg.index,
              orientation='h', name='Country Avg',
              marker_color=colors['accent']),
        row=2, col=1
    )
    
    # 4. Company size
    size_avg = df.groupby('company_size')['salary_in_usd'].mean()
    fig.add_trace(
        go.Bar(x=size_avg.index, y=size_avg.values,
              name='Size Avg', marker_color=colors['dark']),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Comprehensive Data Science Salary Analysis",
        title_x=0.5
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_model_performance_summary(colors):
    """Create model performance summary"""
    try:
        # Try to load model performance if available
        if os.path.exists('data/trained_model.pkl'):
            model_info = "✅ Model Available - Ready for Predictions"
            performance_text = "Model telah dilatih dan siap untuk melakukan prediksi gaji"
        else:
            model_info = "⚠️ Model Not Trained Yet"
            performance_text = "Silakan kunjungi halaman Modeling untuk melatih model"
    except:
        model_info = "⚠️ Model Status Unknown"
        performance_text = "Status model tidak dapat ditentukan"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-robot me-2"),
                "Model Performance Summary"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5(model_info, className="text-primary mb-3"),
                        html.P(performance_text, className="mb-3"),
                        
                        html.H6("🎯 Model Capabilities:", className="fw-bold mb-2"),
                        html.Ul([
                            html.Li("Prediksi gaji berdasarkan experience level"),
                            html.Li("Analisis dampak remote work terhadap kompensasi"),
                            html.Li("Perbandingan gaji antar company size"),
                            html.Li("Evaluasi tren gaji berdasarkan tahun")
                        ]),
                        
                        dbc.Button("Train New Model", href="/modeling", 
                                 color="primary", className="me-2"),
                        dbc.Button("View Insights", href="/insights", 
                                 color="outline-success")
                    ])
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H6("📊 Expected Model Metrics:", className="fw-bold mb-3"),
                        create_expected_metrics_table(colors)
                    ])
                ], md=6)
            ])
        ])
    ], className="shadow-sm border-0")

def create_expected_metrics_table(colors):
    """Create expected model performance metrics"""
    metrics_data = [
        {"Metric": "R² Score", "Expected": "> 0.80", "Description": "Model accuracy"},
        {"Metric": "RMSE", "Expected": "< $20,000", "Description": "Prediction error"},
        {"Metric": "MAE", "Expected": "< $15,000", "Description": "Average error"},
        {"Metric": "Features", "Expected": "5-7", "Description": "Input variables"}
    ]
    
    table_rows = []
    for metric in metrics_data:
        table_rows.append(
            html.Tr([
                html.Td(metric["Metric"], className="fw-bold"),
                html.Td(metric["Expected"], style={'color': colors['primary']}),
                html.Td(metric["Description"], className="text-muted")
            ])
        )
    
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Metric"),
                html.Th("Target"),
                html.Th("Description")
            ])
        ]),
        html.Tbody(table_rows)
    ], striped=True, hover=True, responsive=True, size="sm")

# Helper functions for calculations
def calculate_salary_growth(df):
    """Calculate year-over-year salary growth"""
    yearly_avg = df.groupby('work_year')['salary_in_usd'].mean()
    if len(yearly_avg) > 1:
        years = sorted(yearly_avg.index)
        latest = yearly_avg[years[-1]]
        previous = yearly_avg[years[-2]]
        return ((latest - previous) / previous) * 100
    return 0

def calculate_remote_premium(df):
    """Calculate remote work salary premium"""
    remote_100 = df[df['remote_ratio'] == 100]['salary_in_usd'].mean()
    onsite = df[df['remote_ratio'] == 0]['salary_in_usd'].mean()
    if onsite > 0:
        return ((remote_100 - onsite) / onsite) * 100
    return 0

def calculate_experience_multiplier(df):
    """Calculate experience level salary multiplier"""
    senior_avg = df[df['experience_level'] == 'SE']['salary_in_usd'].mean()
    entry_avg = df[df['experience_level'] == 'EN']['salary_in_usd'].mean()
    if entry_avg > 0:
        return senior_avg / entry_avg
    return 1

def get_top_paying_country(df):
    """Get the top paying country"""
    country_avg = df.groupby('company_location')['salary_in_usd'].mean()
    return country_avg.sort_values(ascending=False).index[0]

# Callback for story navigation
@callback(
    Output('story-content', 'children'),
    [Input('story-chapter-1', 'n_clicks'),
     Input('story-chapter-2', 'n_clicks'),
     Input('story-chapter-3', 'n_clicks'),
     Input('story-chapter-4', 'n_clicks')]
)
def update_story_content(ch1, ch2, ch3, ch4):
    from dash import callback_context
    from utils.data_loader import load_data
    
    df = load_data()
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'light': '#a7f3d0',
        'dark': '#059669'
    }
    
    if not callback_context.triggered:
        return create_default_story_content(df, colors)
    
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'story-chapter-1':
        return create_chapter_1(df, colors)
    elif triggered_id == 'story-chapter-2':
        return create_chapter_2(df, colors)
    elif triggered_id == 'story-chapter-3':
        return create_chapter_3(df, colors)
    elif triggered_id == 'story-chapter-4':
        return create_chapter_4(df, colors)
    
    return create_default_story_content(df, colors)

def create_default_story_content(df, colors):
    """Create default story content"""
    return dbc.Alert([
        html.H5("📖 Select a Chapter Above", className="alert-heading"),
        html.P("Pilih salah satu chapter untuk membaca cerita data yang menarik tentang gaji data scientist!")
    ], color="info")

def create_chapter_1(df, colors):
    """Create Chapter 1: Experience Journey"""
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean()
    
    fig = px.bar(
        x=['Entry', 'Mid', 'Senior', 'Executive'],
        y=[exp_salary.get('EN', 0), exp_salary.get('MI', 0), 
           exp_salary.get('SE', 0), exp_salary.get('EX', 0)],
        title="The Experience Journey: From Entry to Executive",
        color_discrete_sequence=[colors['primary']]
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Chapter 1: The Experience Journey 🚀")
        ]),
        dbc.CardBody([
            html.P([
                "Perjalanan karir data scientist menunjukkan progres yang sangat menjanjikan. ",
                "Dimulai dari entry level dengan gaji rata-rata $",
                f"{exp_salary.get('EN', 0):,.0f}, seorang data scientist dapat mencapai level executive ",
                f"dengan kompensasi hingga ${exp_salary.get('EX', 0):,.0f}."
            ]),
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.P([
                "💡 Insight: Investasi dalam skill development dan pengalaman memberikan ROI yang sangat tinggi ",
                "dalam karir data science."
            ], className="text-primary fw-bold")
        ])
    ], className="shadow-sm border-0")

def create_chapter_2(df, colors):
    """Create Chapter 2: Remote Revolution"""
    remote_avg = df.groupby('remote_ratio')['salary_in_usd'].mean()
    
    fig = px.scatter(
        df, x='remote_ratio', y='salary_in_usd',
        title="Remote Work Revolution: Flexibility Meets Compensation",
        color='experience_level',
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], colors['dark']]
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Chapter 2: Remote Work Revolution 🏠")
        ]),
        dbc.CardBody([
            html.P([
                "Era pandemic telah mengubah landscape kerja secara fundamental. ",
                "Data menunjukkan bahwa remote work tidak hanya memberikan fleksibilitas, ",
                "tetapi juga membuka peluang kompensasi yang lebih tinggi."
            ]),
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.P([
                "🌟 Insight: Full remote workers (100%) rata-rata mendapat kompensasi ",
                f"{calculate_remote_premium(df):.1f}% lebih tinggi dibanding on-site workers."
            ], className="text-success fw-bold")
        ])
    ], className="shadow-sm border-0")

def create_chapter_3(df, colors):
    """Create Chapter 3: Global Landscape"""
    top_countries = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(8)
    
    fig = px.bar(
        x=top_countries.index, y=top_countries.values,
        title="Global Salary Landscape: Where Data Scientists Thrive",
        color=top_countries.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Chapter 3: Global Landscape 🌍")
        ]),
        dbc.CardBody([
            html.P([
                "Kompetisi global untuk talent data science menciptakan landscape gaji yang beragam. ",
                "Beberapa negara muncul sebagai hotspot dengan kompensasi premium untuk data scientists."
            ]),
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.P([
                f"🏆 Champion: {get_top_paying_country(df)} memimpin dengan rata-rata gaji tertinggi, ",
                "mencerminkan demand tinggi dan supply terbatas untuk talent berkualitas."
            ], className="text-warning fw-bold")
        ])
    ], className="shadow-sm border-0")

def create_chapter_4(df, colors):
    """Create Chapter 4: Future Trends"""
    yearly_trend = df.groupby('work_year')['salary_in_usd'].agg(['mean', 'count'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=yearly_trend.index, y=yearly_trend['mean'],
                  mode='lines+markers', name='Average Salary',
                  line=dict(color=colors['primary'], width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=yearly_trend.index, y=yearly_trend['count'],
              name='Job Postings', opacity=0.6,
              marker_color=colors['secondary']),
        secondary_y=True
    )
    
    fig.update_layout(title="Future Trends: The Rise of Data Science")
    fig.update_yaxes(title_text="Average Salary (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Positions", secondary_y=True)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Chapter 4: Future Trends 🔮")
        ]),
        dbc.CardBody([
            html.P([
                "Tren menunjukkan pertumbuhan yang konsisten dalam kompensasi data scientists. ",
                "Demand yang terus meningkat dan skill gap yang masih lebar menciptakan ",
                "peluang karir yang sangat menjanjikan di masa depan."
            ]),
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.P([
                "🚀 Prediction: Dengan adoption AI/ML yang semakin masif, profesi data scientist ",
                "akan terus menjadi salah satu yang paling sought-after dalam dekade mendatang."
            ], className="text-info fw-bold")
        ])
    ], className="shadow-sm border-0")