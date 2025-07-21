"""
Insights & Recommendations Page
Key findings and actionable recommendations
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def layout(df, colors, lang='id'):
    """Create insights and recommendations page layout"""
    from app import get_text
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-lightbulb me-3", style={'color': colors['primary']}),
                            get_text('insights', 'title', lang)
                        ], className="display-5 fw-bold mb-3"),
                        html.P(get_text('insights', 'subtitle', lang),
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Executive Summary
            dbc.Row([
                dbc.Col([
                    create_executive_summary(df, colors)
                ])
            ], className="mb-4"),
            
            # Key Insights Grid
            dbc.Row([
                dbc.Col([
                    html.H3([
                        html.I(className="fas fa-key me-2", style={'color': colors['primary']}),
                        "Key Insights"
                    ], className="mb-4 text-center")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_insight_card(
                        "💰", "Salary Premium Factors",
                        generate_salary_insights(df),
                        colors['primary']
                    )
                ], md=6, className="mb-4"),
                dbc.Col([
                    create_insight_card(
                        "🌍", "Geographic Advantages", 
                        generate_geographic_insights(df),
                        colors['secondary']
                    )
                ], md=6, className="mb-4")
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_insight_card(
                        "📈", "Career Progression",
                        generate_career_insights(df),
                        colors['accent']
                    )
                ], md=6, className="mb-4"),
                dbc.Col([
                    create_insight_card(
                        "🏠", "Remote Work Impact",
                        generate_remote_insights(df),
                        colors['dark']
                    )
                ], md=6, className="mb-4")
            ]),
            
            # Recommendations Section
            dbc.Row([
                dbc.Col([
                    create_recommendations_section(df, colors)
                ])
            ], className="mb-4"),
            
            # Benchmarking Tools
            dbc.Row([
                dbc.Col([
                    create_benchmarking_section(df, colors)
                ], md=8, className="mb-4"),
                dbc.Col([
                    create_action_plan(colors)
                ], md=4, className="mb-4")
            ]),
            
            # Market Analysis
            dbc.Row([
                dbc.Col([
                    create_market_analysis(df, colors)
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_executive_summary(df, colors):
    """Create executive summary card"""
    
    # Calculate key metrics
    avg_salary = df['salary_in_usd'].mean()
    salary_range = df['salary_in_usd'].max() - df['salary_in_usd'].min()
    top_country = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).index[0]
    remote_premium = calculate_remote_premium(df)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-chart-line me-2"),
                "Executive Summary"
            ], className="mb-0 text-white")
        ], style={'background': colors['gradient']}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("🎯 Key Findings", className="text-primary mb-3"),
                    html.Ul([
                        html.Li([
                            html.Strong("Salary Range: "),
                            f"Data scientist salaries vary significantly, with an average of ${avg_salary:,.0f} ",
                            f"and range of ${salary_range:,.0f}"
                        ], className="mb-2"),
                        html.Li([
                            html.Strong("Experience Premium: "),
                            "Senior level positions command significantly higher salaries, ",
                            "with up to 3x difference from entry level"
                        ], className="mb-2"),
                        html.Li([
                            html.Strong("Remote Advantage: "),
                            f"Full remote positions offer {remote_premium:.1f}% premium ",
                            "compared to on-site roles"
                        ], className="mb-2"),
                        html.Li([
                            html.Strong("Geographic Leaders: "),
                            f"{top_country} leads in average compensation, ",
                            "reflecting market demand and cost of living"
                        ], className="mb-2")
                    ])
                ], md=6),
                dbc.Col([
                    html.H5("📊 Market Outlook", className="text-success mb-3"),
                    html.Div([
                        create_market_indicator("Strong", "Job Market", "High demand for DS skills", colors['primary']),
                        create_market_indicator("Growing", "Salary Trends", "Consistent upward trajectory", colors['secondary']),
                        create_market_indicator("Competitive", "Talent War", "Companies competing for top talent", colors['accent']),
                        create_market_indicator("Positive", "Remote Shift", "Increased remote opportunities", colors['dark'])
                    ])
                ], md=6)
            ])
        ])
    ], className="shadow-sm border-0 mb-4")

def create_market_indicator(status, title, description, color):
    """Create market indicator"""
    return html.Div([
        html.Div([
            html.Span(status, className="badge fw-bold px-3 py-2", 
                     style={'backgroundColor': color, 'color': 'white'}),
            html.Span(title, className="fw-bold ms-2"),
            html.Br(),
            html.Small(description, className="text-muted ms-3")
        ], className="mb-2")
    ])

def create_insight_card(icon, title, insights, color):
    """Create insight card with analysis"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.Span(icon, className="me-2"),
                title
            ], className="mb-0")
        ], style={'backgroundColor': color + '20', 'borderBottom': f'2px solid {color}'}),
        dbc.CardBody([
            html.Div(insights)
        ])
    ], className="h-100 shadow-sm border-0")

def generate_salary_insights(df):
    """Generate salary-related insights"""
    
    # Calculate statistics
    exp_multiplier = df[df['experience_level'] == 'SE']['salary_in_usd'].mean() / df[df['experience_level'] == 'EN']['salary_in_usd'].mean()
    size_impact = df.groupby('company_size')['salary_in_usd'].mean()
    
    return html.Div([
        html.P([
            html.Strong("Experience Level Impact: "),
            f"Senior level professionals earn {exp_multiplier:.1f}x more than entry level, ",
            "highlighting the importance of skill development and experience."
        ], className="mb-2"),
        
        html.P([
            html.Strong("Company Size Effect: "),
            f"Large companies (${size_impact.get('L', 0):,.0f}) vs ",
            f"Small companies (${size_impact.get('S', 0):,.0f}) shows ",
            f"${size_impact.get('L', 0) - size_impact.get('S', 0):,.0f} difference."
        ], className="mb-2"),
        
        html.P([
            html.Strong("Skill Premium: "),
            "Specialized roles like ML Engineers and Data Scientists ",
            "command higher premiums than general analyst positions."
        ], className="mb-2"),
        
        html.Div([
            html.Strong("💡 Takeaway: ", className="text-primary"),
            "Focus on skill development, gain experience, and target larger companies for maximum compensation."
        ], className="p-2 rounded", style={'backgroundColor': '#f0fdf4'})
    ])

def generate_geographic_insights(df):
    """Generate geographic insights"""
    
    top_countries = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(5)
    us_vs_others = df[df['company_location'] == 'US']['salary_in_usd'].mean() / df[df['company_location'] != 'US']['salary_in_usd'].mean()
    
    return html.Div([
        html.P([
            html.Strong("Top Markets: "),
            f"Leading countries by average salary: {', '.join(top_countries.head(3).index)}"
        ], className="mb-2"),
        
        html.P([
            html.Strong("US Premium: "),
            f"US positions offer {us_vs_others:.1f}x higher compensation ",
            "than international markets on average."
        ], className="mb-2"),
        
        html.P([
            html.Strong("Remote Opportunities: "),
            "Geographic barriers are reducing due to remote work, ",
            "allowing access to global salary standards."
        ], className="mb-2"),
        
        html.Div([
            html.Strong("🌍 Strategy: ", className="text-success"),
            "Consider remote positions with US companies or relocate to high-paying markets."
        ], className="p-2 rounded", style={'backgroundColor': '#ecfdf5'})
    ])

def generate_career_insights(df):
    """Generate career progression insights"""
    
    job_distribution = df['job_title'].value_counts().head(5)
    exp_distribution = df['experience_level'].value_counts()
    
    return html.Div([
        html.P([
            html.Strong("Popular Roles: "),
            f"Most common positions: {job_distribution.index[0]} ({job_distribution.iloc[0]} positions), ",
            f"{job_distribution.index[1]} ({job_distribution.iloc[1]} positions)"
        ], className="mb-2"),
        
        html.P([
            html.Strong("Experience Distribution: "),
            f"Senior level dominates ({exp_distribution.get('SE', 0)} positions), ",
            "indicating market maturity and demand for experienced professionals."
        ], className="mb-2"),
        
        html.P([
            html.Strong("Career Path: "),
            "Clear progression from Data Analyst → Data Scientist → ML Engineer → Lead positions ",
            "with corresponding salary increases."
        ], className="mb-2"),
        
        html.Div([
            html.Strong("🚀 Growth Plan: ", className="text-info"),
            "Build portfolio, gain certifications, and aim for senior positions within 3-5 years."
        ], className="p-2 rounded", style={'backgroundColor': '#f0f9ff'})
    ])

def generate_remote_insights(df):
    """Generate remote work insights"""
    
    remote_dist = df['remote_ratio'].value_counts()
    remote_premium = calculate_remote_premium(df)
    
    return html.Div([
        html.P([
            html.Strong("Remote Distribution: "),
            f"Fully remote: {remote_dist.get(100, 0)} positions, ",
            f"Hybrid: {remote_dist.get(50, 0)} positions, ",
            f"On-site: {remote_dist.get(0, 0)} positions"
        ], className="mb-2"),
        
        html.P([
            html.Strong("Salary Impact: "),
            f"Full remote workers earn {remote_premium:.1f}% more ",
            "than on-site counterparts, reflecting premium for flexibility."
        ], className="mb-2"),
        
        html.P([
            html.Strong("Market Trend: "),
            "Remote-first culture is becoming standard, ",
            "especially for senior and specialized roles."
        ], className="mb-2"),
        
        html.Div([
            html.Strong("🏠 Recommendation: ", className="text-warning"),
            "Negotiate for remote work options to maximize both flexibility and compensation."
        ], className="p-2 rounded", style={'backgroundColor': '#fffbeb'})
    ])

def create_recommendations_section(df, colors):
    """Create recommendations section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-bullseye me-2"),
                "Strategic Recommendations"
            ], className="mb-0 text-white")
        ], style={'background': colors['gradient']}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("👨‍💼 For Job Seekers", className="text-primary mb-3"),
                    create_recommendation_list([
                        "Target senior-level positions for maximum salary potential",
                        "Consider remote opportunities with US-based companies",
                        "Focus on specialized skills like ML, AI, and cloud technologies",
                        "Build strong portfolio demonstrating business impact",
                        "Negotiate based on market data and location premiums"
                    ], colors['primary'])
                ], md=6),
                dbc.Col([
                    html.H5("🏢 For Employers", className="text-success mb-3"),
                    create_recommendation_list([
                        "Offer competitive remote work packages to attract talent",
                        "Benchmark salaries against top-paying markets",
                        "Invest in employee upskilling and career development",
                        "Create clear progression paths with salary transparency",
                        "Consider location-based compensation adjustments"
                    ], colors['secondary'])
                ], md=6)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H5("🎓 For Career Development", className="text-info mb-3"),
                    create_recommendation_list([
                        "Pursue advanced degrees or professional certifications",
                        "Gain experience with modern tools and frameworks",
                        "Build domain expertise in high-demand industries",
                        "Develop leadership and communication skills",
                        "Participate in open source projects and competitions"
                    ], colors['accent'])
                ], md=6),
                dbc.Col([
                    html.H5("💼 For Negotiation", className="text-warning mb-3"),
                    create_recommendation_list([
                        "Research market rates for your experience level",
                        "Highlight unique skills and achievements",
                        "Consider total compensation, not just base salary",
                        "Leverage remote work options for better offers",
                        "Be prepared to demonstrate ROI and business value"
                    ], colors['dark'])
                ], md=6)
            ])
        ])
    ], className="shadow-sm border-0")

def create_recommendation_list(items, color):
    """Create styled recommendation list"""
    list_items = []
    for item in items:
        list_items.append(
            html.Li([
                html.I(className="fas fa-check-circle me-2", style={'color': color}),
                item
            ], className="mb-2")
        )
    return html.Ul(list_items, className="list-unstyled")

def create_benchmarking_section(df, colors):
    """Create salary benchmarking section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-chart-bar me-2"),
                "Salary Benchmarking Tool"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            create_benchmarking_chart(df, colors)
        ])
    ], className="shadow-sm border-0")

def create_benchmarking_chart(df, colors):
    """Create interactive benchmarking chart"""
    
    # Create salary ranges by experience level
    exp_stats = df.groupby('experience_level')['salary_in_usd'].agg([
        'min', 'quantile', 'median', lambda x: x.quantile(0.75), 'max'
    ]).round(0)
    
    exp_labels = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
    
    fig = go.Figure()
    
    for exp in exp_stats.index:
        fig.add_trace(go.Box(
            y=df[df['experience_level'] == exp]['salary_in_usd'],
            name=exp_labels.get(exp, exp),
            marker_color=colors['primary'],
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title="Salary Benchmarking by Experience Level",
        yaxis_title="Annual Salary (USD)",
        xaxis_title="Experience Level",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.P([
            html.Strong("💡 How to use: "),
            "Compare your current or target salary against market benchmarks. ",
            "The box shows the middle 50% of salaries, with outliers displayed as points."
        ], className="text-muted mt-3")
    ])

def create_action_plan(colors):
    """Create action plan section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-tasks me-2"),
                "30-60-90 Day Action Plan"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.Div([
                html.H6("📅 30 Days", className="text-primary fw-bold"),
                html.Ul([
                    html.Li("Research current market rates"),
                    html.Li("Update resume and LinkedIn profile"),
                    html.Li("Identify skill gaps"),
                    html.Li("Start networking activities")
                ], className="small mb-3")
            ]),
            
            html.Div([
                html.H6("📅 60 Days", className="text-success fw-bold"),
                html.Ul([
                    html.Li("Complete relevant certifications"),
                    html.Li("Build portfolio projects"),
                    html.Li("Apply to target positions"),
                    html.Li("Practice interview skills")
                ], className="small mb-3")
            ]),
            
            html.Div([
                html.H6("📅 90 Days", className="text-warning fw-bold"),
                html.Ul([
                    html.Li("Negotiate offers effectively"),
                    html.Li("Make career transition"),
                    html.Li("Set up for continued growth"),
                    html.Li("Plan next career milestone")
                ], className="small mb-3")
            ])
        ])
    ], className="shadow-sm border-0 h-100")

def create_market_analysis(df, colors):
    """Create market analysis section"""
    
    # Job title analysis
    job_salary = df.groupby('job_title')['salary_in_usd'].agg(['mean', 'count']).reset_index()
    job_salary = job_salary[job_salary['count'] >= 10].sort_values('mean', ascending=False).head(10)
    
    fig = px.bar(
        job_salary, x='mean', y='job_title',
        orientation='h',
        title="Top 10 Highest Paying Data Science Roles",
        color='mean',
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
    )
    
    fig.update_layout(
        xaxis_title="Average Salary (USD)",
        yaxis_title="Job Title",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-industry me-2"),
                "Market Analysis & Trends"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig, config={'displayModeBar': False})
                ], md=8),
                dbc.Col([
                    html.H6("🎯 Market Insights", className="fw-bold mb-3"),
                    html.Div([
                        html.P([
                            html.Strong("Highest Demand: "),
                            f"{df['job_title'].value_counts().index[0]} positions dominate the market"
                        ], className="small mb-2"),
                        html.P([
                            html.Strong("Growth Areas: "),
                            "ML Engineering, Data Engineering, and AI roles show premium compensation"
                        ], className="small mb-2"),
                        html.P([
                            html.Strong("Emerging Trends: "),
                            "Cloud-native roles and AI/ML specializations leading salary growth"
                        ], className="small mb-2"),
                        html.P([
                            html.Strong("Market Maturity: "),
                            "Data science field showing signs of specialization and premium for expertise"
                        ], className="small mb-2")
                    ]),
                    
                    html.Hr(),
                    
                    html.Div([
                        html.H6("📈 Future Outlook", className="fw-bold text-success"),
                        html.P("Continued growth expected with AI adoption", className="small text-success"),
                        html.P("Remote work will remain prevalent", className="small text-info"),
                        html.P("Specialization will drive salary premiums", className="small text-warning")
                    ])
                ], md=4)
            ])
        ])
    ], className="shadow-sm border-0")

# Helper function
def calculate_remote_premium(df):
    """Calculate remote work premium"""
    remote_100 = df[df['remote_ratio'] == 100]['salary_in_usd'].mean()
    onsite = df[df['remote_ratio'] == 0]['salary_in_usd'].mean()
    if onsite > 0:
        return ((remote_100 - onsite) / onsite) * 100
    return 0