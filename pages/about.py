"""
About Team Page
Information about the project creators and team
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def layout(df, colors, lang='id'):
    """Create about team page layout with improved contact & footer sections"""
    from app import get_text
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1(get_text('about', 'title', lang), className="display-5 fw-bold mb-3"),
                        html.P(get_text('about', 'subtitle', lang),
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        # Main Content
        dbc.Container([
            # Team Introduction
            dbc.Row([
                dbc.Col([
                    create_team_introduction(colors)
                ])
            ], className="mb-5"),
            
            # Team Members Section dengan foto
            dbc.Row([
                dbc.Col([
                    html.H2("Tim Pengembang", className="text-center mb-5 fw-bold display-5", 
                           style={'color': colors['darker']})
                ])
            ]),
            
            # Profile Lingga
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-user-circle fa-5x mb-3", style={'color': colors['primary']})
                    ], className="text-center")
                ], md=5),
                dbc.Col([
                    html.H3("Lingga Dwi Satria Vigio", className="fw-bold"),
                    html.H6("LEAD DEVELOPER & MACHINE LEARNING", className="text-muted mb-3", 
                           style={'letterSpacing': '1px'}),
                    html.P("Bertanggung jawab atas arsitektur backend, pemrosesan data, "
                          "pengembangan model machine learning, dan memastikan semua "
                          "logika aplikasi berjalan dengan lancar dari data cleaning hingga deployment model.",
                          className="text-secondary"),
                    html.Div([
                        html.A(html.I(className="fas fa-envelope fa-lg"), 
                              href="mailto:lingga22si@mahasiswa.pcr.ac.id", 
                              className="text-dark me-3", title="Email"),
                        html.A(html.I(className="fab fa-linkedin fa-lg"), 
                              href="#", target="_blank", 
                              className="text-dark me-3", title="LinkedIn"),
                        html.A(html.I(className="fab fa-github fa-lg"), 
                              href="https://github.com/lingga", target="_blank", 
                              className="text-dark", title="GitHub")
                    ], className="mt-4")
                ], md=7, className="d-flex flex-column justify-content-center")
            ], className="align-items-center mb-5"),
            
            # Profile Azzahara
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-user-circle fa-5x mb-3", style={'color': colors['primary']})
                    ], className="text-center")
                ], md=5, className="order-md-1"),
                dbc.Col([
                    html.H3("Azzahara Tunisyah", className="fw-bold"),
                    html.H6("LEAD ANALYST & UI/UX DESIGNER", className="text-muted mb-3", 
                           style={'letterSpacing': '1px'}),
                    html.P("Memimpin analisis data eksploratif (EDA), visualisasi data, dan storytelling. "
                          "Bertugas merancang user interface yang intuitif dan menarik, serta menerjemahkan "
                          "data kompleks menjadi insight yang mudah dipahami.",
                          className="text-secondary"),
                    html.Div([
                        html.A(html.I(className="fas fa-envelope fa-lg"), 
                              href="mailto:azzahara22si@mahasiswa.pcr.ac.id", 
                              className="text-dark me-3", title="Email"),
                        html.A(html.I(className="fab fa-linkedin fa-lg"), 
                              href="#", target="_blank", 
                              className="text-dark me-3", title="LinkedIn"),
                        html.A(html.I(className="fab fa-github fa-lg"), 
                              href="https://github.com/azzahara", target="_blank", 
                              className="text-dark", title="GitHub")
                    ], className="mt-4")
                ], md=7, className="d-flex flex-column justify-content-center order-md-2")
            ], className="align-items-center mb-5"),
            
            # Project Information
            dbc.Row([
                dbc.Col([
                    create_project_info_improved(colors)
                ])
            ], className="mb-5"),
            
            # Technologies Used
            dbc.Row([
                dbc.Col([
                    create_technologies_section(colors)
                ], md=8, className="mb-4"),
                dbc.Col([
                    create_project_timeline(colors)
                ], md=4, className="mb-4")
            ]),
            
        ], fluid=True, className="py-4")
    ])

def create_team_introduction(colors):
    """Create team introduction section"""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🎓 Tentang DataSalary Analytics", className="text-primary mb-3"),
                        html.P([
                            "DataSalary Analytics adalah platform analytics terdepan untuk menganalisis ",
                            "tren gaji, faktor kompensasi, dan peluang karir di industri data science. ",
                            "Platform ini menggabingan analisis eksploratif mendalam, machine learning, ",
                            "dan dashboard interaktif untuk memberikan insights yang actionable."
                        ], className="mb-3", style={'textAlign': 'justify'}),
                        
                        html.P([
                            "Berdasarkan data komprehensif dari ribuan profesional global, kami ",
                            "menghadirkan analisis end-to-end dari preprocessing hingga prediksi, ",
                            "dilengkapi dengan visualisasi yang intuitif dan user-friendly."
                        ], style={'textAlign': 'justify'})
                    ])
                ], md=8),
                dbc.Col([
                    html.Div([
                        create_project_stats(colors)
                    ])
                ], md=4)
            ])
        ])
    ], className="shadow-sm border-0", 
       style={'background': f'linear-gradient(135deg, {colors["white"]}, {colors["light"]}20)'})

def create_project_stats(colors):
    """Create project statistics"""
    return html.Div([
        html.H6("📊 Project Stats", className="fw-bold mb-3 text-center"),
        
        create_stat_item("🗓️", "Duration", "3 Months", colors['primary']),
        create_stat_item("📝", "Code Lines", "2,500+", colors['secondary']),
        create_stat_item("📊", "Data Points", "3,755", colors['accent']),
        create_stat_item("🎯", "Features", "11", colors['dark']),
        create_stat_item("📱", "Pages", "9", colors['primary']),
        create_stat_item("🔧", "Technologies", "8+", colors['secondary'])
    ])

def create_stat_item(icon, label, value, color):
    """Create project statistic item"""
    return html.Div([
        html.Div([
            html.Span(icon, className="me-2"),
            html.Strong(label + ": "),
            html.Span(value, style={'color': color, 'fontWeight': 'bold'})
        ], className="mb-2 small")
    ])

def create_team_member_card(name, role, contact, position, skills, email, linkedin, github, color):
    """Create improved team member card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                # Avatar placeholder
                html.Div([
                    html.I(className="fas fa-user-circle fa-4x", style={'color': color})
                ], className="text-center mb-3"),
                
                # Basic Info
                html.H5(name, className="fw-bold text-center mb-2"),
                html.P(role, className="text-center text-muted mb-1"),
                html.P(contact, className="text-center small text-primary mb-3"),
                
                # Skills/Badges
                html.Div([
                    dbc.Badge(skill, color="primary" if color == '#10b981' else "info", className="me-1 mb-1") 
                    for skill in skills
                ], className="text-center mb-3"),
                
                # Contact buttons - simplified
                html.Div([
                    dbc.Button([
                        html.I(className="fas fa-envelope me-1"),
                        "Contact"
                    ], color="outline-primary", size="sm", className="w-100")
                ])
            ])
        ])
    ], className="h-100 shadow-sm border-0 text-center team-member-card",
       style={'borderTop': f'4px solid {color}'})

def create_project_info_improved(colors):
    """Create improved project information section"""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={'color': colors['primary']}),
                html.Span("Kelompok 6", style={'color': colors['primary'], 'fontWeight': 'bold'}),
                html.Br(),
                html.Small("Data Science", className="text-muted")
            ], className="text-center")
        ], className="bg-light project-info-section"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-user-circle fa-3x", style={'color': colors['primary']})
                        ], className="text-center mb-3"),
                        html.H5("Lingga Dwi Satria Vigio", className="fw-bold text-center mb-1"),
                        html.P("Data Science Student", className="text-center text-muted mb-1"),
                        html.P([
                            html.I(className="fas fa-phone me-1"),
                            "08580590809"
                        ], className="text-center small"),
                        html.Div([
                            dbc.Badge("Team Member", color="secondary", className="me-1"),
                            dbc.Badge("Python", color="secondary", className="me-1"),
                            dbc.Badge("Machine Learning", color="secondary")
                        ], className="text-center")
                    ])
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-user-circle fa-3x", style={'color': colors['secondary']})
                        ], className="text-center mb-3"),
                        html.H5("Azzahara Tunisyah", className="fw-bold text-center mb-1"),
                        html.P("Data Science Student", className="text-center text-muted mb-1"),
                        html.P([
                            html.I(className="fas fa-envelope me-1"),
                            "Team Member"
                        ], className="text-center small"),
                        html.Div([
                            dbc.Badge("Team Member", color="info", className="me-1"),
                            dbc.Badge("Data Analysis", color="info", className="me-1"),
                            dbc.Badge("Visualization", color="info")
                        ], className="text-center")
                    ])
                ], md=6)
            ]),
            
            html.Hr(className="my-4"),
            
            # Project Information Table
            dbc.Row([
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-clipboard-list me-2"),
                        "Project Information"
                    ], className="fw-bold mb-3"),
                    create_project_info_table_improved()
                ])
            ])
        ])
    ], className="shadow-sm border-0")

def create_project_info_table_improved():
    """Create improved project information table"""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-graduation-cap me-2", style={'color': '#10b981'}),
                    html.Strong("Course: "),
                    html.Span("Data Science")
                ], className="mb-2"),
                html.Div([
                    html.I(className="fas fa-calendar me-2", style={'color': '#10b981'}),
                    html.Strong("Year: "),
                    html.Span("2023/2024")
                ], className="mb-2"),
                html.Div([
                    html.I(className="fas fa-code me-2", style={'color': '#10b981'}),
                    html.Strong("Tech: "),
                    html.Span("Python, Dash, ML")
                ], className="mb-2")
            ])
        ], md=6),
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-bullseye me-2", style={'color': '#10b981'}),
                    html.Strong("Focus: "),
                    html.Span("Salary Analysis & Prediction")
                ], className="mb-2"),
                html.Div([
                    html.I(className="fas fa-database me-2", style={'color': '#10b981'}),
                    html.Strong("Dataset: "),
                    html.Span("3,755 records")
                ], className="mb-2"),
                html.Div([
                    html.I(className="fas fa-chart-line me-2", style={'color': '#10b981'}),
                    html.Strong("Analysis: "),
                    html.Span("End-to-end ML Pipeline")
                ], className="mb-2")
            ])
        ], md=6)
    ])

def create_technologies_section(colors):
    """Create technologies used section"""
    technologies = {
        "Backend": [
            {"name": "Python", "icon": "fab fa-python", "color": "#3776ab"},
            {"name": "Pandas", "icon": "fas fa-table", "color": "#150458"},
            {"name": "Scikit-learn", "icon": "fas fa-robot", "color": "#f7931e"},
            {"name": "Plotly", "icon": "fas fa-chart-line", "color": "#3f4f75"}
        ],
        "Frontend": [
            {"name": "Dash", "icon": "fas fa-tachometer-alt", "color": "#00cc96"},
            {"name": "Bootstrap", "icon": "fab fa-bootstrap", "color": "#7952b3"},
            {"name": "HTML5", "icon": "fab fa-html5", "color": "#e34f26"},
            {"name": "CSS3", "icon": "fab fa-css3-alt", "color": "#1572b6"}
        ],
        "Tools": [
            {"name": "Google Colab", "icon": "fas fa-laptop-code", "color": "#f9ab00"},
            {"name": "GitHub", "icon": "fab fa-github", "color": "#181717"},
            {"name": "VS Code", "icon": "fas fa-code", "color": "#007acc"},
            {"name": "Jupyter", "icon": "fas fa-book", "color": "#f37626"}
        ]
    }
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-tools me-2"),
                "Technologies Used"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6(category, className="fw-bold mb-3"),
                    dbc.Row([
                        dbc.Col([
                            create_tech_item(tech["name"], tech["icon"], tech["color"])
                        ], md=6, className="mb-3") for tech in techs
                    ])
                ], md=4) for category, techs in technologies.items()
            ])
        ])
    ], className="shadow-sm border-0")

def create_tech_item(name, icon, color):
    """Create technology item"""
    return html.Div([
        html.I(className=f"{icon} fa-2x me-2", style={'color': color}),
        html.Span(name, className="fw-bold")
    ], className="d-flex align-items-center")

def create_project_timeline(colors):
    """Create project timeline"""
    timeline_items = [
        {"phase": "Planning", "duration": "Week 1-2", "status": "completed"},
        {"phase": "Data Collection", "duration": "Week 3", "status": "completed"},
        {"phase": "EDA & Cleaning", "duration": "Week 4-6", "status": "completed"},
        {"phase": "Modeling", "duration": "Week 7-8", "status": "completed"},
        {"phase": "Dashboard Dev", "duration": "Week 9-11", "status": "completed"},
        {"phase": "Testing & Deploy", "duration": "Week 12", "status": "completed"}
    ]
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-calendar-alt me-2"),
                "Project Timeline"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.Div([
                create_timeline_item(item["phase"], item["duration"], item["status"], colors)
                for item in timeline_items
            ])
        ])
    ], className="shadow-sm border-0 h-100")

def create_timeline_item(phase, duration, status, colors):
    """Create timeline item"""
    status_color = colors['primary'] if status == 'completed' else colors['secondary']
    status_icon = "fas fa-check-circle" if status == 'completed' else "fas fa-clock"
    
    return html.Div([
        html.Div([
            html.I(className=f"{status_icon} me-2", style={'color': status_color}),
            html.Strong(phase),
            html.Br(),
            html.Small(duration, className="text-muted")
        ], className="border-start border-3 ps-3 mb-3",
           style={'borderColor': f'{status_color}!important'})
    ])