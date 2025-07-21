"""
Statistics and Overview Components
Handles statistical cards and overview sections
"""

import dash_bootstrap_components as dbc
from dash import html

def create_modern_stat_card(icon_type, value, label, description, color):
    """Create modern statistics card with creative icons"""
    
    # Define modern icon mappings
    icon_map = {
        "salary": "fas fa-dollar-sign",
        "data": "fas fa-database", 
        "globe": "fas fa-globe-americas",
        "briefcase": "fas fa-briefcase"
    }
    
    # Get icon with fallback
    icon_class = icon_map.get(icon_type, "fas fa-chart-bar")
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                # Modern Icon Container
                html.Div([
                    html.I(className=f"{icon_class} fa-2x", style={'color': 'white'})
                ], className="modern-icon-container", style={
                    'width': '60px',
                    'height': '60px',
                    'borderRadius': '15px',
                    'background': f'linear-gradient(135deg, {color}, {color}dd)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'margin': '0 auto 20px auto',
                    'boxShadow': f'0 8px 20px {color}30'
                }),
                
                # Value and Label
                html.H2(value, className="fw-bold mb-2", style={'color': '#1f2937', 'fontSize': '2rem'}),
                html.H6(label, className="fw-bold mb-2", style={'color': color, 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                html.P(description, className="mb-0 text-muted", style={'fontSize': '0.9rem'})
            ], className="text-center")
        ], className="p-4")
    ], className="h-100 border-0 shadow-sm modern-stat-card",
       style={
           'background': 'rgba(255, 255, 255, 0.95)',
           'backdropFilter': 'blur(10px)',
           'borderRadius': '20px',
           'transition': 'all 0.3s ease',
           'border': f'1px solid {color}20'
       })

def create_data_overview_section(df, colors):
    """Create simplified data overview section"""
    return html.Div([
        html.H3("Data Overview", className="text-center mb-4"),
        dbc.Row([
            dbc.Col([
                html.H5(f"{len(df):,} Records"),
                html.P("Total data points analyzed")
            ], md=3, className="text-center mb-3"),
            dbc.Col([
                html.H5(f"{df['job_title'].nunique()} Jobs"),
                html.P("Unique job positions")
            ], md=3, className="text-center mb-3"),
            dbc.Col([
                html.H5(f"{df['company_location'].nunique()} Countries"),
                html.P("Global coverage")
            ], md=3, className="text-center mb-3"),
            dbc.Col([
                html.H5(f"${df['salary_in_usd'].mean():,.0f}"),
                html.P("Average salary")
            ], md=3, className="text-center mb-3")
        ])
    ], className="mb-5")

def create_nav_card(icon, title, description, href, color):
    """Create navigation card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, className="fs-1 mb-3"),
                html.H5(title, className="fw-bold mb-2"),
                html.P(description, className="text-muted mb-3"),
                dbc.Button("Explore", color="primary", size="sm", href=href,
                          style={'borderRadius': '20px'})
            ], className="text-center")
        ])
    ], className="h-100 border-0 shadow-sm card-hover",
       style={'background': f'linear-gradient(145deg, #ffffff, {color}10)'})

def create_download_section(colors):
    """Create download section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-download me-3", style={'color': 'white'}),
                "Download & Export"
            ], className="mb-0 text-center")
        ], style={'background': colors['gradient'], 'color': 'white'}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("📥 Quick Downloads", className="text-primary fw-bold mb-4 text-center"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Div([
                                            html.I(className="fas fa-file-csv fa-2x text-success mb-3"),
                                            html.H6("Dataset (CSV)", className="fw-bold mb-2"),
                                            html.P("Unduh dataset lengkap", className="text-muted small mb-3"),
                                            dbc.Button([
                                                html.I(className="fas fa-download me-2"),
                                                "Unduh CSV"
                                            ], color="success", size="sm", className="w-100", id="btn-download-csv-home")
                                        ], className="text-center")
                                    ])
                                ], className="h-100 shadow-sm border-0",
                                   style={'borderTop': f'3px solid {colors["secondary"]}'})
                            ], md=4, className="mb-3"),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Div([
                                            html.I(className="fas fa-chart-line fa-2x text-info mb-3"),
                                            html.H6("Laporan Analisis", className="fw-bold mb-2"),
                                            html.P("Ringkasan insight & temuan", className="text-muted small mb-3"),
                                            dbc.Button([
                                                html.I(className="fas fa-file-pdf me-2"),
                                                "Unduh PDF"
                                            ], color="info", size="sm", className="w-100")
                                        ], className="text-center")
                                    ])
                                ], className="h-100 shadow-sm border-0",
                                   style={'borderTop': f'3px solid {colors["accent"]}'})
                            ], md=4, className="mb-3"),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Div([
                                            html.I(className="fas fa-robot fa-2x text-warning mb-3"),
                                            html.H6("Model ML", className="fw-bold mb-2"),
                                            html.P("Model prediksi yang sudah dilatih", className="text-muted small mb-3"),
                                            dbc.Button([
                                                html.I(className="fas fa-download me-2"),
                                                "Unduh Model"
                                            ], color="warning", size="sm", className="w-100")
                                        ], className="text-center")
                                    ])
                                ], className="h-100 shadow-sm border-0",
                                   style={'borderTop': f'3px solid {colors["dark"]}'})
                            ], md=4, className="mb-3")
                        ])
                    ])
                ], md=10, className="mx-auto"),
            ]),
            
            html.Hr(),
            
            html.Div([
                html.P([
                    html.I(className="fas fa-info-circle me-2 text-primary"),
                    "Semua file unduhan tersedia dalam format yang kompatibel dengan alat analisis populer."
                ], className="text-center text-muted mb-0 small")
            ])
        ])
    ], className="shadow-sm border-0 mb-4")