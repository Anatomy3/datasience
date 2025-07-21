"""
Footer Component
Modern footer for Data Science Dashboard
"""

import dash_bootstrap_components as dbc
from dash import html

def create_footer(colors):
    """Create modern footer component"""
    return html.Footer([
        # Main Footer Content
        dbc.Container([
            dbc.Row([
                # About Section
                dbc.Col([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-chart-line me-2", style={'color': colors['secondary']}),
                            "DS Salaries"
                        ], className="fw-bold mb-3", style={'color': 'white'}),
                        html.P([
                            "Comprehensive analysis of Data Science salaries worldwide. "
                            "Built with modern technologies to provide insights into "
                            "industry trends and career opportunities."
                        ], className="mb-3", style={'color': '#d1d5db', 'fontSize': '0.9rem'}),
                        html.Div([
                            html.Span([
                                html.I(className="fas fa-database me-2"),
                                "3,755 Records"
                            ], className="badge me-2", 
                               style={'backgroundColor': colors['secondary'], 'color': 'white'}),
                            html.Span([
                                html.I(className="fas fa-globe me-2"),
                                "Global Data"
                            ], className="badge", 
                               style={'backgroundColor': colors['accent'], 'color': colors['darker']})
                        ])
                    ])
                ], md=4, className="mb-4"),
                
                # Quick Links Section
                dbc.Col([
                    html.H6("Quick Links", className="fw-bold mb-3", style={'color': 'white'}),
                    html.Ul([
                        html.Li([
                            html.A([
                                html.I(className="fas fa-home me-2"),
                                "Home"
                            ], href="/", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-table me-2"),
                                "Data Overview"
                            ], href="/data-overview", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Visualizations"
                            ], href="/eda", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-robot me-2"),
                                "ML Models"
                            ], href="/modeling", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-download me-2"),
                                "Downloads"
                            ], href="/download", className="footer-link")
                        ])
                    ], className="list-unstyled")
                ], md=2, className="mb-4"),
                
                # Analysis Section
                dbc.Col([
                    html.H6("Analysis", className="fw-bold mb-3", style={'color': 'white'}),
                    html.Ul([
                        html.Li([
                            html.A([
                                html.I(className="fas fa-search me-2"),
                                "EDA"
                            ], href="/eda", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-broom me-2"),
                                "Data Cleaning"
                            ], href="/cleaning", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Results"
                            ], href="/results", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-lightbulb me-2"),
                                "Insights"
                            ], href="/insights", className="footer-link")
                        ], className="mb-2"),
                        html.Li([
                            html.A([
                                html.I(className="fas fa-info-circle me-2"),
                                "About"
                            ], href="/about", className="footer-link")
                        ])
                    ], className="list-unstyled")
                ], md=2, className="mb-4"),
                
                # Tech Stack Section
                dbc.Col([
                    html.H6("Built With", className="fw-bold mb-3", style={'color': 'white'}),
                    html.Div([
                        html.Div([
                            html.I(className="fab fa-python me-2", style={'color': colors['secondary']}),
                            html.Span("Python", style={'color': '#d1d5db'})
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-chart-line me-2", style={'color': colors['accent']}),
                            html.Span("Plotly Dash", style={'color': '#d1d5db'})
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fab fa-bootstrap me-2", style={'color': colors['secondary']}),
                            html.Span("Bootstrap", style={'color': '#d1d5db'})
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-robot me-2", style={'color': colors['accent']}),
                            html.Span("Scikit-learn", style={'color': '#d1d5db'})
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-table me-2", style={'color': colors['secondary']}),
                            html.Span("Pandas", style={'color': '#d1d5db'})
                        ])
                    ])
                ], md=2, className="mb-4"),
                
                # Contact & Social Section
                dbc.Col([
                    html.H6("Connect", className="fw-bold mb-3", style={'color': 'white'}),
                    html.Div([
                        # Social Links
                        html.Div([
                            html.A([
                                html.I(className="fab fa-github fa-lg")
                            ], href="#", className="social-link me-3", title="GitHub"),
                            html.A([
                                html.I(className="fab fa-linkedin fa-lg")
                            ], href="#", className="social-link me-3", title="LinkedIn"),
                            html.A([
                                html.I(className="fas fa-envelope fa-lg")
                            ], href="mailto:contact@example.com", className="social-link", title="Email")
                        ], className="mb-3"),
                        
                        # Contact Info
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-envelope me-2", style={'color': colors['secondary']}),
                                html.Small("info@dashboard.com", style={'color': '#d1d5db'})
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-globe me-2", style={'color': colors['accent']}),
                                html.Small("www.datasalaries.com", style={'color': '#d1d5db'})
                            ])
                        ])
                    ])
                ], md=2, className="mb-4")
            ], className="py-4"),
            
            # Divider
            html.Hr(style={'borderColor': colors['primary'] + '40', 'margin': '2rem 0 1rem 0'}),
            
            # Bottom Footer
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P([
                            "© 2024 DS Salaries Dashboard. All rights reserved. | ",
                            html.A("Privacy Policy", href="#", className="footer-link-small me-2"),
                            " | ",
                            html.A("Terms of Service", href="#", className="footer-link-small ms-2")
                        ], className="mb-1", style={'color': '#9ca3af', 'fontSize': '0.85rem'}),
                        html.P([
                            "Made with ❤️ for the Data Science Community | ",
                            "Last updated: ",
                            html.Span("June 2024", style={'color': colors['secondary']})
                        ], className="mb-0", style={'color': '#6b7280', 'fontSize': '0.8rem'})
                    ], className="text-center")
                ])
            ], className="pb-3")
            
        ], fluid=True)
    ], 
    style={
        'background': f'linear-gradient(135deg, {colors["darker"]} 0%, {colors["dark"]} 100%)',
        'marginTop': '4rem'
    })

# CSS styles for footer (sudah diintegrasikan ke app.py)
FOOTER_STYLES = """
.footer-link {
    color: #d1d5db !important;
    text-decoration: none !important;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}

.footer-link:hover {
    color: #34d399 !important;
    transform: translateX(5px);
}

.footer-link-small {
    color: #9ca3af !important;
    text-decoration: none !important;
    font-size: 0.85rem;
    transition: color 0.3s ease;
}

.footer-link-small:hover {
    color: #34d399 !important;
}

.social-link {
    color: #d1d5db !important;
    text-decoration: none !important;
    transition: all 0.3s ease;
    display: inline-block;
}

.social-link:hover {
    color: #34d399 !important;
    transform: translateY(-3px);
}

footer {
    box-shadow: 0 -4px 20px rgba(16, 185, 129, 0.1);
}
"""