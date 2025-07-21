"""
Prediction Components and Callbacks
Handles salary prediction functionality
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import joblib

# Load model
try:
    model = joblib.load('data/model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Using demo predictions.")
    model = None

def create_prediction_section(colors):
    """Create prediction section with form and results"""
    return dbc.Row([
        # Left Column - Form Input
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="fas fa-sliders-h me-2"),
                        "Input Parameters"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    # Row 1: Experience & Employment Type
                    dbc.Row([
                        dbc.Col([
                            html.Label("Experience Level", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='home-experience-dropdown',
                                options=[
                                    {'label': 'Entry Level', 'value': 'EN'},
                                    {'label': 'Mid Level', 'value': 'MI'},
                                    {'label': 'Senior Level', 'value': 'SE'},
                                    {'label': 'Executive Level', 'value': 'EX'}
                                ],
                                value='SE',
                                className="mb-3"
                            )
                        ], md=6),
                        dbc.Col([
                            html.Label("Employment Type", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='home-employment-dropdown',
                                options=[
                                    {'label': 'Full-time', 'value': 'FT'},
                                    {'label': 'Part-time', 'value': 'PT'},
                                    {'label': 'Contract', 'value': 'CT'},
                                    {'label': 'Freelance', 'value': 'FL'}
                                ],
                                value='FT',
                                className="mb-3"
                            )
                        ], md=6)
                    ]),
                    
                    # Row 2: Job Title & Company Size
                    dbc.Row([
                        dbc.Col([
                            html.Label("Job Title", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='home-job-title-dropdown',
                                options=[
                                    {'label': 'Data Scientist', 'value': 'Data Scientist'},
                                    {'label': 'Data Engineer', 'value': 'Data Engineer'},
                                    {'label': 'Data Analyst', 'value': 'Data Analyst'},
                                    {'label': 'ML Engineer', 'value': 'Machine Learning Engineer'}
                                ],
                                value='Data Scientist',
                                className="mb-3"
                            )
                        ], md=6),
                        dbc.Col([
                            html.Label("Company Size", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='home-company-size-dropdown',
                                options=[
                                    {'label': 'Small (S)', 'value': 'S'},
                                    {'label': 'Medium (M)', 'value': 'M'},
                                    {'label': 'Large (L)', 'value': 'L'}
                                ],
                                value='M',
                                className="mb-3"
                            )
                        ], md=6)
                    ]),
                    
                    # Row 3: Location & Work Year
                    dbc.Row([
                        dbc.Col([
                            html.Label("Company Location", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='home-location-dropdown',
                                options=[
                                    {'label': 'United States', 'value': 'US'},
                                    {'label': 'Canada', 'value': 'CA'},
                                    {'label': 'United Kingdom', 'value': 'GB'},
                                    {'label': 'Germany', 'value': 'DE'}
                                ],
                                value='US',
                                className="mb-3"
                            )
                        ], md=6),
                        dbc.Col([
                            html.Label("Work Year", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='home-work-year-dropdown',
                                options=[
                                    {'label': '2020', 'value': 2020},
                                    {'label': '2021', 'value': 2021},
                                    {'label': '2022', 'value': 2022},
                                    {'label': '2023', 'value': 2023},
                                    {'label': '2024', 'value': 2024}
                                ],
                                value=2023,
                                className="mb-3"
                            )
                        ], md=6)
                    ]),
                    
                    # Row 4: Remote Ratio
                    dbc.Row([
                        dbc.Col([
                            html.Label("Remote Work Ratio (%)", className="fw-bold mb-2"),
                            dcc.Slider(
                                id='home-remote-slider',
                                min=0,
                                max=100,
                                step=25,
                                value=50,
                                marks={
                                    0: '0%',
                                    25: '25%',
                                    50: '50%',
                                    75: '75%',
                                    100: '100%'
                                },
                                className="mb-4"
                            )
                        ])
                    ]),
                    
                    # Predict Button
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-magic me-2"),
                                "Prediksi Gaji"
                            ], 
                            id='home-predict-button',
                            color="primary",
                            size="lg",
                            className="w-100",
                            style={'background': colors['gradient'], 'border': 'none'})
                        ])
                    ])
                ])
            ], className="shadow-sm border-0")
        ], md=5),
        
        # Right Column - Results
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="fas fa-chart-line me-2"),
                        "Hasil Prediksi"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    html.Div(id='home-prediction-results', children=[
                        html.Div([
                            html.I(className="fas fa-robot fa-4x text-muted mb-3"),
                            html.H5("Siap Memprediksi", className="text-muted"),
                            html.P("Isi parameter di sebelah kiri dan klik 'Prediksi Gaji' untuk memulai.")
                        ], className="text-center py-5")
                    ])
                ])
            ], className="shadow-sm border-0")
        ], md=7)
    ], className="mb-5")

# Helper functions for labels
def get_experience_label(exp):
    mapping = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
    return mapping.get(exp, exp)

def get_employment_label(emp):
    mapping = {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}
    return mapping.get(emp, emp)

# Callback for home page prediction
@callback(
    Output('home-prediction-results', 'children'),
    Input('home-predict-button', 'n_clicks'),
    [
        State('home-work-year-dropdown', 'value'),
        State('home-experience-dropdown', 'value'),
        State('home-employment-dropdown', 'value'),
        State('home-job-title-dropdown', 'value'),
        State('home-location-dropdown', 'value'),
        State('home-company-size-dropdown', 'value'),
        State('home-remote-slider', 'value')
    ],
    prevent_initial_call=True
)
def predict_salary_home(n_clicks, work_year, experience, employment, job_title, location, company_size, remote_ratio):
    """Predict salary based on input parameters for home page"""
    if n_clicks is None:
        return html.Div([
            html.Div([
                html.I(className="fas fa-robot fa-4x text-muted mb-3"),
                html.H5("Siap Memprediksi", className="text-muted"),
                html.P("Isi parameter di sebelah kiri dan klik 'Prediksi Gaji' untuk memulai.")
            ], className="text-center py-5")
        ])
    
    if model is None:
        # Demo prediction
        base_salaries = {
            'Data Scientist': {'EN': 70000, 'MI': 95000, 'SE': 125000, 'EX': 180000},
            'Data Engineer': {'EN': 75000, 'MI': 100000, 'SE': 130000, 'EX': 190000},
            'Data Analyst': {'EN': 60000, 'MI': 80000, 'SE': 105000, 'EX': 150000},
            'Machine Learning Engineer': {'EN': 85000, 'MI': 115000, 'SE': 150000, 'EX': 220000},
            'Default': {'EN': 65000, 'MI': 85000, 'SE': 115000, 'EX': 165000}
        }
        
        job_salaries = base_salaries.get(job_title, base_salaries['Default'])
        base_salary = job_salaries.get(experience, job_salaries['SE'])
        
        location_multiplier = {
            'US': 1.0, 'CA': 0.85, 'GB': 0.9, 'DE': 0.8
        }
        
        size_multiplier = {'S': 0.9, 'M': 1.0, 'L': 1.1}
        remote_adjustment = 1.0 + (remote_ratio - 50) * 0.002
        
        predicted_salary = int(base_salary * 
                             location_multiplier.get(location, 0.7) * 
                             size_multiplier.get(company_size, 1.0) * 
                             remote_adjustment)
    else:
        try:
            # Real model prediction
            input_data = pd.DataFrame({
                'work_year': [work_year],
                'experience_level': [experience],
                'employment_type': [employment],
                'job_title': [job_title],
                'company_location': [location],
                'company_size': [company_size],
                'remote_ratio': [remote_ratio]
            })
            
            predicted_salary = model.predict(input_data)[0]
            
        except Exception as e:
            return dbc.Alert([
                html.H5("Prediksi Error", className="alert-heading"),
                html.P(f"Error: {str(e)}"),
                html.P("Silakan periksa file model dan parameter input.")
            ], color="danger")
    
    # Categorize salary
    if predicted_salary < 50000:
        category = "Low"
        category_color = "warning"
        category_icon = "fas fa-arrow-down"
    elif predicted_salary <= 100000:
        category = "Medium"
        category_color = "info"
        category_icon = "fas fa-minus"
    else:
        category = "High"
        category_color = "success"
        category_icon = "fas fa-arrow-up"
    
    return html.Div([
        # Main Prediction
        html.Div([
            html.H2(f"${predicted_salary:,.0f}", 
                   className="display-4 fw-bold text-primary mb-2"),
            html.H5("Prediksi Gaji Tahunan", className="text-muted mb-3"),
            
            dbc.Badge([
                html.I(className=f"{category_icon} me-2"),
                f"{category} Range"
            ], color=category_color, className="mb-4 p-3 fs-6")
        ], className="text-center mb-4"),
        
        # Salary Breakdown
        html.Hr(),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Small("Bulanan", className="text-muted d-block"),
                    html.Strong(f"${predicted_salary/12:,.0f}", className="h5")
                ], className="text-center"),
                dbc.Col([
                    html.Small("Mingguan", className="text-muted d-block"),
                    html.Strong(f"${predicted_salary/52:,.0f}", className="h5")
                ], className="text-center"),
                dbc.Col([
                    html.Small("Harian", className="text-muted d-block"),
                    html.Strong(f"${predicted_salary/365:,.0f}", className="h5")
                ], className="text-center")
            ])
        ], className="mt-3"),
        
        # Input Summary
        html.Hr(),
        html.Div([
            html.H6("Parameter Input:", className="fw-bold mb-2"),
            html.Div([
                dbc.Badge(f"Experience: {get_experience_label(experience)}", color="light", text_color="dark", className="me-2 mb-1"),
                dbc.Badge(f"Employment: {get_employment_label(employment)}", color="light", text_color="dark", className="me-2 mb-1"),
                dbc.Badge(f"Company: {company_size}", color="light", text_color="dark", className="me-2 mb-1"),
                dbc.Badge(f"Remote: {remote_ratio}%", color="light", text_color="dark", className="me-2 mb-1"),
                dbc.Badge(f"Location: {location}", color="light", text_color="dark", className="me-2 mb-1")
            ])
        ], className="mt-3"),
        
        # Confidence Note
        html.Hr(),
        html.Div([
            html.I(className=f"fas fa-{'shield-alt' if model else 'exclamation-triangle'} me-2 {'text-success' if model else 'text-warning'}"),
            html.Small("Prediksi berdasarkan Random Forest model" if model else "⚠️ DEMO MODE: Prediksi deterministik",
                      className="text-muted")
        ], className="text-center mt-3"),
        
        # Action Buttons
        html.Div([
            dbc.Button([
                html.I(className="fas fa-chart-line me-2"),
                "Lihat Detail"
            ], href="/prediction", color="outline-primary", size="sm", className="me-2"),
            dbc.Button([
                html.I(className="fas fa-share me-2"),
                "Bagikan"
            ], color="outline-secondary", size="sm")
        ], className="text-center mt-3")
    ])