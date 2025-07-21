"""
Salary Prediction Page
Interactive salary prediction using trained ML model
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('data/model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    model = None

def layout(df, colors, lang='id'):
    """Create prediction page layout"""
    
    return dbc.Container([
        
        # Header Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2([
                        html.I(className="fas fa-crystal-ball me-3", style={'color': colors['primary']}),
                        "Salary Prediction"
                    ], className="text-center mb-3", style={'color': colors['darker']}),
                    html.P([
                        "Prediksi gaji Data Scientist berdasarkan pengalaman, lokasi, dan faktor lainnya. ",
                        "Model ini menggunakan Random Forest dengan akurasi 100% pada data kategori gaji."
                    ], className="text-center text-muted mb-4")
                ], className="mb-5")
            ])
        ]),
        
        dbc.Row([
            # Input Form Section
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-sliders-h me-2"),
                            "Input Parameters"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        # Work Year
                        dbc.Row([
                            dbc.Col([
                                html.Label("Work Year", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='work-year-dropdown',
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
                            ], md=6),
                            dbc.Col([
                                html.Label("Experience Level", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='experience-dropdown',
                                    options=[
                                        {'label': 'Entry Level', 'value': 'EN'},
                                        {'label': 'Mid Level', 'value': 'MI'},
                                        {'label': 'Senior Level', 'value': 'SE'},
                                        {'label': 'Executive Level', 'value': 'EX'}
                                    ],
                                    value='SE',
                                    className="mb-3"
                                )
                            ], md=6)
                        ]),
                        
                        # Employment Type & Job Title
                        dbc.Row([
                            dbc.Col([
                                html.Label("Employment Type", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='employment-dropdown',
                                    options=[
                                        {'label': 'Full-time', 'value': 'FT'},
                                        {'label': 'Part-time', 'value': 'PT'},
                                        {'label': 'Contract', 'value': 'CT'},
                                        {'label': 'Freelance', 'value': 'FL'}
                                    ],
                                    value='FT',
                                    className="mb-3"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Job Title", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='job-title-dropdown',
                                    options=get_job_title_options(df),
                                    value='Data Scientist',
                                    className="mb-3"
                                )
                            ], md=6)
                        ]),
                        
                        # Company Location & Size
                        dbc.Row([
                            dbc.Col([
                                html.Label("Company Location", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='location-dropdown',
                                    options=get_location_options(df),
                                    value='US',
                                    className="mb-3"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Company Size", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='company-size-dropdown',
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
                        
                        # Remote Ratio
                        dbc.Row([
                            dbc.Col([
                                html.Label("Remote Work Ratio (%)", className="fw-bold mb-2"),
                                dcc.Slider(
                                    id='remote-slider',
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
                                    "Predict Salary"
                                ], 
                                id='predict-button',
                                color="primary",
                                size="lg",
                                className="w-100",
                                style={'background': colors['gradient']})
                            ])
                        ])
                    ])
                ], className="shadow-sm border-0")
            ], md=5),
            
            # Results Section
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-line me-2"),
                            "Prediction Results"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id='prediction-results', children=[
                            html.Div([
                                html.I(className="fas fa-robot fa-4x text-muted mb-3"),
                                html.H5("Ready to Predict", className="text-muted"),
                                html.P("Fill in the parameters and click 'Predict Salary' to get started.")
                            ], className="text-center py-5")
                        ])
                    ])
                ], className="shadow-sm border-0")
            ], md=7)
        ], className="mb-5"),
        
        # Model Information
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-info-circle me-2"),
                            "Model Information"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🤖 Algorithm", className="fw-bold text-primary"),
                                html.P("Random Forest Regressor", className="mb-3")
                            ], md=3),
                            dbc.Col([
                                html.H6("📊 Accuracy", className="fw-bold text-success"),
                                html.P("100% Category Accuracy", className="mb-3")
                            ], md=3),
                            dbc.Col([
                                html.H6("📈 R² Score", className="fw-bold text-info"),
                                html.P("0.9999", className="mb-3")
                            ], md=3),
                            dbc.Col([
                                html.H6("🎯 Dataset", className="fw-bold text-warning"),
                                html.P("3,755 Records", className="mb-3")
                            ], md=3)
                        ])
                    ])
                ], className="shadow-sm border-0", 
                   style={'background': f'linear-gradient(145deg, #ffffff, {colors["light"]}10)'})
            ])
        ])
        
    ], fluid=True, className="py-4")

def get_job_title_options(df):
    """Get job title options from dataset"""
    job_titles = sorted(df['job_title'].unique())
    return [{'label': title, 'value': title} for title in job_titles[:20]]  # Top 20

def get_location_options(df):
    """Get location options from dataset"""
    locations = sorted(df['company_location'].unique())
    return [{'label': loc, 'value': loc} for loc in locations]

# Callback for prediction
@callback(
    Output('prediction-results', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        State('work-year-dropdown', 'value'),
        State('experience-dropdown', 'value'),
        State('employment-dropdown', 'value'),
        State('job-title-dropdown', 'value'),
        State('location-dropdown', 'value'),
        State('company-size-dropdown', 'value'),
        State('remote-slider', 'value')
    ],
    prevent_initial_call=True
)
def predict_salary(n_clicks, work_year, experience, employment, job_title, location, company_size, remote_ratio):
    """Predict salary based on input parameters"""
    if n_clicks is None or model is None:
        return html.Div("Model not available")
    
    try:
        # Create input DataFrame (sesuaikan dengan fitur yang digunakan model Anda)
        input_data = pd.DataFrame({
            'work_year': [work_year],
            'experience_level': [experience],
            'employment_type': [employment],
            'job_title': [job_title],
            'company_location': [location],
            'company_size': [company_size],
            'remote_ratio': [remote_ratio]
        })
        
        # Make prediction
        predicted_salary = model.predict(input_data)[0]
        
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
                html.H5("Predicted Annual Salary", className="text-muted mb-3"),
                
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
                        html.Small("Monthly", className="text-muted d-block"),
                        html.Strong(f"${predicted_salary/12:,.0f}", className="h5")
                    ], className="text-center"),
                    dbc.Col([
                        html.Small("Weekly", className="text-muted d-block"),
                        html.Strong(f"${predicted_salary/52:,.0f}", className="h5")
                    ], className="text-center"),
                    dbc.Col([
                        html.Small("Daily", className="text-muted d-block"),
                        html.Strong(f"${predicted_salary/365:,.0f}", className="h5")
                    ], className="text-center")
                ])
            ], className="mt-3"),
            
            # Confidence Note
            html.Hr(),
            html.Div([
                html.I(className="fas fa-shield-alt me-2 text-success"),
                html.Small("Prediction based on Random Forest model with 100% category accuracy",
                          className="text-muted")
            ], className="text-center mt-3")
        ])
        
    except Exception as e:
        return dbc.Alert([
            html.H5("Prediction Error", className="alert-heading"),
            html.P(f"Error: {str(e)}"),
            html.P("Please check your model file and input parameters.")
        ], color="danger")