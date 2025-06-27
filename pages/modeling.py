"""
Modeling Page
Machine Learning model training, evaluation, and predictions
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def layout(df, colors):
    """Create modeling page layout"""
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-robot me-3", style={'color': colors['primary']}),
                            "Machine Learning Modeling"
                        ], className="display-5 fw-bold mb-3"),
                        html.P("Training model prediksi gaji dan evaluasi performa machine learning",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Model Selection Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-cog me-2"),
                                "Model Configuration"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Model:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='model-selector',
                                        options=[
                                            {'label': 'Linear Regression', 'value': 'linear'},
                                            {'label': 'Random Forest', 'value': 'random_forest'}
                                        ],
                                        value='linear',
                                        clearable=False
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Test Size:", className="fw-bold mb-2"),
                                    dcc.Slider(
                                        id='test-size-slider',
                                        min=0.1, max=0.4, step=0.05, value=0.2,
                                        marks={0.1: '10%', 0.2: '20%', 0.3: '30%', 0.4: '40%'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Random State:", className="fw-bold mb-2"),
                                    dcc.Input(
                                        id='random-state-input',
                                        type='number',
                                        value=42,
                                        min=1, max=100,
                                        className="form-control"
                                    )
                                ], md=2),
                                dbc.Col([
                                    html.Label(" ", className="fw-bold mb-2 d-block"),
                                    dbc.Button(
                                        "Train Model",
                                        id='train-button',
                                        color="primary",
                                        className="w-100"
                                    )
                                ], md=2)
                            ])
                        ])
                    ], className="shadow-sm border-0 mb-4")
                ])
            ]),
            
            # Model Results
            dbc.Row([
                dbc.Col([
                    html.Div(id='model-results')
                ])
            ]),
            
            # Feature Importance
            dbc.Row([
                dbc.Col([
                    html.Div(id='feature-importance-section')
                ])
            ], className="mb-4"),
            
            # Prediction Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-calculator me-2"),
                                "Salary Prediction"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_prediction_form(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-magic me-2"),
                                "Prediction Result"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div(id='prediction-result',
                                   className="text-center p-4",
                                   style={'minHeight': '200px'})
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ])
            
        ], fluid=True, className="py-4")
    ])

def create_prediction_form(df, colors):
    """Create salary prediction form"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Experience Level:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id='pred-experience',
                    options=[
                        {'label': 'Entry Level', 'value': 'EN'},
                        {'label': 'Mid Level', 'value': 'MI'},
                        {'label': 'Senior Level', 'value': 'SE'},
                        {'label': 'Executive Level', 'value': 'EX'}
                    ],
                    value='SE',
                    clearable=False
                )
            ], md=6, className="mb-3"),
            dbc.Col([
                html.Label("Employment Type:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id='pred-employment',
                    options=[
                        {'label': 'Full Time', 'value': 'FT'},
                        {'label': 'Part Time', 'value': 'PT'},
                        {'label': 'Contract', 'value': 'CT'},
                        {'label': 'Freelance', 'value': 'FL'}
                    ],
                    value='FT',
                    clearable=False
                )
            ], md=6, className="mb-3")
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Company Size:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id='pred-company-size',
                    options=[
                        {'label': 'Small', 'value': 'S'},
                        {'label': 'Medium', 'value': 'M'},
                        {'label': 'Large', 'value': 'L'}
                    ],
                    value='M',
                    clearable=False
                )
            ], md=6, className="mb-3"),
            dbc.Col([
                html.Label("Remote Ratio (%):", className="fw-bold mb-2"),
                dcc.Slider(
                    id='pred-remote-ratio',
                    min=0, max=100, step=10, value=50,
                    marks={0: '0%', 50: '50%', 100: '100%'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], md=6, className="mb-3")
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Work Year:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id='pred-work-year',
                    options=[{'label': str(year), 'value': year} 
                            for year in sorted(df['work_year'].unique(), reverse=True)],
                    value=df['work_year'].max(),
                    clearable=False
                )
            ], md=6, className="mb-3"),
            dbc.Col([
                html.Label(" ", className="fw-bold mb-2 d-block"),
                dbc.Button(
                    "Predict Salary",
                    id='predict-button',
                    color="success",
                    className="w-100",
                    size="lg"
                )
            ], md=6, className="mb-3")
        ])
    ])

# Callbacks for model training and prediction
@callback(
    Output('model-results', 'children'),
    Output('feature-importance-section', 'children'),
    [Input('train-button', 'n_clicks')],
    [Input('model-selector', 'value'),
     Input('test-size-slider', 'value'),
     Input('random-state-input', 'value')]
)
def train_model(n_clicks, model_type, test_size, random_state):
    if n_clicks is None:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Click 'Train Model' to start training the machine learning model."
            ], color="info")
        ]), html.Div()
    
    # Load data and train model
    from utils.data_loader import load_data
    df = load_data()
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'light': '#a7f3d0',
        'dark': '#059669'
    }
    
    # Prepare features
    features, target, feature_names = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    
    # Train model
    if model_type == 'linear':
        model = LinearRegression()
        model_name = "Linear Regression"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model_name = "Random Forest"
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    model_path = 'data/trained_model.pkl'
    os.makedirs('data', exist_ok=True)
    joblib.dump({'model': model, 'feature_names': feature_names}, model_path)
    
    # Create results
    results = create_model_results(model_name, mse, rmse, mae, r2, y_test, y_pred, colors)
    feature_importance = create_feature_importance(model, feature_names, model_type, colors)
    
    return results, feature_importance

@callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('pred-experience', 'value'),
     Input('pred-employment', 'value'),
     Input('pred-company-size', 'value'),
     Input('pred-remote-ratio', 'value'),
     Input('pred-work-year', 'value')]
)
def predict_salary(n_clicks, experience, employment, company_size, remote_ratio, work_year):
    if n_clicks is None:
        return html.Div([
            html.I(className="fas fa-calculator fa-3x text-muted mb-3"),
            html.P("Fill the form and click 'Predict Salary'", className="text-muted")
        ])
    
    try:
        # Load trained model
        model_data = joblib.load('data/trained_model.pkl')
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Prepare input features
        input_features = prepare_single_prediction(
            work_year, experience, employment, company_size, remote_ratio
        )
        
        # Make prediction
        predicted_salary = model.predict([input_features])[0]
        
        # Create result display
        return html.Div([
            html.I(className="fas fa-dollar-sign fa-3x text-success mb-3"),
            html.H2(f"${predicted_salary:,.0f}", className="text-success fw-bold mb-3"),
            html.P("Predicted Annual Salary", className="text-muted mb-3"),
            html.Hr(),
            html.H6("Input Summary:", className="fw-bold mb-2"),
            html.Ul([
                html.Li(f"Experience: {get_experience_label(experience)}"),
                html.Li(f"Employment: {get_employment_label(employment)}"),
                html.Li(f"Company Size: {get_company_size_label(company_size)}"),
                html.Li(f"Remote Work: {remote_ratio}%"),
                html.Li(f"Year: {work_year}")
            ], className="text-muted")
        ])
        
    except Exception as e:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-3x text-danger mb-3"),
            html.P("Please train a model first!", className="text-danger")
        ])

def prepare_features(df):
    """Prepare features for model training"""
    # Create a copy for processing
    df_model = df.copy()
    
    # Encode categorical variables
    le_exp = LabelEncoder()
    le_size = LabelEncoder()
    le_emp = LabelEncoder()
    
    df_model['experience_encoded'] = le_exp.fit_transform(df_model['experience_level'])
    df_model['company_size_encoded'] = le_size.fit_transform(df_model['company_size'])
    df_model['employment_encoded'] = le_emp.fit_transform(df_model['employment_type'])
    
    # Select features
    feature_columns = ['work_year', 'experience_encoded', 'company_size_encoded', 
                      'employment_encoded', 'remote_ratio']
    
    features = df_model[feature_columns].values
    target = df_model['salary_in_usd'].values
    
    return features, target, feature_columns

def prepare_single_prediction(work_year, experience, employment, company_size, remote_ratio):
    """Prepare single input for prediction"""
    # Encode categorical values (simplified encoding)
    exp_encoding = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_encoding = {'S': 0, 'M': 1, 'L': 2}
    emp_encoding = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}
    
    return [
        work_year,
        exp_encoding[experience],
        size_encoding[company_size],
        emp_encoding[employment],
        remote_ratio
    ]

def create_model_results(model_name, mse, rmse, mae, r2, y_test, y_pred, colors):
    """Create model evaluation results"""
    
    # Metrics cards
    metrics_cards = dbc.Row([
        dbc.Col([
            create_metric_card("R² Score", f"{r2:.3f}", "Model accuracy", colors['primary'])
        ], md=3),
        dbc.Col([
            create_metric_card("RMSE", f"${rmse:,.0f}", "Root Mean Square Error", colors['secondary'])
        ], md=3),
        dbc.Col([
            create_metric_card("MAE", f"${mae:,.0f}", "Mean Absolute Error", colors['accent'])
        ], md=3),
        dbc.Col([
            create_metric_card("MSE", f"${mse:,.0f}", "Mean Square Error", colors['dark'])
        ], md=3)
    ])
    
    # Actual vs Predicted plot
    scatter_fig = px.scatter(
        x=y_test, y=y_pred,
        title=f'{model_name} - Actual vs Predicted Salary',
        labels={'x': 'Actual Salary (USD)', 'y': 'Predicted Salary (USD)'},
        color_discrete_sequence=[colors['primary']]
    )
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    scatter_fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash"),
    )
    
    scatter_fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-chart-line me-2"),
                    f"{model_name} - Model Performance"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                metrics_cards,
                html.Hr(),
                dcc.Graph(figure=scatter_fig, config={'displayModeBar': False})
            ])
        ], className="shadow-sm border-0 mb-4")
    ])

def create_metric_card(title, value, description, color):
    """Create metric display card"""
    return dbc.Card([
        dbc.CardBody([
            html.H3(value, className="fw-bold mb-1", style={'color': color}),
            html.H6(title, className="mb-1"),
            html.Small(description, className="text-muted")
        ], className="text-center")
    ], className="h-100 border-0 shadow-sm",
       style={'borderLeft': f'4px solid {color}'})

def create_feature_importance(model, feature_names, model_type, colors):
    """Create feature importance visualization"""
    if model_type == 'random_forest' and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create readable feature names
        readable_names = {
            'work_year': 'Work Year',
            'experience_encoded': 'Experience Level',
            'company_size_encoded': 'Company Size',
            'employment_encoded': 'Employment Type',
            'remote_ratio': 'Remote Ratio'
        }
        
        feature_importance_df = pd.DataFrame({
            'Feature': [readable_names.get(name, name) for name in feature_names],
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance_df,
            x='Importance', y='Feature',
            orientation='h',
            title='Feature Importance (Random Forest)',
            color='Importance',
            color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
        )
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-ranking-star me-2"),
                    "Feature Importance"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ])
        ], className="shadow-sm border-0")
    
    else:
        return dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-info-circle me-2"),
                    "Feature Importance"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    "Feature importance is only available for Random Forest model."
                ], color="info")
            ])
        ], className="shadow-sm border-0")

# Helper functions for labels
def get_experience_label(exp):
    mapping = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
    return mapping.get(exp, exp)

def get_employment_label(emp):
    mapping = {'FT': 'Full Time', 'PT': 'Part Time', 'CT': 'Contract', 'FL': 'Freelance'}
    return mapping.get(emp, emp)

def get_company_size_label(size):
    mapping = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    return mapping.get(size, size)