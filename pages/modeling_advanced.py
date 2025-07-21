"""
Advanced Modeling Page
Comprehensive ML model comparison, evaluation, and analysis
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                           explained_variance_score, max_error)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def layout(df, colors, lang='id'):
    """Create advanced modeling page layout"""
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-brain me-3", style={'color': colors['primary']}),
                            "Advanced ML Model Analysis"
                        ], className="display-5 fw-bold mb-3"),
                        html.P("Comprehensive model comparison, evaluation, dan hyperparameter tuning dengan cross-validation",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Model Comparison Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-balance-scale me-2"),
                                "Multi-Algorithm Comparison"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Test Size:", className="fw-bold mb-2"),
                                    dcc.Slider(
                                        id='adv-test-size',
                                        min=0.15, max=0.35, step=0.05, value=0.2,
                                        marks={0.15: '15%', 0.2: '20%', 0.25: '25%', 0.3: '30%', 0.35: '35%'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Cross-Validation Folds:", className="fw-bold mb-2"),
                                    dcc.Slider(
                                        id='cv-folds',
                                        min=3, max=10, step=1, value=5,
                                        marks={3: '3', 5: '5', 7: '7', 10: '10'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label(" ", className="fw-bold mb-2 d-block"),
                                    dbc.Button(
                                        [html.I(className="fas fa-play me-2"), "Run Comparison"],
                                        id='run-comparison-btn',
                                        color="primary",
                                        className="w-100"
                                    )
                                ], md=4)
                            ])
                        ])
                    ], className="shadow-sm border-0 mb-4")
                ])
            ]),
            
            # Model Comparison Results
            dbc.Row([
                dbc.Col([
                    html.Div(id='model-comparison-results')
                ])
            ]),
            
            # Best Model Analysis
            dbc.Row([
                dbc.Col([
                    html.Div(id='best-model-analysis')
                ])
            ]),
            
            # Hyperparameter Tuning Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-cogs me-2"),
                                "Hyperparameter Tuning"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Model for Tuning:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='tuning-model-selector',
                                        options=[
                                            {'label': 'Random Forest', 'value': 'rf'},
                                            {'label': 'Gradient Boosting', 'value': 'gb'},
                                            {'label': 'Support Vector Regressor', 'value': 'svr'}
                                        ],
                                        value='rf',
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label(" ", className="fw-bold mb-2 d-block"),
                                    dbc.Button(
                                        [html.I(className="fas fa-search me-2"), "Start Tuning"],
                                        id='start-tuning-btn',
                                        color="success",
                                        className="w-100"
                                    )
                                ], md=6)
                            ])
                        ])
                    ], className="shadow-sm border-0 mb-4")
                ])
            ]),
            
            # Tuning Results
            dbc.Row([
                dbc.Col([
                    html.Div(id='tuning-results')
                ])
            ]),
            
            # Model Interpretability
            dbc.Row([
                dbc.Col([
                    html.Div(id='model-interpretability')
                ])
            ]),
            
            # Advanced Prediction Interface
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-crystal-ball me-2"),
                                "Advanced Prediction Interface"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_advanced_prediction_interface(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_advanced_prediction_interface(df, colors):
    """Create advanced prediction interface with multiple models"""
    return dbc.Row([
        dbc.Col([
            html.H5("Input Features", className="fw-bold mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Experience Level:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='adv-pred-experience',
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
                        id='adv-pred-employment',
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
                        id='adv-pred-company-size',
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
                    html.Label("Work Year:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='adv-pred-work-year',
                        options=[{'label': str(year), 'value': year} 
                                for year in sorted(df['work_year'].unique(), reverse=True)],
                        value=df['work_year'].max(),
                        clearable=False
                    )
                ], md=6, className="mb-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Remote Ratio (%):", className="fw-bold mb-2"),
                    dcc.Slider(
                        id='adv-pred-remote-ratio',
                        min=0, max=100, step=10, value=50,
                        marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=8, className="mb-3"),
                dbc.Col([
                    html.Label(" ", className="fw-bold mb-2 d-block"),
                    dbc.Button(
                        [html.I(className="fas fa-magic me-2"), "Predict All Models"],
                        id='adv-predict-btn',
                        color="primary",
                        className="w-100"
                    )
                ], md=4, className="mb-3")
            ])
        ], md=5),
        
        dbc.Col([
            html.H5("Model Predictions Comparison", className="fw-bold mb-3"),
            html.Div(id='adv-prediction-results', className="bg-light p-3 rounded")
        ], md=7)
    ])

# Advanced Callbacks
@callback(
    Output('model-comparison-results', 'children'),
    [Input('run-comparison-btn', 'n_clicks')],
    [State('adv-test-size', 'value'),
     State('cv-folds', 'value')]
)
def run_model_comparison(n_clicks, test_size, cv_folds):
    if n_clicks is None:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Click 'Run Comparison' to compare multiple ML algorithms with cross-validation."
            ], color="info")
        ])
    
    # Load data
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
    features, target, feature_names = prepare_features_advanced(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42
    )
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regressor': SVR(kernel='rbf', C=1.0)
    }
    
    # Train and evaluate models
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        
        # Train on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        
        results.append({
            'Model': name,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            'Test R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Explained Variance': explained_var,
            'Max Error': max_err
        })
        
        trained_models[name] = model
    
    # Save best model
    best_model_name = max(results, key=lambda x: x['Test R²'])['Model']
    best_model = trained_models[best_model_name]
    
    model_data = {
        'model': best_model,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'all_models': trained_models
    }
    
    os.makedirs('data', exist_ok=True)
    joblib.dump(model_data, 'data/advanced_models.pkl')
    
    # Create results visualization
    return create_comparison_visualization(results, colors)

@callback(
    Output('best-model-analysis', 'children'),
    [Input('model-comparison-results', 'children')]
)
def update_best_model_analysis(comparison_results):
    if comparison_results is None:
        return html.Div()
    
    try:
        # Load model data
        model_data = joblib.load('data/advanced_models.pkl')
        best_model = model_data['model']
        model_name = model_data['model_name']
        
        colors = {
            'primary': '#10b981',
            'secondary': '#34d399',
            'accent': '#6ee7b7',
            'dark': '#059669'
        }
        
        return create_best_model_analysis(best_model, model_name, colors)
        
    except:
        return html.Div()

@callback(
    Output('tuning-results', 'children'),
    [Input('start-tuning-btn', 'n_clicks')],
    [State('tuning-model-selector', 'value')]
)
def perform_hyperparameter_tuning(n_clicks, model_type):
    if n_clicks is None:
        return html.Div()
    
    # Load data
    from utils.data_loader import load_data
    df = load_data()
    features, target, feature_names = prepare_features_advanced(df)
    
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'dark': '#059669'
    }
    
    # Define parameter grids
    param_grids = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gb': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        },
        'svr': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
    }
    
    models = {
        'rf': RandomForestRegressor(random_state=42),
        'gb': GradientBoostingRegressor(random_state=42),
        'svr': SVR()
    }
    
    model = models[model_type]
    param_grid = param_grids[model_type]
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(features, target)
    
    return create_tuning_results(grid_search, model_type, colors)

@callback(
    Output('model-interpretability', 'children'),
    [Input('tuning-results', 'children'),
     Input('best-model-analysis', 'children')]
)
def update_model_interpretability(tuning_results, best_model_analysis):
    if best_model_analysis is None:
        return html.Div()
    
    try:
        model_data = joblib.load('data/advanced_models.pkl')
        best_model = model_data['model']
        feature_names = model_data['feature_names']
        
        colors = {
            'primary': '#10b981',
            'secondary': '#34d399',
            'accent': '#6ee7b7',
            'dark': '#059669'
        }
        
        return create_model_interpretability_section(best_model, feature_names, colors)
        
    except:
        return html.Div()

@callback(
    Output('adv-prediction-results', 'children'),
    [Input('adv-predict-btn', 'n_clicks')],
    [State('adv-pred-experience', 'value'),
     State('adv-pred-employment', 'value'),
     State('adv-pred-company-size', 'value'),
     State('adv-pred-remote-ratio', 'value'),
     State('adv-pred-work-year', 'value')]
)
def advanced_prediction(n_clicks, experience, employment, company_size, remote_ratio, work_year):
    if n_clicks is None:
        return html.Div([
            html.I(className="fas fa-chart-line fa-3x text-muted mb-3"),
            html.P("Fill the form and click 'Predict All Models' to see predictions from multiple algorithms.", 
                   className="text-muted text-center")
        ])
    
    try:
        model_data = joblib.load('data/advanced_models.pkl')
        all_models = model_data['all_models']
        
        # Prepare input
        input_features = prepare_single_prediction_advanced(
            work_year, experience, employment, company_size, remote_ratio
        )
        
        predictions = {}
        for name, model in all_models.items():
            pred = model.predict([input_features])[0]
            predictions[name] = pred
        
        return create_prediction_comparison(predictions)
        
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Please run model comparison first to train the models."
            ], color="warning")
        ])

# Helper Functions
def prepare_features_advanced(df):
    """Prepare features for advanced modeling"""
    df_model = df.copy()
    
    # Encode categorical variables
    categorical_mappings = {
        'experience_level': {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4},
        'employment_type': {'PT': 1, 'FL': 2, 'CT': 3, 'FT': 4},
        'company_size': {'S': 1, 'M': 2, 'L': 3}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_model.columns:
            df_model[f'{col}_encoded'] = df_model[col].map(mapping)
    
    # Select features
    feature_columns = ['work_year', 'remote_ratio']
    for col in categorical_mappings.keys():
        if f'{col}_encoded' in df_model.columns:
            feature_columns.append(f'{col}_encoded')
    
    features = df_model[feature_columns].values
    target = df_model['salary_in_usd'].values
    
    return features, target, feature_columns

def prepare_single_prediction_advanced(work_year, experience, employment, company_size, remote_ratio):
    """Prepare single prediction input for advanced models"""
    exp_mapping = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
    emp_mapping = {'PT': 1, 'FL': 2, 'CT': 3, 'FT': 4}
    size_mapping = {'S': 1, 'M': 2, 'L': 3}
    
    return [
        work_year,
        remote_ratio,
        exp_mapping[experience],
        emp_mapping[employment],
        size_mapping[company_size]
    ]

def create_comparison_visualization(results, colors):
    """Create model comparison visualization"""
    df_results = pd.DataFrame(results)
    
    # Sort by Test R²
    df_results = df_results.sort_values('Test R²', ascending=True)
    
    # Create comparison chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Performance (R²)', 'Cross-Validation Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² comparison
    fig.add_trace(
        go.Bar(
            x=df_results['Test R²'],
            y=df_results['Model'],
            orientation='h',
            name='Test R²',
            marker_color=colors['primary']
        ),
        row=1, col=1
    )
    
    # CV scores with error bars
    fig.add_trace(
        go.Bar(
            x=df_results['CV R² Mean'],
            y=df_results['Model'],
            orientation='h',
            name='CV R² Mean',
            marker_color=colors['secondary'],
            error_x=dict(type='data', array=df_results['CV R² Std'])
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text="Model Performance Comparison",
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create metrics table
    metrics_table = dash_table.DataTable(
        data=df_results.round(4).to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_results.columns],
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'row_index': len(df_results) - 1},  # Best model (last row after sorting)
                'backgroundColor': colors['light'],
                'color': 'black',
                'fontWeight': 'bold'
            }
        ]
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-trophy me-2"),
                "Model Comparison Results"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.Hr(),
            html.H5("Detailed Metrics", className="mb-3"),
            metrics_table,
            html.Div([
                dbc.Badge([
                    html.I(className="fas fa-star me-2"),
                    f"Best Model: {df_results.iloc[-1]['Model']} (R² = {df_results.iloc[-1]['Test R²']:.4f})"
                ], color="success", className="p-3 fs-6 mt-3")
            ], className="text-center")
        ])
    ], className="shadow-sm border-0 mb-4")

def create_best_model_analysis(model, model_name, colors):
    """Create detailed analysis of the best model"""
    analysis_components = []
    
    # Model info card
    model_info = dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-medal me-2"),
                f"Best Model: {model_name}"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P(f"Model Type: {type(model).__name__}", className="mb-2"),
            html.P(f"Algorithm: {model_name}", className="mb-2"),
            get_model_description(model_name)
        ])
    ], className="shadow-sm border-0 mb-3")
    
    analysis_components.append(model_info)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Work Year', 'Remote Ratio', 'Experience Level', 'Employment Type', 'Company Size']
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {model_name}',
            color='Importance',
            color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
        )
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        importance_card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Feature Importance Analysis"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ])
        ], className="shadow-sm border-0 mb-3")
        
        analysis_components.append(importance_card)
    
    return html.Div(analysis_components)

def create_tuning_results(grid_search, model_type, colors):
    """Create hyperparameter tuning results"""
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Create parameter importance analysis
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Create visualization of parameter effects
    fig = go.Figure()
    
    # Simple visualization of best parameters
    param_names = list(best_params.keys())
    param_values = [str(v) for v in best_params.values()]
    
    fig.add_trace(go.Bar(
        x=param_names,
        y=[1] * len(param_names),
        text=param_values,
        textposition='auto',
        marker_color=colors['primary']
    ))
    
    fig.update_layout(
        title=f"Best Hyperparameters for {model_type.upper()}",
        xaxis_title="Parameters",
        yaxis_title="Optimal Values",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-bullseye me-2"),
                "Hyperparameter Tuning Results"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Alert([
                html.H5(f"Best CV Score: {best_score:.4f}", className="alert-heading"),
                html.P("Optimal hyperparameters found through grid search:", className="mb-2"),
                html.Ul([
                    html.Li(f"{param}: {value}") 
                    for param, value in best_params.items()
                ])
            ], color="success"),
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
    ], className="shadow-sm border-0 mb-4")

def create_model_interpretability_section(model, feature_names, colors):
    """Create model interpretability analysis"""
    interpretability_components = []
    
    # Model coefficients (for linear models)
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': ['Work Year', 'Remote Ratio', 'Experience Level', 'Employment Type', 'Company Size'],
            'Coefficient': model.coef_
        })
        
        fig = px.bar(
            coef_df,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Model Coefficients (Linear Relationship)',
            color='Coefficient',
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        coef_card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-functions me-2"),
                    "Model Coefficients"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(figure=fig, config={'displayModeBar': False}),
                html.P("Positive coefficients increase salary, negative coefficients decrease salary.", 
                       className="small text-muted mt-2")
            ])
        ], className="shadow-sm border-0 mb-3")
        
        interpretability_components.append(coef_card)
    
    # Feature importance (for ensemble models)
    elif hasattr(model, 'feature_importances_'):
        importance_card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-weight-hanging me-2"),
                    "Model Interpretability"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.P("This ensemble model uses feature importance to determine salary predictions:", className="mb-3"),
                create_feature_importance_explanation(model.feature_importances_)
            ])
        ], className="shadow-sm border-0 mb-3")
        
        interpretability_components.append(importance_card)
    
    return html.Div(interpretability_components)

def create_feature_importance_explanation(importances):
    """Create feature importance explanation"""
    feature_names = ['Work Year', 'Remote Ratio', 'Experience Level', 'Employment Type', 'Company Size']
    
    explanations = []
    for i, (feature, importance) in enumerate(zip(feature_names, importances)):
        percentage = importance * 100
        
        if percentage > 30:
            impact = "Very High"
            color = "danger"
        elif percentage > 20:
            impact = "High"
            color = "warning"
        elif percentage > 10:
            impact = "Medium"
            color = "info"
        else:
            impact = "Low"
            color = "secondary"
        
        explanations.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Strong(feature),
                    dbc.Badge(f"{percentage:.1f}%", color=color, className="ms-2"),
                    dbc.Badge(impact, color=color, className="ms-1")
                ])
            ])
        )
    
    return dbc.ListGroup(explanations)

def create_prediction_comparison(predictions):
    """Create prediction comparison visualization"""
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    cards = []
    for i, (model_name, prediction) in enumerate(sorted_predictions):
        if i == 0:  # Highest prediction
            color = "success"
            icon = "fas fa-arrow-up"
        elif i == len(sorted_predictions) - 1:  # Lowest prediction
            color = "info"
            icon = "fas fa-arrow-down"
        else:
            color = "primary"
            icon = "fas fa-equals"
        
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className=f"{icon} me-2"),
                        html.Span(model_name, className="fw-bold"),
                    ], className="mb-2"),
                    html.H4(f"${prediction:,.0f}", className=f"text-{color} fw-bold")
                ])
            ], className="mb-2 shadow-sm border-0")
        )
    
    # Calculate statistics
    pred_values = list(predictions.values())
    avg_pred = np.mean(pred_values)
    std_pred = np.std(pred_values)
    
    stats_card = dbc.Card([
        dbc.CardHeader([
            html.H6("Prediction Statistics", className="mb-0 fw-bold")
        ]),
        dbc.CardBody([
            html.P(f"Average: ${avg_pred:,.0f}", className="mb-1"),
            html.P(f"Std Dev: ${std_pred:,.0f}", className="mb-1"),
            html.P(f"Range: ${max(pred_values) - min(pred_values):,.0f}", className="mb-0")
        ])
    ], className="mt-3 shadow-sm border-0")
    
    return html.Div([
        html.Div(cards),
        stats_card
    ])

def get_model_description(model_name):
    """Get model description"""
    descriptions = {
        'Linear Regression': "Simple linear relationship between features and salary. Fast and interpretable.",
        'Ridge Regression': "Linear regression with L2 regularization to prevent overfitting.",
        'Lasso Regression': "Linear regression with L1 regularization for feature selection.",
        'Random Forest': "Ensemble of decision trees with voting mechanism. Robust and handles non-linear relationships.",
        'Gradient Boosting': "Sequential ensemble method that builds models to correct previous errors.",
        'Support Vector Regressor': "Uses support vectors to find optimal hyperplane for regression."
    }
    
    return html.P(descriptions.get(model_name, "Advanced machine learning algorithm."), 
                  className="text-muted")