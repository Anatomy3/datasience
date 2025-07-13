"""
Data Cleaning Page
Data preprocessing and cleaning analysis
"""

import dash_bootstrap_components as dbc
from dash import html, dash_table, dcc
import pandas as pd
import plotly.express as px

def layout(df, colors):
    """Create data cleaning page layout"""
    
    return dbc.Container([
        
        # Header Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1([
                        html.I(className="fas fa-broom me-3", style={'color': colors['primary']}),
                        "Data Cleaning & Preprocessing"
                    ], className="text-center mb-3", style={'color': colors['darker']}),
                    html.P([
                        "Analisis kualitas data, deteksi outlier, dan proses pembersihan data ",
                        "untuk memastikan dataset siap untuk modeling dan analisis."
                    ], className="text-center text-muted mb-4 lead")
                ], className="mb-5")
            ])
        ]),
        
        # Data Quality Overview
        dbc.Row([
            dbc.Col([
                create_quality_stat_card(
                    "🎯", "Clean Dataset",
                    "Data Quality", colors['primary']
                )
            ], md=3, className="mb-4"),
            dbc.Col([
                create_quality_stat_card(
                    "🔍", f"{check_missing_values(df)}",
                    "Missing Values", colors['secondary']
                )
            ], md=3, className="mb-4"),
            dbc.Col([
                create_quality_stat_card(
                    "📊", f"{check_duplicates(df)}",
                    "Duplicate Rows", colors['accent']
                )
            ], md=3, className="mb-4"),
            dbc.Col([
                create_quality_stat_card(
                    "⚡", f"{check_outliers(df)} potential",
                    "Outliers", colors['dark']
                )
            ], md=3, className="mb-4")
        ]),
        
        # Missing Values Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-search me-2"),
                            "Missing Values Analysis"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        create_missing_values_analysis(df)
                    ])
                ], className="shadow-sm border-0")
            ], md=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-copy me-2"),
                            "Duplicate Rows Check"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        create_duplicate_analysis(df)
                    ])
                ], className="shadow-sm border-0")
            ], md=6, className="mb-4")
        ]),
        
        # Data Types and Distribution
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-list me-2"),
                            "Data Types Information"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        create_data_types_info(df)
                    ])
                ], className="shadow-sm border-0")
            ], md=12, className="mb-4")
        ]),
        
        # Outlier Detection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            "Outlier Detection (Salary Data)"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=create_outlier_boxplot(df, colors),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className="shadow-sm border-0")
            ], md=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-area me-2"),
                            "Salary Distribution Analysis"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=create_distribution_plot(df, colors),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className="shadow-sm border-0")
            ], md=6, className="mb-4")
        ]),
        
        # Data Cleaning Recommendations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-lightbulb me-2"),
                            "Data Cleaning Recommendations"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        create_cleaning_recommendations(df, colors)
                    ])
                ], className="shadow-sm border-0")
            ], md=8, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-check-circle me-2"),
                            "Cleaning Status"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        create_cleaning_status(df, colors)
                    ])
                ], className="shadow-sm border-0")
            ], md=4, className="mb-4")
        ]),
        
        # Sample of Clean Data
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-table me-2"),
                            "Sample of Cleaned Data"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        create_sample_table(df)
                    ])
                ], className="shadow-sm border-0")
            ])
        ], className="mb-4")
        
    ], fluid=True, className="py-4")

def create_quality_stat_card(icon, value, label, color):
    """Create data quality statistics card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, className="fs-2 mb-2"),
                html.H4(value, className="fw-bold mb-1", style={'color': color}),
                html.P(label, className="mb-0 text-muted fw-medium")
            ], className="text-center")
        ])
    ], className="h-100 border-0 shadow-sm",
       style={
           'background': f'linear-gradient(145deg, #ffffff, #f8fffe)',
           'borderLeft': f'4px solid {color}'
       })

def check_missing_values(df):
    """Check for missing values"""
    missing_count = df.isnull().sum().sum()
    return "0 Found" if missing_count == 0 else f"{missing_count} Found"

def check_duplicates(df):
    """Check for duplicate rows"""
    duplicate_count = df.duplicated().sum()
    return "0 Found" if duplicate_count == 0 else f"{duplicate_count} Found"

def check_outliers(df):
    """Simple outlier detection using IQR method"""
    if 'salary_in_usd' in df.columns:
        Q1 = df['salary_in_usd'].quantile(0.25)
        Q3 = df['salary_in_usd'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df['salary_in_usd'] < lower_bound) | (df['salary_in_usd'] > upper_bound)]
        return len(outliers)
    return 0

def create_missing_values_analysis(df):
    """Create missing values analysis"""
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    if missing_data.sum() == 0:
        return html.Div([
            html.Div([
                html.I(className="fas fa-check-circle fa-4x text-success mb-3"),
                html.H5("No Missing Values", className="text-success"),
                html.P("Dataset is complete! All columns have complete data with no missing values.")
            ], className="text-center py-4")
        ])
    else:
        missing_info = []
        for col in missing_data[missing_data > 0].index:
            missing_info.append(
                html.Div([
                    html.Strong(f"{col}: "),
                    html.Span(f"{missing_data[col]} ({missing_percentage[col]:.1f}%)", 
                            className="text-warning")
                ], className="mb-2")
            )
        return html.Div(missing_info)

def create_duplicate_analysis(df):
    """Create duplicate analysis"""
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count == 0:
        return html.Div([
            html.Div([
                html.I(className="fas fa-check-circle fa-4x text-success mb-3"),
                html.H5("No Duplicates", className="text-success"),
                html.P("Dataset has no duplicate rows. Each record is unique.")
            ], className="text-center py-4")
        ])
    else:
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-3x text-warning mb-3"),
                html.H5(f"{duplicate_count} Duplicate Rows Found", className="text-warning"),
                html.P("Consider removing duplicate rows before analysis.")
            ], className="text-center py-4")
        ])

def create_data_types_info(df):
    """Create data types information table"""
    data_types_info = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        # Determine if column needs attention
        status = "✅ Good"
        if null_count > 0:
            status = "⚠️ Has Nulls"
        elif dtype == 'object' and unique_count > len(df) * 0.9:
            status = "🔍 High Cardinality"
        
        data_types_info.append(
            html.Tr([
                html.Td(html.Strong(col)),
                html.Td(dbc.Badge(dtype, color="primary" if "int" in dtype or "float" in dtype else "secondary")),
                html.Td(f"{unique_count:,}"),
                html.Td(f"{null_count}"),
                html.Td(status)
            ])
        )
    
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Column Name"),
                html.Th("Data Type"),
                html.Th("Unique Values"),
                html.Th("Missing Values"),
                html.Th("Status")
            ])
        ]),
        html.Tbody(data_types_info)
    ], striped=True, hover=True, responsive=True)

def create_outlier_boxplot(df, colors):
    """Create boxplot for outlier detection"""
    fig = px.box(
        df, 
        y='salary_in_usd',
        title='Salary Distribution - Outlier Detection',
        color_discrete_sequence=[colors['primary']]
    )
    
    fig.update_layout(
        yaxis_title="Salary (USD)",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_distribution_plot(df, colors):
    """Create distribution plot"""
    fig = px.histogram(
        df, 
        x='salary_in_usd',
        nbins=30,
        title='Salary Distribution',
        color_discrete_sequence=[colors['primary']]
    )
    
    # Add mean line
    mean_salary = df['salary_in_usd'].mean()
    fig.add_vline(
        x=mean_salary,
        line_dash="dash",
        line_color=colors['dark'],
        annotation_text=f"Mean: ${mean_salary:,.0f}"
    )
    
    fig.update_layout(
        xaxis_title="Salary (USD)",
        yaxis_title="Frequency",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_cleaning_recommendations(df, colors):
    """Create data cleaning recommendations"""
    recommendations = []
    
    # Check missing values
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        recommendations.append({
            'icon': '✅',
            'title': 'No Missing Values',
            'description': 'Dataset is complete with no missing data.',
            'priority': 'success'
        })
    else:
        recommendations.append({
            'icon': '⚠️',
            'title': 'Handle Missing Values',
            'description': f'{missing_count} missing values found. Consider imputation or removal.',
            'priority': 'warning'
        })
    
    # Check duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        recommendations.append({
            'icon': '✅',
            'title': 'No Duplicate Rows',
            'description': 'All records are unique.',
            'priority': 'success'
        })
    else:
        recommendations.append({
            'icon': '🔄',
            'title': 'Remove Duplicates',
            'description': f'{duplicate_count} duplicate rows should be removed.',
            'priority': 'warning'
        })
    
    # Check outliers
    outlier_count = check_outliers(df)
    if outlier_count > 0:
        recommendations.append({
            'icon': '🎯',
            'title': 'Review Outliers',
            'description': f'{outlier_count} potential outliers detected. Investigate before modeling.',
            'priority': 'info'
        })
    else:
        recommendations.append({
            'icon': '✅',
            'title': 'No Significant Outliers',
            'description': 'Data distribution looks normal.',
            'priority': 'success'
        })
    
    # Data types
    recommendations.append({
        'icon': '📊',
        'title': 'Data Types Look Good',
        'description': 'All columns have appropriate data types for analysis.',
        'priority': 'success'
    })
    
    recommendation_cards = []
    for rec in recommendations:
        color_map = {
            'success': 'success',
            'warning': 'warning', 
            'info': 'info',
            'danger': 'danger'
        }
        
        recommendation_cards.append(
            dbc.Alert([
                html.H6([
                    html.Span(rec['icon'], className="me-2"),
                    rec['title']
                ], className="mb-2"),
                html.P(rec['description'], className="mb-0")
            ], color=color_map[rec['priority']], className="mb-3")
        )
    
    return html.Div(recommendation_cards)

def create_cleaning_status(df, colors):
    """Create overall cleaning status"""
    total_issues = 0
    
    # Count issues
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    outlier_count = check_outliers(df)
    
    total_issues = missing_count + duplicate_count
    
    if total_issues == 0:
        status_color = "success"
        status_icon = "fas fa-check-circle"
        status_text = "Clean"
        status_message = "Dataset is ready for analysis!"
    elif total_issues <= 5:
        status_color = "warning"
        status_icon = "fas fa-exclamation-triangle"
        status_text = "Minor Issues"
        status_message = "A few issues need attention."
    else:
        status_color = "danger"
        status_icon = "fas fa-times-circle"
        status_text = "Needs Cleaning"
        status_message = "Several issues require fixing."
    
    return html.Div([
        html.Div([
            html.I(className=f"{status_icon} fa-3x text-{status_color} mb-3"),
            html.H4(status_text, className=f"text-{status_color} mb-2"),
            html.P(status_message, className="mb-3"),
            
            # Issue summary
            html.Hr(),
            html.Small([
                html.Strong("Summary:"), html.Br(),
                f"Missing Values: {missing_count}", html.Br(),
                f"Duplicates: {duplicate_count}", html.Br(),
                f"Potential Outliers: {outlier_count}"
            ], className="text-muted")
        ], className="text-center")
    ])

def create_sample_table(df):
    """Create sample of cleaned data"""
    sample_df = df.head(5)
    
    return dash_table.DataTable(
        data=sample_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in sample_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'fontSize': '12px',
            'fontFamily': 'Inter, Arial, sans-serif',
            'padding': '10px'
        },
        style_header={
            'backgroundColor': '#10b981',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f0fdf4'
            }
        ]
    )