"""
Data Overview Page
Comprehensive data statistics, preview, and analysis
"""

import dash_bootstrap_components as dbc
from dash import html, dash_table, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def layout(df, colors, lang='id'):
    """Create clean and organized data overview page layout"""
    
    return dbc.Container([
        
        # Clean Header Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1([
                        html.I(className="fas fa-database me-3", style={'color': colors['primary']}),
                        "Data Overview"
                    ], className="mb-3", style={'color': colors['darker'], 'fontWeight': '700'}),
                    html.P([
                        "Comprehensive overview of the Data Science Salaries dataset with basic statistics, ",
                        "data preview, and initial analysis to understand data structure and quality."
                    ], className="text-muted mb-0 lead", style={'fontSize': '1.1rem'})
                ], className="text-center py-4")
            ])
        ], className="mb-5"),
        
        # Clean Statistics Cards Row
        dbc.Row([
            dbc.Col([
                create_stat_card(
                    "📊", f"{len(df):,}",
                    "Total Records", colors['primary']
                )
            ], xl=3, lg=6, md=6, sm=12, className="mb-3"),
            dbc.Col([
                create_stat_card(
                    "🔢", str(df.shape[1]),
                    "Total Columns", colors['secondary']
                )
            ], xl=3, lg=6, md=6, sm=12, className="mb-3"),
            dbc.Col([
                create_stat_card(
                    "💰", f"${df['salary_in_usd'].mean():,.0f}",
                    "Average Salary", colors['accent']
                )
            ], xl=3, lg=6, md=6, sm=12, className="mb-3"),
            dbc.Col([
                create_stat_card(
                    "🌍", str(df['company_location'].nunique()),
                    "Countries", colors['dark']
                )
            ], xl=3, lg=6, md=6, sm=12, className="mb-3")
        ], className="mb-5"),
        
        # Dataset Information Section - Better Spacing
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-info-circle me-2", style={'color': colors['primary']}),
                            "Dataset Information"
                        ], className="mb-0", style={'fontWeight': '600'})
                    ], style={'background': 'linear-gradient(135deg, #f8f9fa, #e9ecef)', 'border': 'none'}),
                    dbc.CardBody([
                        create_dataset_info(df)
                    ], className="p-4")
                ], className="shadow-sm border-0 h-100")
            ], lg=6, md=12, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-shield-check me-2", style={'color': colors['success']}),
                            "Data Quality Check"
                        ], className="mb-0", style={'fontWeight': '600'})
                    ], style={'background': 'linear-gradient(135deg, #f8f9fa, #e9ecef)', 'border': 'none'}),
                    dbc.CardBody([
                        create_data_quality_info(df)
                    ], className="p-4")
                ], className="shadow-sm border-0 h-100")
            ], lg=6, md=12, className="mb-4")
        ], className="mb-5"),
        
        # Column Information - Full Width Clean Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-columns me-2", style={'color': colors['info']}),
                            "Column Information"
                        ], className="mb-0", style={'fontWeight': '600'})
                    ], style={'background': 'linear-gradient(135deg, #f8f9fa, #e9ecef)', 'border': 'none'}),
                    dbc.CardBody([
                        create_column_info(df)
                    ], className="p-4")
                ], className="shadow-sm border-0")
            ])
        ], className="mb-5"),
        
        # Quick Visualizations - Better Layout
        dbc.Row([
            dbc.Col([
                html.H4("Data Visualizations", className="mb-4", style={'color': colors['darker'], 'fontWeight': '600'})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="fas fa-chart-bar me-2", style={'color': colors['primary']}),
                            "Salary Distribution"
                        ], className="mb-0", style={'fontWeight': '600'})
                    ], style={'background': 'linear-gradient(135deg, #f8f9fa, #e9ecef)', 'border': 'none'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=create_salary_distribution(df, colors),
                            config={'displayModeBar': False}
                        )
                    ], className="p-3")
                ], className="shadow-sm border-0 h-100")
            ], lg=6, md=12, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="fas fa-chart-pie me-2", style={'color': colors['secondary']}),
                            "Experience Level Distribution"
                        ], className="mb-0", style={'fontWeight': '600'})
                    ], style={'background': 'linear-gradient(135deg, #f8f9fa, #e9ecef)', 'border': 'none'}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=create_experience_distribution(df, colors),
                            config={'displayModeBar': False}
                        )
                    ], className="p-3")
                ], className="shadow-sm border-0 h-100")
            ], lg=6, md=12, className="mb-4")
        ], className="mb-5"),
        
        # Data Preview Section
        dbc.Row([
            dbc.Col([
                html.H4("Data Preview", className="mb-4", style={'color': colors['darker'], 'fontWeight': '600'})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="fas fa-table me-2", style={'color': colors['warning']}),
                            "First 10 Rows"
                        ], className="mb-0", style={'fontWeight': '600'})
                    ], style={'background': 'linear-gradient(135deg, #f8f9fa, #e9ecef)', 'border': 'none'}),
                    dbc.CardBody([
                        create_data_preview_table(df)
                    ], className="p-3")
                ], className="shadow-sm border-0")
            ])
        ], className="mb-5"),
        
        # Summary Statistics Section
        dbc.Row([
            dbc.Col([
                html.H4("Statistical Summary", className="mb-4", style={'color': colors['darker'], 'fontWeight': '600'})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="fas fa-calculator me-2", style={'color': colors['accent']}),
                            "Numerical Columns Statistics"
                        ], className="mb-0", style={'fontWeight': '600'})
                    ], style={'background': 'linear-gradient(135deg, #f8f9fa, #e9ecef)', 'border': 'none'}),
                    dbc.CardBody([
                        create_summary_statistics(df)
                    ], className="p-3")
                ], className="shadow-sm border-0")
            ])
        ], className="mb-5"),
        
        # Clean Next Steps Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("What's Next?", className="text-center mb-4", style={'color': colors['darker'], 'fontWeight': '600'}),
                    html.P("Continue your data science journey with these next steps", className="text-center text-muted mb-5")
                ])
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-chart-line fa-2x mb-3", style={'color': colors['primary']}),
                            html.H6("Explore Data", className="card-title fw-bold"),
                            html.P("Interactive visualizations and analysis", className="card-text small text-muted mb-3"),
                            dbc.Button([
                                "Go to EDA ",
                                html.I(className="fas fa-arrow-right ms-1")
                            ], href="/eda", color="primary", size="sm", className="w-100")
                        ], className="text-center")
                    ], className="p-3")
                ], className="h-100 shadow-sm border-0")
            ], lg=4, md=6, sm=12, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-robot fa-2x mb-3", style={'color': colors['success']}),
                            html.H6("Build Model", className="card-title fw-bold"),
                            html.P("Train ML models for predictions", className="card-text small text-muted mb-3"),
                            dbc.Button([
                                "Go to Modeling ",
                                html.I(className="fas fa-arrow-right ms-1")
                            ], href="/modeling", color="success", size="sm", className="w-100")
                        ], className="text-center")
                    ], className="p-3")
                ], className="h-100 shadow-sm border-0")
            ], lg=4, md=6, sm=12, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-magic fa-2x mb-3", style={'color': colors['warning']}),
                            html.H6("Predict Salary", className="card-title fw-bold"),
                            html.P("Use models to predict salaries", className="card-text small text-muted mb-3"),
                            dbc.Button([
                                "Go to Prediction ",
                                html.I(className="fas fa-arrow-right ms-1")
                            ], href="/prediction", color="warning", size="sm", className="w-100")
                        ], className="text-center")
                    ], className="p-3")
                ], className="h-100 shadow-sm border-0")
            ], lg=4, md=12, sm=12, className="mb-3")
        ], className="mb-4")
        
    ], fluid=True, className="py-4")

def create_stat_card(icon, value, label, color):
    """Create clean statistics card with modern design"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Span(icon, className="fs-1 mb-3", style={'color': color}),
                    html.H4(value, className="fw-bold mb-2", style={'color': color, 'fontSize': '1.8rem'}),
                    html.P(label, className="mb-0 text-muted", style={'fontSize': '0.9rem', 'fontWeight': '500'})
                ], className="text-center")
            ])
        ], className="p-4")
    ], className="h-100 border-0 shadow-sm card-hover",
       style={
           'background': 'linear-gradient(135deg, #ffffff, #f8fafe)',
           'borderTop': f'3px solid {color}',
       })

def create_dataset_info(df):
    """Create comprehensive dataset information"""
    memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # Convert to MB
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("📏 Dimensions", className="fw-bold text-primary mb-2"),
                    html.P(f"{df.shape[0]:,} rows × {df.shape[1]} columns", className="mb-3"),
                    
                    html.H6("💾 Memory Usage", className="fw-bold text-primary mb-2"),
                    html.P(f"{memory_usage:.2f} MB", className="mb-3"),
                    
                    html.H6("📅 Data Period", className="fw-bold text-primary mb-2"),
                    html.P(f"{df['work_year'].min()} - {df['work_year'].max()}", className="mb-0")
                ])
            ], md=6),
            dbc.Col([
                html.Div([
                    html.H6("💼 Job Variety", className="fw-bold text-success mb-2"),
                    html.P(f"{df['job_title'].nunique()} unique job titles", className="mb-3"),
                    
                    html.H6("🏢 Company Diversity", className="fw-bold text-success mb-2"),
                    html.P(f"{df['company_size'].nunique()} company sizes", className="mb-3"),
                    
                    html.H6("🌐 Global Reach", className="fw-bold text-success mb-2"),
                    html.P(f"{df['company_location'].nunique()} countries", className="mb-0")
                ])
            ], md=6)
        ])
    ])

def create_data_quality_info(df):
    """Create data quality assessment"""
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Check for potential issues
    issues = []
    if missing_values > 0:
        issues.append(f"⚠️ {missing_values} missing values")
    if duplicate_rows > 0:
        issues.append(f"⚠️ {duplicate_rows} duplicate rows")
    
    return html.Div([
        # Quality Score
        html.Div([
            html.H5([
                html.I(className="fas fa-shield-alt me-2 text-success"),
                "Data Quality Score"
            ], className="mb-3"),
            
            # Quality indicators
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Missing Values", className="small text-muted d-block"),
                        html.H4("0", className="text-success mb-0") if missing_values == 0 else html.H4(str(missing_values), className="text-warning mb-0")
                    ])
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.Span("Duplicate Rows", className="small text-muted d-block"),
                        html.H4("0", className="text-success mb-0") if duplicate_rows == 0 else html.H4(str(duplicate_rows), className="text-warning mb-0")
                    ])
                ], md=6)
            ], className="mb-3"),
            
            # Data types distribution
            html.H6("Data Types:", className="fw-bold mb-2"),
            html.Div([
                dbc.Badge(f"{dtype}: {count}", color="light", text_color="dark", className="me-2 mb-1")
                for dtype, count in df.dtypes.value_counts().items()
            ]),
            
            # Overall assessment
            html.Hr(),
            html.Div([
                html.I(className="fas fa-check-circle me-2 text-success"),
                html.Span("Dataset is clean and ready for analysis!", className="text-success fw-bold")
            ] if missing_values == 0 and duplicate_rows == 0 else [
                html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                html.Span("Dataset needs some cleaning", className="text-warning fw-bold")
            ])
        ])
    ])

def create_column_info(df):
    """Create detailed column information"""
    column_info = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        # Column description based on name
        descriptions = {
            'work_year': 'Year when the salary was paid',
            'experience_level': 'Experience level in the job (EN/MI/SE/EX)',
            'employment_type': 'Type of employment (FT/PT/CT/FL)',
            'job_title': 'Job title of the employee',
            'salary': 'Salary in local currency',
            'salary_currency': 'Currency of the salary',
            'salary_in_usd': 'Salary converted to USD',
            'employee_residence': 'Employee residence country',
            'remote_ratio': 'Percentage of remote work',
            'company_location': 'Country where company is located',
            'company_size': 'Size of the company (S/M/L)'
        }
        
        description = descriptions.get(col, 'Data column')
        
        column_info.append(
            html.Tr([
                html.Td(html.Strong(col), style={'verticalAlign': 'middle'}),
                html.Td(
                    dbc.Badge(dtype, color="primary" if "int" in dtype or "float" in dtype else "secondary"),
                    style={'verticalAlign': 'middle'}
                ),
                html.Td(f"{unique_count:,}", style={'verticalAlign': 'middle'}),
                html.Td(
                    html.Span("0", className="text-success") if null_count == 0 else html.Span(str(null_count), className="text-warning"),
                    style={'verticalAlign': 'middle'}
                ),
                html.Td(html.Small(description, className="text-muted"), style={'verticalAlign': 'middle'})
            ])
        )
    
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Column Name"),
                html.Th("Data Type"),
                html.Th("Unique Values"),
                html.Th("Missing Values"),
                html.Th("Description")
            ])
        ]),
        html.Tbody(column_info)
    ], striped=True, hover=True, responsive=True, className="mb-0")

def create_data_preview_table(df):
    """Create data preview table with styling"""
    return dash_table.DataTable(
        data=df.head(10).to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={
            'overflowX': 'auto',
            'border': '1px solid #dee2e6'
        },
        style_cell={
            'textAlign': 'left',
            'fontSize': '13px',
            'fontFamily': 'Inter, Arial, sans-serif',
            'padding': '12px',
            'border': '1px solid #dee2e6'
        },
        style_header={
            'backgroundColor': '#10b981',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            },
            {
                'if': {'column_id': 'salary_in_usd'},
                'backgroundColor': '#a7f3d0',
                'color': '#047857',
                'fontWeight': 'bold'
            }
        ],
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in df.head(10).to_dict('records')
        ],
        tooltip_duration=None
    )

def create_summary_statistics(df):
    """Create summary statistics for numerical columns"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numerical_cols) == 0:
        return html.P("No numerical columns found for statistical summary.", className="text-muted")
    
    stats_data = []
    for col in numerical_cols:
        stats = df[col].describe()
        stats_data.append({
            'Column': col,
            'Count': f"{stats['count']:,.0f}",
            'Mean': f"{stats['mean']:,.2f}",
            'Std': f"{stats['std']:,.2f}",
            'Min': f"{stats['min']:,.2f}",
            '25%': f"{stats['25%']:,.2f}",
            '50%': f"{stats['50%']:,.2f}",
            '75%': f"{stats['75%']:,.2f}",
            'Max': f"{stats['max']:,.2f}"
        })
    
    return dash_table.DataTable(
        data=stats_data,
        columns=[{"name": i, "id": i} for i in stats_data[0].keys()],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'fontSize': '12px',
            'fontFamily': 'Inter, Arial, sans-serif',
            'padding': '10px'
        },
        style_header={
            'backgroundColor': '#34d399',
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

def create_salary_distribution(df, colors):
    """Create salary distribution histogram"""
    fig = px.histogram(
        df, 
        x='salary_in_usd',
        nbins=30,
        title='Distribution of Salaries (USD)',
        color_discrete_sequence=[colors['primary']]
    )
    
    fig.update_layout(
        xaxis_title="Salary (USD)",
        yaxis_title="Frequency",
        showlegend=False,
        margin=dict(t=50, b=40, l=40, r=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    fig.update_traces(
        hovertemplate='<b>Salary Range:</b> $%{x:,.0f}<br><b>Count:</b> %{y}<extra></extra>'
    )
    
    return fig

def create_experience_distribution(df, colors):
    """Create experience level pie chart"""
    exp_counts = df['experience_level'].value_counts()
    exp_labels = {
        'EN': 'Entry Level',
        'MI': 'Mid Level', 
        'SE': 'Senior Level',
        'EX': 'Executive Level'
    }
    
    fig = px.pie(
        values=exp_counts.values,
        names=[exp_labels.get(x, x) for x in exp_counts.index],
        title='Experience Level Distribution',
        color_discrete_sequence=[colors['primary'], colors['secondary'], colors['accent'], colors['dark']]
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        margin=dict(t=50, b=40, l=40, r=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig