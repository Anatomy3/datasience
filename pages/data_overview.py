"""
Data Overview Page
Dataset statistics, preview, and data quality information
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def layout(df, colors):
    """Create data overview page layout"""
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-table me-3", style={'color': colors['primary']}),
                            "Data Overview"
                        ], className="display-5 fw-bold mb-3"),
                        html.P("Statistik deskriptif, preview dataset, dan informasi kualitas data",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Dataset Information Cards
            dbc.Row([
                dbc.Col([
                    create_info_card(
                        "📊", "Total Records", f"{len(df):,}", 
                        f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}", colors['primary']
                    )
                ], md=3, className="mb-4"),
                dbc.Col([
                    create_info_card(
                        "📅", "Period", f"{df['work_year'].min()}-{df['work_year'].max()}", 
                        f"Years covered in dataset", colors['secondary']
                    )
                ], md=3, className="mb-4"),
                dbc.Col([
                    create_info_card(
                        "💾", "Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB", 
                        "Dataset size in memory", colors['accent']
                    )
                ], md=3, className="mb-4"),
                dbc.Col([
                    create_info_card(
                        "✅", "Completeness", f"{((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100):.1f}%", 
                        "Data completeness ratio", colors['dark']
                    )
                ], md=3, className="mb-4")
            ]),
            
            # Data Preview Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-eye me-2"),
                                "Dataset Preview"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.P(f"Showing first 10 rows of {len(df):,} total records", 
                                  className="text-muted mb-3"),
                            create_data_table(df)
                        ])
                    ], className="shadow-sm border-0 mb-4")
                ])
            ]),
            
            # Data Types and Statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-info-circle me-2"),
                                "Column Information"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_column_info_table(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Descriptive Statistics"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_stats_table(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            # Missing Values Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-exclamation-triangle me-2"),
                                "Data Quality Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_missing_values_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-list me-2"),
                                "Unique Values Count"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_unique_values_chart(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            # Category Distribution
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Categorical Variables Distribution"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_categorical_charts(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_info_card(icon, title, value, description, color):
    """Create information card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Span(icon, className="fs-1 me-3"),
                    html.Div([
                        html.H3(value, className="fw-bold mb-1", style={'color': color}),
                        html.H6(title, className="mb-1"),
                        html.Small(description, className="text-muted")
                    ])
                ], className="d-flex align-items-center")
            ])
        ])
    ], className="h-100 border-0 shadow-sm",
       style={'borderLeft': f'4px solid {color}'})

def create_data_table(df):
    """Create data preview table"""
    return dash_table.DataTable(
        data=df.head(10).to_dict('records'),
        columns=[{"name": col, "id": col} for col in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#10b981',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data={
            'backgroundColor': '#f8fffe',
            'border': '1px solid #e5e7eb'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#ffffff'
            }
        ],
        page_size=10,
        sort_action="native"
    )

def create_column_info_table(df, colors):
    """Create column information table"""
    column_info = []
    for col in df.columns:
        column_info.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null': f"{df[col].count():,}",
            'Missing': f"{df[col].isnull().sum():,}",
            'Unique': f"{df[col].nunique():,}"
        })
    
    return dash_table.DataTable(
        data=column_info,
        columns=[
            {"name": "Column Name", "id": "Column"},
            {"name": "Data Type", "id": "Type"},
            {"name": "Non-Null Count", "id": "Non-Null"},
            {"name": "Missing Values", "id": "Missing"},
            {"name": "Unique Values", "id": "Unique"}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '13px'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data={
            'backgroundColor': '#f8fffe'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#ffffff'
            }
        ]
    )

def create_stats_table(df, colors):
    """Create descriptive statistics table"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    stats_data = []
    
    for col in numeric_cols:
        stats_data.append({
            'Metric': col,
            'Mean': f"{df[col].mean():,.0f}",
            'Median': f"{df[col].median():,.0f}",
            'Min': f"{df[col].min():,.0f}",
            'Max': f"{df[col].max():,.0f}",
            'Std': f"{df[col].std():,.0f}"
        })
    
    return dash_table.DataTable(
        data=stats_data,
        columns=[
            {"name": "Column", "id": "Metric"},
            {"name": "Mean", "id": "Mean"},
            {"name": "Median", "id": "Median"},
            {"name": "Min", "id": "Min"},
            {"name": "Max", "id": "Max"},
            {"name": "Std Dev", "id": "Std"}
        ],
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '13px'
        },
        style_header={
            'backgroundColor': colors['secondary'],
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data={
            'backgroundColor': '#f8fffe'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#ffffff'
            }
        ]
    )

def create_missing_values_chart(df, colors):
    """Create missing values visualization"""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) == 0:
        # No missing values
        fig = go.Figure()
        fig.add_annotation(
            text="🎉 No Missing Values Found!<br>Dataset is complete",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16, color=colors['primary']),
            showarrow=False
        )
    else:
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            color=missing_data.values,
            color_continuous_scale=[[0, colors['light']], [1, colors['primary']]]
        )
        
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Count of Missing Values",
            yaxis_title="Columns"
        )
    
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_unique_values_chart(df, colors):
    """Create unique values count chart"""
    unique_counts = df.nunique().sort_values(ascending=True)
    
    fig = px.bar(
        x=unique_counts.values,
        y=unique_counts.index,
        orientation='h',
        color=unique_counts.values,
        color_continuous_scale=[[0, colors['light']], [1, colors['accent']]]
    )
    
    fig.update_layout(
        title="Unique Values Count by Column",
        xaxis_title="Number of Unique Values",
        yaxis_title="Columns",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_categorical_charts(df, colors):
    """Create categorical distribution charts"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    charts = []
    for i, col in enumerate(categorical_cols[:4]):  # Show first 4 categorical columns
        value_counts = df[col].value_counts().head(5)
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"{col.replace('_', ' ').title()} Distribution",
            color_discrete_sequence=[colors['primary'], colors['secondary'], 
                                   colors['accent'], colors['dark'], colors['light']]
        )
        
        fig.update_layout(
            height=250,
            margin=dict(t=40, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            font=dict(size=10)
        )
        
        charts.append(
            dbc.Col([
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ], md=6, className="mb-3")
        )
    
    return dbc.Row(charts)