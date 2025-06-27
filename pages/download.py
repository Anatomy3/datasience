"""
Download & Export Page
Data export, reports generation, and file downloads
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io
import os

def layout(df, colors):
    """Create download and export page layout"""
    
    return html.Div([
        # Download Components (PENTING!)
        dcc.Download(id="download-dataframe-csv"),
        dcc.Download(id="download-cleaned-csv"), 
        dcc.Download(id="download-analysis-json"),
        dcc.Download(id="download-predictions-csv"),
        dcc.Download(id="download-custom-export"),
        dcc.Download(id="download-report-pdf"),
        
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-download me-3", style={'color': colors['primary']}),
                            "Download & Export"
                        ], className="display-5 fw-bold mb-3"),
                        html.P("Export data, generate reports, dan download hasil analisis dalam berbagai format",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Quick Download Section
            dbc.Row([
                dbc.Col([
                    create_quick_downloads(df, colors)
                ])
            ], className="mb-4"),
            
            # Custom Export Options
            dbc.Row([
                dbc.Col([
                    create_custom_export_section(df, colors)
                ], md=8, className="mb-4"),
                dbc.Col([
                    create_export_history(colors)
                ], md=4, className="mb-4")
            ]),
            
            # Report Generation
            dbc.Row([
                dbc.Col([
                    create_report_generation(df, colors)
                ])
            ], className="mb-4"),
            
            # Data Processing Options
            dbc.Row([
                dbc.Col([
                    create_data_processing_options(df, colors)
                ], md=6, className="mb-4"),
                dbc.Col([
                    create_model_export_section(colors)
                ], md=6, className="mb-4")
            ])
            
        ], fluid=True, className="py-4")
    ])

def create_quick_downloads(df, colors):
    """Create quick download section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-bolt me-2"),
                "Quick Downloads"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P("Download dataset dan hasil analisis dengan sekali klik", 
                  className="text-muted mb-4"),
            
            dbc.Row([
                dbc.Col([
                    create_download_card(
                        "📊", "Original Dataset", 
                        f"Complete dataset ({len(df):,} records)",
                        "dataset-original", "CSV Format", colors['primary']
                    )
                ], md=3, className="mb-3"),
                dbc.Col([
                    create_download_card(
                        "🧹", "Cleaned Dataset",
                        "Processed and cleaned data",
                        "dataset-cleaned", "CSV Format", colors['secondary']
                    )
                ], md=3, className="mb-3"),
                dbc.Col([
                    create_download_card(
                        "📈", "Analysis Summary",
                        "Key statistics and insights",
                        "analysis-summary", "JSON Format", colors['accent']
                    )
                ], md=3, className="mb-3"),
                dbc.Col([
                    create_download_card(
                        "🤖", "Model Predictions",
                        "ML model predictions",
                        "model-predictions", "CSV Format", colors['dark']
                    )
                ], md=3, className="mb-3")
            ])
        ])
    ], className="shadow-sm border-0")

def create_download_card(icon, title, description, download_id, format_info, color):
    """Create download option card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, className="fs-2 mb-2 text-center"),
                html.H6(title, className="fw-bold mb-2 text-center"),
                html.P(description, className="small text-muted text-center mb-3"),
                html.P(format_info, className="small text-center mb-3", 
                      style={'color': color}),
                dbc.Button([
                    html.I(className="fas fa-download me-2"),
                    "Download"
                ], id=download_id, color="primary", size="sm", 
                   className="w-100")
            ])
        ])
    ], className="h-100 border-0 shadow-sm hover-card",
       style={'borderTop': f'3px solid {color}'})

def create_custom_export_section(df, colors):
    """Create custom export options"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-sliders-h me-2"),
                "Custom Export Options"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Columns:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='columns-selector',
                        options=[{'label': col, 'value': col} for col in df.columns],
                        value=list(df.columns),
                        multi=True,
                        placeholder="Select columns to export"
                    )
                ], md=6, className="mb-3"),
                dbc.Col([
                    html.Label("Export Format:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='format-selector',
                        options=[
                            {'label': 'CSV', 'value': 'csv'},
                            {'label': 'Excel', 'value': 'excel'},
                            {'label': 'JSON', 'value': 'json'},
                            {'label': 'Parquet', 'value': 'parquet'}
                        ],
                        value='csv',
                        clearable=False
                    )
                ], md=6, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Filter by Experience Level:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='experience-filter-export',
                        options=[{'label': 'All Levels', 'value': 'all'}] +
                               [{'label': f'{exp} Level', 'value': exp} 
                                for exp in sorted(df['experience_level'].unique())],
                        value='all',
                        clearable=False
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    html.Label("Salary Range (USD):", className="fw-bold mb-2"),
                    dcc.RangeSlider(
                        id='salary-range-export',
                        min=df['salary_in_usd'].min(),
                        max=df['salary_in_usd'].max(),
                        value=[df['salary_in_usd'].min(), df['salary_in_usd'].max()],
                        marks={
                            df['salary_in_usd'].min(): f'${df["salary_in_usd"].min()/1000:.0f}K',
                            df['salary_in_usd'].max(): f'${df["salary_in_usd"].max()/1000:.0f}K'
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    html.Label("Include Metadata:", className="fw-bold mb-2"),
                    dbc.Checklist(
                        id='metadata-options',
                        options=[
                            {'label': 'Export Info', 'value': 'export_info'},
                            {'label': 'Statistics', 'value': 'statistics'},
                            {'label': 'Data Quality', 'value': 'quality'}
                        ],
                        value=['export_info']
                    )
                ], md=4, className="mb-3")
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id='export-preview', className="mb-3")
                ], md=8),
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-cog me-2"),
                        "Generate Custom Export"
                    ], id='custom-export-btn', color="success", size="lg", 
                       className="w-100")
                ], md=4)
            ])
        ])
    ], className="shadow-sm border-0")

def create_export_history(colors):
    """Create export history section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-history me-2"),
                "Export History"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P("Recent exports and downloads", className="text-muted mb-3"),
            
            create_history_item("dataset_export_2023.csv", "5 minutes ago", "2.1 MB"),
            create_history_item("analysis_report.pdf", "1 hour ago", "856 KB"),
            create_history_item("predictions.json", "2 hours ago", "1.3 MB"),
            create_history_item("clean_dataset.xlsx", "1 day ago", "3.2 MB"),
            
            html.Hr(),
            
            html.Div([
                html.Small("💡 Tip: Files are available for 7 days", 
                          className="text-muted")
            ])
        ])
    ], className="shadow-sm border-0 h-100")

def create_history_item(filename, time, size):
    """Create export history item"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-file-alt me-2 text-muted"),
            html.Strong(filename, className="small"),
            html.Br(),
            html.Small(f"{time} • {size}", className="text-muted")
        ], className="p-2 border rounded mb-2")
    ])

def create_report_generation(df, colors):
    """Create report generation section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-file-pdf me-2"),
                "Report Generation"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("📋 Available Report Templates", className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            create_report_template(
                                "Executive Summary", "High-level insights for leadership",
                                ["Key metrics", "Trends", "Recommendations"],
                                "executive-report", colors['primary']
                            )
                        ], md=4, className="mb-3"),
                        dbc.Col([
                            create_report_template(
                                "Technical Analysis", "Detailed statistical analysis",
                                ["Data quality", "Statistical tests", "Model performance"],
                                "technical-report", colors['secondary']
                            )
                        ], md=4, className="mb-3"),
                        dbc.Col([
                            create_report_template(
                                "Market Research", "Salary benchmarking report",
                                ["Market trends", "Geographic analysis", "Career insights"],
                                "market-report", colors['accent']
                            )
                        ], md=4, className="mb-3")
                    ])
                ], md=8),
                dbc.Col([
                    html.H6("⚙️ Report Settings", className="mb-3"),
                    
                    html.Label("Include Sections:", className="fw-bold mb-2"),
                    dbc.Checklist(
                        id='report-sections',
                        options=[
                            {'label': 'Data Overview', 'value': 'overview'},
                            {'label': 'Visualizations', 'value': 'charts'},
                            {'label': 'Statistical Analysis', 'value': 'stats'},
                            {'label': 'Insights', 'value': 'insights'},
                            {'label': 'Recommendations', 'value': 'recommendations'}
                        ],
                        value=['overview', 'charts', 'insights'],
                        className="mb-3"
                    ),
                    
                    html.Label("Output Format:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='report-format',
                        options=[
                            {'label': 'PDF Report', 'value': 'pdf'},
                            {'label': 'HTML Page', 'value': 'html'},
                            {'label': 'Word Document', 'value': 'docx'}
                        ],
                        value='html',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    dbc.Button([
                        html.I(className="fas fa-magic me-2"),
                        "Generate Report"
                    ], id='generate-report-btn', color="warning", 
                       className="w-100")
                ], md=4)
            ])
        ])
    ], className="shadow-sm border-0")

def create_report_template(title, description, features, report_id, color):
    """Create report template card"""
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="fw-bold mb-2"),
            html.P(description, className="small text-muted mb-2"),
            html.Ul([html.Li(feature, className="small") for feature in features], 
                   className="mb-3"),
            dbc.Button("Select", id=report_id, size="sm", 
                      color="outline-primary", className="w-100")
        ])
    ], className="h-100 border-0 shadow-sm",
       style={'borderLeft': f'3px solid {color}'})

def create_data_processing_options(df, colors):
    """Create data processing options"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-cogs me-2"),
                "Data Processing Options"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.H6("🔧 Processing Features", className="mb-3"),
            
            dbc.Checklist(
                id='processing-options',
                options=[
                    {'label': 'Remove Outliers', 'value': 'remove_outliers'},
                    {'label': 'Normalize Salaries', 'value': 'normalize'},
                    {'label': 'Add Calculated Fields', 'value': 'calculated_fields'},
                    {'label': 'Anonymize Data', 'value': 'anonymize'},
                    {'label': 'Aggregate by Groups', 'value': 'aggregate'}
                ],
                value=[],
                className="mb-3"
            ),
            
            html.Hr(),
            
            html.H6("📊 Data Aggregation", className="mb-2"),
            dcc.Dropdown(
                id='aggregation-level',
                options=[
                    {'label': 'No Aggregation', 'value': 'none'},
                    {'label': 'By Experience Level', 'value': 'experience'},
                    {'label': 'By Country', 'value': 'country'},
                    {'label': 'By Company Size', 'value': 'company_size'},
                    {'label': 'By Job Title', 'value': 'job_title'}
                ],
                value='none',
                className="mb-3"
            ),
            
            dbc.Button([
                html.I(className="fas fa-play me-2"),
                "Process & Export"
            ], id='process-export-btn', color="info", className="w-100")
        ])
    ], className="shadow-sm border-0")

def create_model_export_section(colors):
    """Create model export section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-robot me-2"),
                "Model & Predictions Export"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.H6("🤖 Machine Learning Assets", className="mb-3"),
            
            html.Div([
                create_model_export_item(
                    "Trained Model", "model.pkl", "Machine learning model file",
                    "model-export", colors['primary']
                ),
                create_model_export_item(
                    "Model Metrics", "metrics.json", "Performance evaluation results",
                    "metrics-export", colors['secondary']
                ),
                create_model_export_item(
                    "Feature Importance", "features.csv", "Feature importance rankings",
                    "features-export", colors['accent']
                ),
                create_model_export_item(
                    "Predictions", "predictions.csv", "Model predictions on test data",
                    "predictions-export", colors['dark']
                )
            ]),
            
            html.Hr(),
            
            html.H6("⚙️ Export Options", className="mb-2"),
            dcc.Dropdown(
                id='model-export-format',
                options=[
                    {'label': 'Pickle Format (.pkl)', 'value': 'pickle'},
                    {'label': 'ONNX Format (.onnx)', 'value': 'onnx'},
                    {'label': 'JSON Format (.json)', 'value': 'json'}
                ],
                value='pickle',
                className="mb-3"
            ),
            
            dbc.Button([
                html.I(className="fas fa-download me-2"),
                "Export All ML Assets"
            ], id='ml-export-btn', color="success", className="w-100")
        ])
    ], className="shadow-sm border-0")

def create_model_export_item(name, filename, description, export_id, color):
    """Create model export item"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-file me-2", style={'color': color}),
            html.Strong(name, className="me-2"),
            html.Code(filename, className="small bg-light px-2 py-1 rounded"),
            html.Br(),
            html.Small(description, className="text-muted")
        ], className="p-2 border rounded mb-2")
    ])

# Callback for export preview
@callback(
    Output('export-preview', 'children'),
    [Input('columns-selector', 'value'),
     Input('experience-filter-export', 'value'),
     Input('salary-range-export', 'value')]
)
def update_export_preview(selected_columns, exp_filter, salary_range):
    if not selected_columns:
        return dbc.Alert("Please select at least one column", color="warning")
    
    from utils.data_loader import data_loader
    df = data_loader.load_dataset()
    
    # Apply filters
    filtered_df = df.copy()
    if exp_filter != 'all':
        filtered_df = filtered_df[filtered_df['experience_level'] == exp_filter]
    
    filtered_df = filtered_df[
        (filtered_df['salary_in_usd'] >= salary_range[0]) & 
        (filtered_df['salary_in_usd'] <= salary_range[1])
    ]
    
    # Select columns
    preview_df = filtered_df[selected_columns].head(5)
    
    return html.Div([
        html.H6(f"📋 Export Preview ({len(filtered_df):,} records)", className="mb-2"),
        html.Div([
            html.Pre(preview_df.to_string(), 
                    className="small bg-light p-3 rounded",
                    style={'fontSize': '11px', 'maxHeight': '200px', 'overflow': 'auto'})
        ])
    ])