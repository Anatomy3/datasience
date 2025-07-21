"""
Download & Export Page
Data export, reports generation, and file downloads
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
from datetime import datetime

def layout(df, colors, lang='id'):
    """Create download and export page layout"""
    
    return html.Div([
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
            
            # Export Options
            dbc.Row([
                dbc.Col([
                    create_export_formats_section(df, colors)
                ], md=8, className="mb-4"),
                dbc.Col([
                    create_file_info_section(df, colors)
                ], md=4, className="mb-4")
            ]),
            
            # Report Templates
            dbc.Row([
                dbc.Col([
                    create_report_templates_section(df, colors)
                ])
            ], className="mb-4"),
            
            # Data Processing & Model Export
            dbc.Row([
                dbc.Col([
                    create_data_processing_section(df, colors)
                ], md=6, className="mb-4"),
                dbc.Col([
                    create_model_export_section(colors)
                ], md=6, className="mb-4")
            ]),
            
            # Export Instructions
            dbc.Row([
                dbc.Col([
                    create_export_instructions(colors)
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_quick_downloads(df, colors):
    """Create quick download section"""
    file_size = len(df) * 0.001  # Rough estimate in MB
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-bolt me-2"),
                "Quick Downloads"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P("Download dataset dan hasil analisis dengan berbagai format", 
                  className="text-muted mb-4"),
            
            dbc.Row([
                dbc.Col([
                    create_download_card(
                        "📊", "Original Dataset", 
                        f"Complete dataset ({len(df):,} records)",
                        f"CSV • ~{file_size:.1f} MB", colors['primary'],
                        "btn-download-csv-main"  # ID yang akan dikaitkan dengan callback di app.py
                    )
                ], md=3, className="mb-3"),
                dbc.Col([
                    create_download_card(
                        "🧹", "Cleaned Dataset",
                        "Processed and ready for analysis",
                        "CSV • Pre-processed", colors['secondary'],
                        "btn-download-cleaned"
                    )
                ], md=3, className="mb-3"),
                dbc.Col([
                    create_download_card(
                        "📈", "Analysis Summary",
                        "Key statistics and insights",
                        "JSON • Summary data", colors['accent'],
                        "btn-download-summary"
                    )
                ], md=3, className="mb-3"),
                dbc.Col([
                    create_download_card(
                        "📋", "Full Report",
                        "Complete analysis report",
                        "HTML • Interactive", colors['dark'],
                        "btn-download-report"
                    )
                ], md=3, className="mb-3")
            ])
        ])
    ], className="shadow-sm border-0")

def create_download_card(icon, title, description, format_info, color, btn_id):
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
                ], id=btn_id, color="primary", size="sm", 
                   className="w-100")
            ])
        ])
    ], className="h-100 border-0 shadow-sm hover-card",
       style={'borderTop': f'3px solid {color}'})

def create_export_formats_section(df, colors):
    """Create export formats information"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-file-export me-2"),
                "Available Export Formats"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P("Choose from various formats for different use cases:", 
                  className="text-muted mb-4"),
            
            # Format options
            dbc.Row([
                dbc.Col([
                    create_format_info_card(
                        "📄", "CSV", "Comma-separated values",
                        ["Excel compatible", "Universal format", "Lightweight"],
                        colors['primary']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_format_info_card(
                        "📊", "Excel", "Microsoft Excel format",
                        ["Multiple sheets", "Formatted data", "Charts included"],
                        colors['secondary']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_format_info_card(
                        "🔗", "JSON", "JavaScript Object Notation",
                        ["API friendly", "Structured data", "Web compatible"],
                        colors['accent']
                    )
                ], md=4, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_format_info_card(
                        "⚡", "Parquet", "Columnar storage format",
                        ["High performance", "Compressed", "Analytics optimized"],
                        colors['dark']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_format_info_card(
                        "📑", "HTML", "Web page format",
                        ["Interactive tables", "Embedded charts", "Shareable"],
                        colors['primary']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_format_info_card(
                        "📋", "PDF", "Portable Document Format",
                        ["Report ready", "Print friendly", "Professional"],
                        colors['secondary']
                    )
                ], md=4, className="mb-3")
            ])
        ])
    ], className="shadow-sm border-0")

def create_format_info_card(icon, format_name, description, features, color):
    """Create format information card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, className="fs-3 mb-2 text-center"),
                html.H6(format_name, className="fw-bold mb-1 text-center"),
                html.P(description, className="small text-muted text-center mb-2"),
                html.Ul([
                    html.Li(feature, className="small") for feature in features
                ], className="mb-0")
            ])
        ])
    ], className="h-100 border-0 shadow-sm",
       style={'borderLeft': f'3px solid {color}'})

def create_file_info_section(df, colors):
    """Create file information section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-info-circle me-2"),
                "Dataset Information"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Dataset stats
            html.Div([
                create_info_item("📊", "Total Records", f"{len(df):,}"),
                create_info_item("🔢", "Columns", str(df.shape[1])),
                create_info_item("💾", "Estimated Size", f"{len(df) * 0.001:.1f} MB"),
                create_info_item("📅", "Data Period", f"{df['work_year'].min()} - {df['work_year'].max()}"),
                create_info_item("🌍", "Countries", str(df['company_location'].nunique())),
                create_info_item("💼", "Job Titles", str(df['job_title'].nunique())),
            ]),
            
            html.Hr(),
            
            html.H6("📋 Column Information", className="mb-2"),
            create_columns_list(df),
            
            html.Hr(),
            
            html.Div([
                html.H6("⏰ Last Updated", className="mb-2"),
                html.P(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                      className="small text-muted")
            ])
        ])
    ], className="shadow-sm border-0")

def create_info_item(icon, label, value):
    """Create information item"""
    return html.Div([
        html.Div([
            html.Span(icon, className="me-2"),
            html.Strong(label + ": "),
            html.Span(value)
        ], className="mb-2 small")
    ])

def create_columns_list(df):
    """Create columns list"""
    columns = list(df.columns)
    return html.Div([
        html.Div([
            dbc.Badge(col, color="light", text_color="dark", className="me-1 mb-1")
            for col in columns[:8]  # Show first 8 columns
        ]),
        html.Small(f"... and {len(columns)-8} more columns" if len(columns) > 8 else "", 
                  className="text-muted")
    ])

def create_report_templates_section(df, colors):
    """Create report templates section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-file-contract me-2"),
                "Report Templates"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P("Pre-built report templates for different audiences:", 
                  className="text-muted mb-4"),
            
            dbc.Row([
                dbc.Col([
                    create_report_template_card(
                        "👔", "Executive Summary", 
                        "High-level insights for leadership team",
                        ["Key metrics", "Market trends", "Strategic recommendations"],
                        colors['primary']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_report_template_card(
                        "🔬", "Technical Analysis", 
                        "Detailed analysis for data teams",
                        ["Statistical tests", "Data quality", "Model performance"],
                        colors['secondary']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_report_template_card(
                        "📊", "Market Research", 
                        "Salary benchmarking insights",
                        ["Geographic analysis", "Role comparisons", "Career trends"],
                        colors['accent']
                    )
                ], md=4, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_report_template_card(
                        "📈", "Trend Analysis", 
                        "Time-series and forecasting report",
                        ["Yearly trends", "Predictions", "Growth patterns"],
                        colors['dark']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_report_template_card(
                        "🎯", "Custom Report", 
                        "Build your own custom report",
                        ["Choose sections", "Custom filters", "Personalized insights"],
                        colors['primary']
                    )
                ], md=4, className="mb-3"),
                dbc.Col([
                    create_report_template_card(
                        "📱", "Dashboard Export", 
                        "Interactive dashboard snapshot",
                        ["Live charts", "Filterable data", "Mobile friendly"],
                        colors['secondary']
                    )
                ], md=4, className="mb-3")
            ])
        ])
    ], className="shadow-sm border-0")

def create_report_template_card(icon, title, description, features, color):
    """Create report template card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, className="fs-2 mb-2 text-center"),
                html.H6(title, className="fw-bold mb-2 text-center"),
                html.P(description, className="small text-muted text-center mb-3"),
                html.Ul([
                    html.Li(feature, className="small") for feature in features
                ], className="mb-3"),
                dbc.Button("Generate", color="outline-primary", size="sm", 
                          className="w-100")
            ])
        ])
    ], className="h-100 border-0 shadow-sm",
       style={'borderTop': f'3px solid {color}'})

def create_data_processing_section(df, colors):
    """Create data processing options"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-cogs me-2"),
                "Data Processing Options"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.H6("🔧 Available Processing", className="mb-3"),
            
            # Processing options
            html.Div([
                create_processing_option(
                    "🧹", "Data Cleaning",
                    "Remove duplicates, handle missing values",
                    colors['primary']
                ),
                create_processing_option(
                    "📊", "Statistical Analysis",
                    "Generate descriptive statistics",
                    colors['secondary']
                ),
                create_processing_option(
                    "🎯", "Outlier Detection",
                    "Identify and handle outliers",
                    colors['accent']
                ),
                create_processing_option(
                    "🔀", "Data Transformation",
                    "Normalize, scale, and encode data",
                    colors['dark']
                ),
                create_processing_option(
                    "📈", "Feature Engineering",
                    "Create new calculated fields",
                    colors['primary']
                ),
                create_processing_option(
                    "🔒", "Data Anonymization",
                    "Remove sensitive information",
                    colors['secondary']
                )
            ]),
            
            html.Hr(),
            
            html.Div([
                html.H6("📋 Processing Summary", className="mb-2"),
                html.Small("Choose processing options when downloading", 
                          className="text-muted")
            ])
        ])
    ], className="shadow-sm border-0")

def create_processing_option(icon, title, description, color):
    """Create processing option item"""
    return html.Div([
        html.Div([
            html.Span(icon, className="me-2 fs-5"),
            html.Div([
                html.Strong(title),
                html.Br(),
                html.Small(description, className="text-muted")
            ], className="d-inline-block")
        ], className="p-2 border rounded mb-2",
           style={'borderLeft': f'3px solid {color}'})
    ])

def create_model_export_section(colors):
    """Create model export section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-robot me-2"),
                "ML Model & Assets"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.H6("🤖 Exportable Assets", className="mb-3"),
            
            # Model assets
            html.Div([
                create_model_asset_item(
                    "🧠", "Trained Model",
                    "Random Forest model file (.pkl)",
                    "Ready for deployment",
                    colors['primary']
                ),
                create_model_asset_item(
                    "📊", "Model Metrics",
                    "Performance evaluation results (.json)",
                    "Accuracy, precision, recall scores",
                    colors['secondary']
                ),
                create_model_asset_item(
                    "🎯", "Feature Importance",
                    "Variable importance rankings (.csv)",
                    "Model interpretability data",
                    colors['accent']
                ),
                create_model_asset_item(
                    "🔮", "Predictions",
                    "Model predictions on test set (.csv)",
                    "Validation and test results",
                    colors['dark']
                ),
                create_model_asset_item(
                    "⚙️", "Model Configuration",
                    "Hyperparameters and settings (.json)",
                    "Model reproducibility info",
                    colors['primary']
                ),
                create_model_asset_item(
                    "📈", "Training History",
                    "Training logs and metrics (.json)",
                    "Model development tracking",
                    colors['secondary']
                )
            ]),
            
            html.Hr(),
            
            dbc.Button([
                html.I(className="fas fa-download me-2"),
                "Download All ML Assets"
            ], color="success", className="w-100")
        ])
    ], className="shadow-sm border-0")

def create_model_asset_item(icon, title, filename, description, color):
    """Create model asset item"""
    return html.Div([
        html.Div([
            html.Span(icon, className="me-2"),
            html.Div([
                html.Strong(title),
                html.Br(),
                html.Code(filename, className="small bg-light px-1 rounded"),
                html.Br(),
                html.Small(description, className="text-muted")
            ], className="d-inline-block")
        ], className="p-2 border rounded mb-2",
           style={'borderLeft': f'3px solid {color}'})
    ])

def create_export_instructions(colors):
    """Create export instructions section"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-question-circle me-2"),
                "Export Instructions & Tips"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("📋 How to Export", className="mb-3"),
                    html.Ol([
                        html.Li("Choose your preferred format from Quick Downloads"),
                        html.Li("Click the Download button for instant export"),
                        html.Li("For custom exports, use the processing options"),
                        html.Li("Reports can be generated with different templates"),
                        html.Li("ML assets include model files and metrics")
                    ], className="mb-3"),
                    
                    html.H6("💡 Best Practices", className="mb-2"),
                    html.Ul([
                        html.Li("Use CSV for general analysis and Excel compatibility"),
                        html.Li("Choose JSON for API integration and web applications"),
                        html.Li("PDF reports are best for presentations"),
                        html.Li("Parquet format for large-scale data processing")
                    ])
                ], md=6),
                dbc.Col([
                    html.H6("🔧 Technical Details", className="mb-3"),
                    html.Div([
                        create_tech_detail("File Encoding", "UTF-8 for maximum compatibility"),
                        create_tech_detail("Compression", "Automatic for large files"),
                        create_tech_detail("Format Validation", "All exports are validated"),
                        create_tech_detail("Download Limit", "No restrictions on file size"),
                        create_tech_detail("Retention", "Files available for 7 days"),
                        create_tech_detail("Security", "Secure download links")
                    ], className="mb-3"),
                    
                    html.Div([
                        dbc.Alert([
                            html.I(className="fas fa-info-circle me-2"),
                            html.Strong("Need Help? "),
                            "Contact support for custom export requirements."
                        ], color="info", className="mb-0")
                    ])
                ], md=6)
            ])
        ])
    ], className="shadow-sm border-0")

def create_tech_detail(label, value):
    """Create technical detail item"""
    return html.Div([
        html.Strong(label + ": "),
        html.Span(value)
    ], className="small mb-1")