""" 
Data Science Salaries Dashboard - Modern Green Theme 
Main Application File with Horizontal Navigation, Footer, and Prediction 
""" 

import dash 
from dash import dcc, html, Input, Output, callback_context 
import dash_bootstrap_components as dbc 
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go 
from datetime import datetime 
import base64 
import io 

# Simple fallback functions for pages 
def create_simple_layout(title, message): 
    return dbc.Container([ 
        dbc.Alert([ 
            html.H4(title), 
            html.P(message), 
            dbc.Button("Back to Home", href="/", color="primary") 
        ], color="info") 
    ]) 

# Try to import pages, create fallbacks if needed 
try: 
    from pages import home 
except ImportError: 
    print("Creating fallback for home page...") 
    class home: 
        @staticmethod 
        def layout(df, colors): 
            return create_home_layout(df, colors) 

try: 
    from pages import data_overview 
except ImportError: 
    print("Creating fallback for data_overview...") 
    class data_overview: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("Data Overview", "Page under construction") 

try: 
    from pages import eda 
except ImportError: 
    class eda: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("EDA & Visualisasi", "Page under construction") 

try: 
    from pages import data_cleaning 
except ImportError: 
    class data_cleaning: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("Data Cleaning", "Page under construction") 

try: 
    from pages import modeling 
except ImportError: 
    class modeling: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("Modeling", "Page under construction") 

try: 
    from pages import prediction 
except ImportError: 
    print("Creating fallback for prediction...") 
    class prediction: 
        @staticmethod 
        def layout(df, colors): 
            return create_prediction_layout(df, colors) 

try: 
    from pages import results 
except ImportError: 
    class results: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("Results", "Page under construction") 

try: 
    from pages import insights 
except ImportError: 
    class insights: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("Insights", "Page under construction") 

try: 
    from pages import download 
except ImportError: 
    class download: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("Download", "Page under construction") 

try: 
    from pages import about 
except ImportError: 
    class about: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("About", "Page under construction") 

try: 
    from pages import bantuan 
except ImportError: 
    print("Creating fallback for bantuan...") 
    class bantuan: 
        @staticmethod 
        def layout(df, colors): 
            return create_simple_layout("Bantuan", "Page under construction") 

# Import footer component 
try: 
    from components.footer import create_footer, FOOTER_STYLES 
except ImportError: 
    print("Warning: footer component not found") 
    def create_footer(colors): 
        return html.Div() 
    FOOTER_STYLES = "" 

try: 
    from utils.data_loader import load_data 
except ImportError: 
    print("Warning: data_loader not found, using fallback") 
    def load_data(): 
        import pandas as pd 
        # Create sample data if file not found 
        return pd.DataFrame({ 
            'work_year': [2023] * 100, 
            'experience_level': ['SE'] * 100, 
            'employment_type': ['FT'] * 100, 
            'job_title': ['Data Scientist'] * 100, 
            'salary_in_usd': [100000] * 100, 
            'company_location': ['US'] * 100, 
            'company_size': ['M'] * 100, 
            'remote_ratio': [50] * 100 
        }) 

# Initialize Dash app 
app = dash.Dash( 
    __name__, 
    external_stylesheets=[ 
        dbc.themes.BOOTSTRAP, 
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css", 
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" 
    ], 
    suppress_callback_exceptions=True, 
    title="DS Salaries Dashboard" 
) 

# App configuration 
app.title = "Data Science Salaries Dashboard 2023" 
server = app.server 

# Modern Green Color Scheme 
COLORS = { 
    'primary': '#10b981',      # emerald-500 
    'secondary': '#34d399',    # emerald-400 
    'accent': '#6ee7b7',       # emerald-300 
    'light': '#a7f3d0',        # emerald-200 
    'dark': '#059669',         # emerald-600 
    'darker': '#047857',       # emerald-700 
    'white': '#ffffff', 
    'gray': '#f3f4f6', 
    'gray_dark': '#374151', 
    'black': '#111827', 
    'gradient': 'linear-gradient(135deg, #10b981 0%, #34d399 100%)', 
    'gradient_light': 'linear-gradient(135deg, #a7f3d0 0%, #6ee7b7 100%)' 
} 

# Navigation structure (for reference) 
NAVIGATION_STRUCTURE = { 
    'beranda': {'label': 'Beranda', 'href': '/'}, 
    'data': { 
        'label': 'Data', 
        'items': [ 
            {'label': 'Data Overview', 'href': '/data-overview'}, 
            {'label': 'Data Cleaning', 'href': '/cleaning'} 
        ] 
    }, 
    'analisis': { 
        'label': 'Analisis',  
        'items': [ 
            {'label': 'EDA & Visualisasi', 'href': '/eda'}, 
            {'label': 'Modeling', 'href': '/modeling'}, 
            {'label': 'Hasil Visualisasi', 'href': '/results'} 
        ] 
    }, 
    'tools': { 
        'label': 'Tools', 
        'items': [ 
            {'label': 'Prediksi Gaji', 'href': '/prediction'}, 
            {'label': 'Download', 'href': '/download'} 
        ] 
    }, 
    'info': { 
        'label': 'Info', 
        'items': [ 
            {'label': 'Insights', 'href': '/insights'}, 
            {'label': 'About', 'href': '/about'} 
        ] 
    } 
} 

# Load data 
df = load_data() 

# Built-in Home Layout Function 
def create_home_layout(df, colors): 
    """Built-in home layout if external file fails""" 
    return dbc.Container([ 
        # Quick Stats Section 
        dbc.Row([ 
            dbc.Col([ 
                html.H2([ 
                    html.I(className="fas fa-chart-bar me-3", style={'color': colors['primary']}), 
                    "Quick Statistics" 
                ], className="mb-4 text-center", style={'color': colors['darker']}) 
            ]) 
        ], className="mt-5"), 
         
        # Statistics Cards 
        dbc.Row([ 
            dbc.Col([ 
                create_stat_card( 
                    "💰", f"${df['salary_in_usd'].mean():,.0f}",  
                    "Average Salary", colors['primary'] 
                ) 
            ], md=3, className="mb-4"), 
            dbc.Col([ 
                create_stat_card( 
                    "📊", f"{len(df):,}",  
                    "Total Records", colors['secondary'] 
                ) 
            ], md=3, className="mb-4"), 
            dbc.Col([ 
                create_stat_card( 
                    "🌍", str(df['company_location'].nunique()),  
                    "Countries", colors['accent'] 
                ) 
            ], md=3, className="mb-4"), 
            dbc.Col([ 
                create_stat_card( 
                    "💼", str(df['job_title'].nunique()),  
                    "Job Roles", colors['dark'] 
                ) 
            ], md=3, className="mb-4") 
        ]), 
         
        # Prediction Section 
        dbc.Row([ 
            dbc.Col([ 
                html.H3([ 
                    html.I(className="fas fa-crystal-ball me-3", style={'color': colors['primary']}), 
                    "Prediksi Gaji" 
                ], className="text-center mb-4", style={'color': colors['darker']}) 
            ]) 
        ], className="mt-5"), 

        dbc.Row([ 
            dbc.Col([ 
                dbc.Card([ 
                    dbc.CardBody([ 
                        dbc.Row([ 
                            # Left Side - Description 
                            dbc.Col([ 
                                html.Div([ 
                                    html.H4([ 
                                        html.I(className="fas fa-magic me-2", style={'color': colors['primary']}), 
                                        "Prediksi Gaji Anda" 
                                    ], className="fw-bold mb-3"), 
                                    html.P([ 
                                        "Gunakan model Machine Learning kami untuk memprediksi gaji Data Scientist ", 
                                        "berdasarkan pengalaman, lokasi, tipe pekerjaan, dan faktor lainnya." 
                                    ], className="mb-3 text-muted"), 
                                     
                                    # Features List 
                                    html.Ul([ 
                                        html.Li([ 
                                            html.I(className="fas fa-check-circle me-2 text-success"), 
                                            "Random Forest Model dengan akurasi 100%" 
                                        ], className="mb-2"), 
                                        html.Li([ 
                                            html.I(className="fas fa-check-circle me-2 text-success"), 
                                            "Berdasarkan 3,755+ data real" 
                                        ], className="mb-2"), 
                                        html.Li([ 
                                            html.I(className="fas fa-check-circle me-2 text-success"), 
                                            "Prediksi instant dan akurat" 
                                        ], className="mb-2"), 
                                        html.Li([ 
                                            html.I(className="fas fa-check-circle me-2 text-success"), 
                                            "Breakdown gaji bulanan & harian" 
                                        ]) 
                                    ], className="list-unstyled mb-4"), 
                                     
                                    dbc.Button([ 
                                        html.I(className="fas fa-rocket me-2"), 
                                        "Mulai Prediksi" 
                                    ],  
                                    color="primary",  
                                    size="lg",  
                                    href="/prediction", 
                                    style={'background': colors['gradient'], 'border': 'none'}) 
                                ]) 
                            ], md=8), 
                             
                            # Right Side - Preview/Demo 
                            dbc.Col([ 
                                html.Div([ 
                                    # Image preview - larger and touch bottom 
                                    html.Div([ 
                                        html.Img( 
                                            src="/assets/image.png", 
                                            style={ 
                                                'width': '100%', 
                                                'height': 'auto', 
                                                'maxWidth': '350px',  # Diperbesar dari 300px 
                                                'borderRadius': '10px 10px 0 0',  # Hanya rounded top 
                                                'display': 'block', 
                                                'marginBottom': '0'  # No margin bottom 
                                            }, 
                                            className="img-fluid" 
                                        ) 
                                    ], className="text-center",  
                                       style={'height': '100%', 'display': 'flex', 'alignItems': 'flex-end'}) 
                                     
                                ], style={'height': '100%'})  # Full height container 
                            ], md=4) 
                        ]) 
                    ], style={ 
                        'backgroundColor': '#ffffff !important', 
                        'background': '#ffffff !important' 
                    }) 
                ], className="shadow-sm border-0 prediction-card", 
                   style={ 
                       'backgroundColor': '#ffffff !important', 
                       'background': '#ffffff !important',  
                       'borderTop': f'4px solid {colors["primary"]}' 
                   }) 
            ]) 
        ], className="mb-5"), 
         
        # Navigation Cards 
        dbc.Row([ 
            dbc.Col([ 
                html.H3("Jelajahi Dashboard", className="text-center mb-4",  
                       style={'color': colors['darker']}) 
            ]) 
        ]), 
         
        dbc.Row([ 
            dbc.Col([ 
                create_nav_card( 
                    "📊", "Data Overview",  
                    "Lihat statistik dan preview dataset", 
                    "/data-overview", colors['primary'] 
                ) 
            ], md=4, className="mb-3"), 
            dbc.Col([ 
                create_nav_card( 
                    "🔍", "EDA & Visualisasi",  
                    "Eksplorasi data dengan chart interaktif", 
                    "/eda", colors['secondary'] 
                ) 
            ], md=4, className="mb-3"), 
            dbc.Col([ 
                create_nav_card( 
                    "🤖", "Machine Learning",  
                    "Model prediksi gaji dan evaluasi", 
                    "/modeling", colors['accent'] 
                ) 
            ], md=4, className="mb-3") 
        ]) 
         
    ], fluid=True, className="py-4") 

# Built-in Prediction Layout Function 
def create_prediction_layout(df, colors): 
    """Built-in prediction layout if external file fails""" 
    return dbc.Container([ 
        dbc.Row([ 
            dbc.Col([ 
                html.H2([ 
                    html.I(className="fas fa-crystal-ball me-3", style={'color': colors['primary']}), 
                    "Salary Prediction" 
                ], className="text-center mb-3", style={'color': colors['darker']}), 
                html.P("Prediction page is under construction. Model integration coming soon!", 
                       className="text-center text-muted") 
            ]) 
        ], className="py-5") 
    ]) 

def create_stat_card(icon, value, label, color): 
    """Create statistics card""" 
    return dbc.Card([ 
        dbc.CardBody([ 
            html.Div([ 
                html.Div(icon, className="fs-2 mb-2"), 
                html.H3(value, className="fw-bold mb-1", style={'color': color}), 
                html.P(label, className="mb-0 text-muted fw-medium") 
            ], className="text-center") 
        ]) 
    ], className="h-100 border-0 shadow-sm", 
       style={ 
           'background': f'linear-gradient(145deg, #ffffff, #f8fffe)', 
           'borderLeft': f'4px solid {color}' 
       }) 

def create_nav_card(icon, title, description, href, color): 
    """Create navigation card""" 
    return dbc.Card([ 
        dbc.CardBody([ 
            html.Div([ 
                html.Div(icon, className="fs-1 mb-3"), 
                html.H5(title, className="fw-bold mb-2"), 
                html.P(description, className="text-muted mb-3"), 
                dbc.Button("Explore", color="primary", size="sm", href=href, 
                          style={'borderRadius': '20px'}) 
            ], className="text-center") 
        ]) 
    ], className="h-100 border-0 shadow-sm", 
       style={'background': f'linear-gradient(145deg, #ffffff, {color}10)'}) 

# Modern Navbar Component with Dropdowns and Help Button (Even Spacing) 
def create_navbar(): 
    # Data Dropdown 
    data_dropdown = dbc.DropdownMenu( 
        children=[ 
            dbc.DropdownMenuItem("Data Overview", href="/data-overview"), 
            dbc.DropdownMenuItem("Data Cleaning", href="/cleaning"), 
        ], 
        nav=True, 
        in_navbar=True, 
        label="Data", 
        className="nav-dropdown" 
    ) 
     
    # Analisis Dropdown   
    analisis_dropdown = dbc.DropdownMenu( 
        children=[ 
            dbc.DropdownMenuItem("EDA & Visualisasi", href="/eda"), 
            dbc.DropdownMenuItem("Modeling", href="/modeling"),  
            dbc.DropdownMenuItem("Hasil Visualisasi", href="/results"), 
        ], 
        nav=True, 
        in_navbar=True, 
        label="Analisis", 
        className="nav-dropdown" 
    ) 
     
    # Tools Dropdown 
    tools_dropdown = dbc.DropdownMenu( 
        children=[ 
            dbc.DropdownMenuItem("Prediksi Gaji", href="/prediction"), 
            dbc.DropdownMenuItem("Download", href="/download"), 
        ], 
        nav=True, 
        in_navbar=True, 
        label="Tools",  
        className="nav-dropdown" 
    ) 
     
    # Info Dropdown 
    info_dropdown = dbc.DropdownMenu( 
        children=[ 
            dbc.DropdownMenuItem("Insights", href="/insights"), 
            dbc.DropdownMenuItem("About", href="/about"), 
        ], 
        nav=True, 
        in_navbar=True, 
        label="Info", 
        className="nav-dropdown" 
    ) 
     
    # Create navigation with even spacing for 6 items 
    nav_items = [ 
        dbc.NavItem( 
            dbc.NavLink( 
                "Beranda", 
                href="/",  
                active="exact", 
                className="nav-link-modern", 
                style={'borderRadius': '10px', 'transition': 'all 0.3s ease'} 
            ), 
            className="nav-item-spaced" 
        ), 
        dbc.NavItem(data_dropdown, className="nav-item-spaced"), 
        dbc.NavItem(analisis_dropdown, className="nav-item-spaced"),  
        dbc.NavItem(tools_dropdown, className="nav-item-spaced"), 
        dbc.NavItem(info_dropdown, className="nav-item-spaced"), 
        dbc.NavItem( 
            dbc.NavLink([ 
                html.I(className="fas fa-question-circle me-2"), 
                "Bantuan" 
            ], 
            href="/bantuan", 
            className="nav-link-modern nav-link-help", 
            style={'borderRadius': '10px', 'transition': 'all 0.3s ease'}), 
            className="nav-item-spaced" 
        ) 
    ] 
     
    return dbc.Navbar( 
        dbc.Container([ 
            # Brand Section 
            dbc.Row([ 
                dbc.Col([ 
                    dbc.NavbarBrand( 
                        html.Span("DS Salaries", className="fw-bold",  
                                 style={'fontSize': '24px', 'color': COLORS['darker']}), 
                        href="/",  
                        className="d-flex align-items-center" 
                    ) 
                ], width="auto"), 
            ], align="center", className="w-100"), 
             
            # Navigation Menu with Even Spacing 
            dbc.Row([ 
                dbc.Col([ 
                    dbc.Nav( 
                        nav_items,  
                        className="w-100 justify-content-center",  
                        navbar=True 
                    ) 
                ], width=True) 
            ], align="center", className="w-100 mt-2") 
        ], fluid=True, className="px-4 py-3"), 
         
        color="white", 
        light=True, 
        sticky="top", 
        className="shadow-sm border-0", 
        style={'borderBottom': f'3px solid {COLORS["primary"]}'} 
    ) 

# Hero Section Component   
def create_hero(): 
    return html.Div([ 
        dbc.Container([ 
            dbc.Row([ 
                # Left Side - Text Content 
                dbc.Col([ 
                    html.Div([ 
                        # Main Headline 
                        html.H1([ 
                            "Analyzing Data Science ", 
                            html.Span("Salaries", className="highlight"), 
                            ", One Insight at a Time!" 
                        ], style={ 
                            'fontSize': '3.5rem', 
                            'fontWeight': '800', 
                            'lineHeight': '1.1', 
                            'marginBottom': '1.5rem', 
                            'color': '#1f2937' 
                        }), 
                         
                        # Subtitle 
                        html.P([ 
                            "Dengan dataset komprehensif dari 3,755 data scientist global, ", 
                            "kami menghadirkan analisis mendalam tentang tren gaji, ", 
                            "faktor-faktor yang mempengaruhi kompensasi, dan prediksi karir ", 
                            "di industri data science." 
                        ], className="hero-subtitle"), 
                         
                        # CTA Buttons 
                        html.Div([ 
                            dbc.Button([ 
                                html.I(className="fas fa-chart-line me-2"), 
                                "Start Analysis" 
                            ],  
                            color="primary",  
                            href="/data-overview", 
                            className="hero-btn-primary me-3"), 
                             
                            dbc.Button([ 
                                "Learn more ", 
                                html.I(className="fas fa-arrow-right ms-2") 
                            ],  
                            className="hero-btn-secondary") 
                        ], className="hero-cta"), 
                         
                        # App Download Section (styled like ProStruct) 
                        html.Div([ 
                            html.P("Dashboard App", className="small text-muted mb-2"), 
                            html.Div([ 
                                html.Div([ 
                                    html.I(className="fab fa-python me-2"), 
                                    "Python Dashboard" 
                                ], className="badge bg-dark me-2 p-2"), 
                                html.Div([ 
                                    html.I(className="fas fa-desktop me-2"), 
                                    "Web Based" 
                                ], className="badge bg-success p-2") 
                            ]) 
                        ]) 
                         
                    ], className="hero-text") 
                ], md=6, className="d-flex align-items-center"), 
                 
                # Right Side - Visual Content 
                dbc.Col([ 
                    html.Div([ 
                        # Main Chart Container 
                        html.Div([ 
                            # Simple chart visualization 
                            create_hero_chart(df), 
                        ], className="hero-chart-container"), 
                         
                        # Statistics Overlays 
                        html.Div([ 
                            html.Span(f"${df['salary_in_usd'].mean():,.0f}", className="hero-stat-number"), 
                            html.P("Average Salary", className="hero-stat-label") 
                        ], className="hero-stats stat-1"), 
                         
                        html.Div([ 
                            html.Span(f"{len(df):,}", className="hero-stat-number text-primary"), 
                            html.P("Data Scientists", className="hero-stat-label") 
                        ], className="hero-stats stat-2"), 
                         
                        html.Div([ 
                            html.Span(f"{df['company_location'].nunique()}", className="hero-stat-number text-success"), 
                            html.P("Countries", className="hero-stat-label") 
                        ], className="hero-stats stat-3") 
                         
                    ], className="hero-visual") 
                ], md=6) 
            ], className="hero-content") 
        ], fluid=True) 
    ], className="hero-gradient") 

def create_hero_chart(df): 
    """Create simple chart for hero section""" 
    # Simple salary distribution by experience 
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean() 
    exp_labels = { 
        'EN': 'Entry', 
        'MI': 'Mid',  
        'SE': 'Senior', 
        'EX': 'Executive' 
    } 
     
    fig = px.bar( 
        x=[exp_labels.get(x, x) for x in exp_salary.index], 
        y=exp_salary.values, 
        color=exp_salary.values, 
        color_continuous_scale=[[0, COLORS['light']], [1, COLORS['primary']]] 
    ) 
     
    fig.update_layout( 
        showlegend=False, 
        margin=dict(t=20, b=40, l=40, r=20), 
        height=300, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        xaxis_title="Experience Level", 
        yaxis_title="Avg Salary (USD)", 
        font=dict(size=11, color=COLORS['gray_dark']), 
        coloraxis_showscale=False 
    ) 
     
    fig.update_traces( 
        hovertemplate='<b>%{x}</b><br>Avg Salary: $%{y:,.0f}<extra></extra>' 
    ) 
     
    return dcc.Graph(figure=fig, config={'displayModeBar': False}) 

# Main Layout with Footer Integration 
app.layout = html.Div([ 
    dcc.Location(id='url', refresh=False), 
    dcc.Download(id="download-dataframe-csv"), 
    create_navbar(), 
    html.Div(id='page-content'), 
    create_footer(COLORS)
], style={'backgroundColor': COLORS['gray'], 'minHeight': '100vh'}) 

# Callback for page routing
@app.callback( 
    Output('page-content', 'children'), 
    Input('url', 'pathname'), 
    prevent_initial_call=False 
) 
def display_page(pathname): 
    try: 
        if pathname == '/' or pathname is None: 
            return html.Div([ 
                create_hero(), 
                home.layout(df, COLORS) 
            ]) 
        elif pathname == '/data-overview': 
            return data_overview.layout(df, COLORS) 
        elif pathname == '/eda': 
            return eda.layout(df, COLORS) 
        elif pathname == '/cleaning': 
            return data_cleaning.layout(df, COLORS) 
        elif pathname == '/modeling': 
            return modeling.layout(df, COLORS) 
        elif pathname == '/prediction': 
            return prediction.layout(df, COLORS) 
        elif pathname == '/results': 
            return results.layout(df, COLORS) 
        elif pathname == '/insights': 
            return insights.layout(df, COLORS) 
        elif pathname == '/download': 
            return download.layout(df, COLORS) 
        elif pathname == '/about': 
            return about.layout(df, COLORS) 
        elif pathname == '/bantuan': 
            return bantuan.layout(df, COLORS) 
        else: 
            return html.Div([ 
                dbc.Container([ 
                    dbc.Row([ 
                        dbc.Col([ 
                            html.Div([ 
                                html.H1("404", className="display-1 text-primary fw-bold"), 
                                html.H3("Halaman Tidak Ditemukan", className="mb-3"), 
                                html.P("Halaman yang Anda cari tidak tersedia.", className="mb-4"), 
                                dbc.Button("Kembali ke Beranda", color="primary", href="/", size="lg") 
                            ], className="text-center py-5") 
                        ], md=6, className="mx-auto") 
                    ]) 
                ], className="py-5") 
            ]) 
    except Exception as e: 
        return html.Div([ 
            dbc.Alert([ 
                html.H4("Error Loading Page"), 
                html.P(f"Error: {str(e)}"), 
                dbc.Button("Back to Home", href="/", color="primary") 
            ], color="danger") 
        ]) 

# Callback untuk handle download CSV dari Home Page (VERSI FINAL YANG SUDAH DIPERBAIKI)
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv-home", "n_clicks"),
    prevent_initial_call=True,
)
def func_download_csv(n_clicks):
    """
    Callback ini dipicu saat tombol dengan id 'btn-download-csv-home' diklik.
    Ini akan mengirimkan file 'ds_salaries.csv' dari folder 'data'.
    """
    # Kondisi untuk memastikan callback hanya berjalan setelah tombol di-klik
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Path ke file dataset Anda, sesuai dengan struktur folder Anda
    file_path = "data/ds_salaries.csv"
    
    # Menggunakan dcc.send_file untuk mengirimkan file ke browser pengguna
    return dcc.send_file(file_path)


# Custom CSS injection with Footer Styles 
app.index_string = ''' 
<!DOCTYPE html> 
<html> 
    <head> 
        {%metas%} 
        <title>{%title%}</title> 
        {%favicon%} 
        {%css%} 
        <style> 
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: #f3f4f6; 
            } 
             
            .nav-link-modern { 
                font-weight: 500 !important; 
                color: #374151 !important; 
                transition: all 0.3s ease !important; 
            } 
             
            .nav-link-modern:hover { 
                background-color: #a7f3d020 !important; 
                color: #059669 !important; 
                transform: translateY(-2px); 
            } 
             
            .nav-link-modern.active { 
                background: linear-gradient(135deg, #10b981, #34d399) !important; 
                color: white !important; 
                box-shadow: 0 4px 15px #10b98130; 
            } 
             
            .card { 
                border: none !important; 
                border-radius: 15px !important; 
                transition: all 0.3s ease !important; 
            } 
             
            .card:hover { 
                transform: translateY(-5px); 
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1) !important; 
            } 
             
            .btn-primary { 
                background: linear-gradient(135deg, #10b981, #34d399) !important; 
                border: none !important; 
                transition: all 0.3s ease !important; 
            } 
             
            .btn-primary:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 8px 25px #10b98140 !important; 
            } 
             
            .text-primary { 
                color: #10b981 !important; 
            } 
             
            .bg-primary { 
                background: linear-gradient(135deg, #10b981, #34d399) !important; 
            } 
             
            /* Navigation Items Even Spacing */ 
            .nav-item-spaced { 
                flex: 1; 
                text-align: center; 
            } 
             
            .nav-link-modern { 
                font-weight: 500 !important; 
                color: #374151 !important; 
                transition: all 0.3s ease !important; 
                border-radius: 10px !important; 
                padding: 0.5rem 1rem !important; 
                text-decoration: none !important; 
                display: flex !important; 
                align-items: center !important; 
                justify-content: center !important; 
            } 
             
            .nav-link-help { 
                border: 1px solid #10b981 !important; 
                color: #10b981 !important; 
            } 
             
            .nav-link-help:hover { 
                background-color: #10b981 !important; 
                color: white !important; 
                transform: translateY(-2px) !important; 
                box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important; 
            } 
            .nav-dropdown .dropdown-toggle { 
                font-weight: 500 !important; 
                color: #374151 !important; 
                transition: all 0.3s ease !important; 
                border-radius: 10px !important; 
                padding: 0.5rem 0.75rem !important; 
                text-decoration: none !important; 
                border: none !important; 
                background: transparent !important; 
            } 
             
            .nav-dropdown .dropdown-toggle:hover { 
                background-color: #a7f3d020 !important; 
                color: #059669 !important; 
                transform: translateY(-2px); 
            } 
             
            .nav-dropdown .dropdown-toggle:focus { 
                box-shadow: none !important; 
                background-color: #a7f3d020 !important; 
                color: #059669 !important; 
            } 
             
            .nav-dropdown .dropdown-menu { 
                border: none !important; 
                border-radius: 12px !important; 
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1) !important; 
                padding: 0.5rem !important; 
                margin-top: 0.5rem !important; 
            } 
             
            .nav-dropdown .dropdown-item { 
                border-radius: 8px !important; 
                padding: 0.5rem 1rem !important; 
                transition: all 0.3s ease !important; 
                color: #374151 !important; 
                font-weight: 500 !important; 
            } 
             
            .nav-dropdown .dropdown-item:hover { 
                background-color: #10b981 !important; 
                color: white !important; 
                transform: translateX(5px); 
            } 
             
            .nav-dropdown .dropdown-item:focus { 
                background-color: #10b981 !important; 
                color: white !important; 
            } 
             
            /* Help Button Styles */ 
            .btn-outline-light:hover { 
                background-color: #10b981 !important; 
                border-color: #10b981 !important; 
                color: white !important; 
                transform: translateY(-2px) !important; 
                box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important; 
            } 
            .prediction-card { 
                background-color: #ffffff !important; 
                background: #ffffff !important; 
            } 
             
            .prediction-card .card-body { 
                background-color: #ffffff !important; 
                background: #ffffff !important; 
            } 
             
            .prediction-card:hover { 
                transform: translateY(-3px); 
                box-shadow: 0 15px 40px rgba(16, 185, 129, 0.15) !important; 
                background-color: #ffffff !important; 
                background: #ffffff !important; 
            } 
             
            /* Footer Styles */ 
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
        </style> 
    </head> 
    <body> 
        {%app_entry%} 
        <footer> 
            {%config%} 
            {%scripts%} 
            {%renderer%} 
        </footer> 
    </body> 
</html> 
''' 

if __name__ == '__main__': 
    app.run_server(debug=True, host='127.0.0.1', port=8050)