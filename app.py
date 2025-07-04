""" 
Data Science Salaries Dashboard - Modern Green Theme 
Main Application File with Horizontal Navigation, Footer, and Prediction 
""" 

import dash 
from dash import dcc, html, Input, Output, State, callback_context 
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

# Load model for prediction
try:
    import joblib
    model = joblib.load('data/model.pkl')
    print("Model loaded successfully!")
except (FileNotFoundError, ImportError, ModuleNotFoundError):
    print("Model file not found or joblib not available. Prediction will show demo data.")
    model = None

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

# Helper functions for dropdown options
def get_job_title_options(df):
    """Get job title options from dataset"""
    job_titles = sorted(df['job_title'].unique())
    return [{'label': title, 'value': title} for title in job_titles[:15]]  # Top 15

def get_location_options(df):
    """Get location options from dataset"""
    locations = sorted(df['company_location'].unique())
    return [{'label': loc, 'value': loc} for loc in locations[:20]]  # Top 20

# Helper functions for labels
def get_experience_label(exp):
    mapping = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
    return mapping.get(exp, exp)

def get_employment_label(emp):
    mapping = {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}
    return mapping.get(emp, emp)

# Team Section Function
def create_team_section_styled(colors):
    """
    Membuat bagian 'Tentang Tim' dengan layout gambar di samping teks,
    sesuai dengan referensi yang diberikan.
    """
    
    # --- Profil Lingga ---
    lingga_profile = dbc.Row([
        # Kolom Teks
        dbc.Col([
            html.H3("Lingga Dwi Satria Vigio", className="fw-bold"),
            html.H6("LEAD DEVELOPER & MACHINE LEARNING", className="text-muted mb-3", style={'letterSpacing': '1px'}),
            html.P(
                "Bertanggung jawab atas arsitektur backend, pemrosesan data, "
                "pengembangan model machine learning, dan memastikan semua "
                "logika aplikasi berjalan dengan lancar dari data cleaning hingga deployment model.",
                className="text-secondary"
            ),
            html.Div([
                html.A(html.I(className="fas fa-envelope fa-lg"), href="mailto:lingga22si@mahasiswa.pcr.ac.id", className="text-dark me-3", title="Email"),
                html.A(html.I(className="fab fa-linkedin fa-lg"), href="#", target="_blank", className="text-dark me-3", title="LinkedIn"),
                html.A(html.I(className="fab fa-github fa-lg"), href="https://github.com/lingga", target="_blank", className="text-dark", title="GitHub")
            ], className="mt-4")
        ], md=7, className="d-flex flex-column justify-content-center"),

        # Kolom Gambar
        dbc.Col([
            # Ganti 'lingga.png' dengan nama file gambar Anda di folder /assets
            html.Img(src="/assets/lingga.png", className="rounded-circle img-fluid shadow-lg", style={'border': f'5px solid {colors["primary"]}'})
        ], md=5)
    ], className="align-items-center mb-5")
    
    # --- Profil Azzahara ---
    azzahara_profile = dbc.Row([
        # Kolom Gambar (di kiri untuk variasi)
        dbc.Col([
            # Ganti 'azzahara.png' dengan nama file gambar Anda di folder /assets
            html.Img(src="/assets/azzahara.png", className="rounded-circle img-fluid shadow-lg", style={'border': f'5px solid {colors["primary"]}'})
        ], md=5, className="order-md-1"),
        
        # Kolom Teks (di kanan untuk variasi)
        dbc.Col([
            html.H3("Azzahara Tunisyah", className="fw-bold"),
            html.H6("LEAD ANALYST & UI/UX DESIGNER", className="text-muted mb-3", style={'letterSpacing': '1px'}),
            html.P(
                "Memimpin analisis data eksploratif (EDA), visualisasi data, dan storytelling. "
                "Bertugas merancang user interface yang intuitif dan menarik, serta menerjemahkan "
                "data kompleks menjadi insight yang mudah dipahami.",
                className="text-secondary"
            ),
            html.Div([
                html.A(html.I(className="fas fa-envelope fa-lg"), href="mailto:azzahara22si@mahasiswa.pcr.ac.id", className="text-dark me-3", title="Email"),
                html.A(html.I(className="fab fa-linkedin fa-lg"), href="#", target="_blank", className="text-dark me-3", title="LinkedIn"),
                html.A(html.I(className="fab fa-github fa-lg"), href="https://github.com/azzahara", target="_blank", className="text-dark", title="GitHub")
            ], className="mt-4")
        ], md=7, className="d-flex flex-column justify-content-center order-md-2")
        
    ], className="align-items-center mb-5")

    return dbc.Container([
        html.Hr(className="my-5"),
        html.H2("Tim Pengembang", className="text-center mb-5 fw-bold display-5", style={'color': colors['darker']}),
        lingga_profile,
        azzahara_profile
    ], fluid=True, className="py-5 bg-light")

# Built-in Home Layout Function with Integrated Prediction Form
def create_home_layout(df, colors): 
    """Built-in home layout with integrated prediction form""" 
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

        # Top Charts Preview 
        dbc.Row([ 
            dbc.Col([ 
                dbc.Card([ 
                    dbc.CardHeader([ 
                        html.H4([ 
                            html.I(className="fas fa-chart-pie me-2"), 
                            "Salary Distribution by Experience" 
                        ], className="mb-0") 
                    ]), 
                    dbc.CardBody([ 
                        dcc.Graph( 
                            figure=create_experience_chart(df, colors), 
                            config={'displayModeBar': False} 
                        ) 
                    ]) 
                ], className="shadow-sm border-0") 
            ], md=6, className="mb-4"), 
             
            dbc.Col([ 
                dbc.Card([ 
                    dbc.CardHeader([ 
                        html.H4([ 
                            html.I(className="fas fa-globe me-2"), 
                            "Top 10 Countries by Average Salary" 
                        ], className="mb-0") 
                    ]), 
                    dbc.CardBody([ 
                        dcc.Graph( 
                            figure=create_country_chart(df, colors), 
                            config={'displayModeBar': False} 
                        ) 
                    ]) 
                ], className="shadow-sm border-0") 
            ], md=6, className="mb-4") 
        ]),
         
        # INTEGRATED PREDICTION SECTION 
        dbc.Row([ 
            dbc.Col([ 
                html.H3([ 
                    html.I(className="fas fa-crystal-ball me-3", style={'color': colors['primary']}), 
                    "Prediksi Gaji Data Science" 
                ], className="text-center mb-4", style={'color': colors['darker']}) 
            ]) 
        ], className="mt-5"), 

        dbc.Row([ 
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
                                    id='app-experience-dropdown',
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
                                    id='app-employment-dropdown',
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
                                    id='app-job-title-dropdown',
                                    options=get_job_title_options(df),
                                    value='Data Scientist',
                                    className="mb-3"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Company Size", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='app-company-size-dropdown',
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
                                    id='app-location-dropdown',
                                    options=get_location_options(df),
                                    value='US',
                                    className="mb-3"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Work Year", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='app-work-year-dropdown',
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
                                    id='app-remote-slider',
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
                                id='app-predict-button',
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
                        html.Div(id='app-prediction-results', children=[
                            html.Div([
                                html.I(className="fas fa-robot fa-4x text-muted mb-3"),
                                html.H5("Siap Memprediksi", className="text-muted"),
                                html.P("Isi parameter di sebelah kiri dan klik 'Prediksi Gaji' untuk memulai.")
                            ], className="text-center py-5")
                        ])
                    ])
                ], className="shadow-sm border-0")
            ], md=7)
        ], className="mb-5"),

        # Key Insights Section   
        dbc.Row([ 
            dbc.Col([ 
                dbc.Card([ 
                    dbc.CardHeader([ 
                        html.H4([ 
                            html.I(className="fas fa-lightbulb me-2"), 
                            "Key Insights" 
                        ], className="mb-0") 
                    ]), 
                    dbc.CardBody([ 
                        create_key_insights(df, colors) 
                    ]) 
                ], className="shadow-sm border-0") 
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
        ]),

        # Additional Navigation Row for Prediction 
        dbc.Row([ 
            dbc.Col([ 
                create_nav_card( 
                    "🔮", "Prediksi Gaji",  
                    "Prediksi gaji berdasarkan parameter Anda", 
                    "/prediction", colors['dark'] 
                ) 
            ], md=4, className="mb-3"), 
            dbc.Col([ 
                create_nav_card( 
                    "📈", "Hasil & Insights",  
                    "Lihat hasil analisis dan temuan penting", 
                    "/results", colors['secondary'] 
                ) 
            ], md=4, className="mb-3"), 
            dbc.Col([ 
                create_nav_card( 
                    "💾", "Download Data",  
                    "Unduh dataset dan model prediksi", 
                    "/download", colors['accent'] 
                ) 
            ], md=4, className="mb-3") 
        ]), 
                 
        # Download Section 
        dbc.Row([ 
            dbc.Col([ 
                html.Hr(className="my-5"), 
                create_download_section(colors) 
            ]) 
        ]),

        # =====================================================================
        #  BAGIAN TIM PENGEMBANG - DITAMBAHKAN DI SINI
        # =====================================================================
        create_team_section_styled(colors),
         
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

def create_experience_chart(df, colors): 
    """Create experience level distribution chart""" 
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
        color_discrete_sequence=[colors['primary'], colors['secondary'],  
                               colors['accent'], colors['dark']] 
    ) 
     
    fig.update_traces( 
        textposition='inside', 
        textinfo='percent+label', 
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>' 
    ) 
     
    fig.update_layout( 
        showlegend=True, 
        margin=dict(t=20, b=20, l=20, r=20), 
        height=300, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(size=12) 
    ) 
     
    return fig 

def create_country_chart(df, colors): 
    """Create top countries by salary chart""" 
    top_countries = (df.groupby('company_location')['salary_in_usd'] 
                    .mean() 
                    .sort_values(ascending=True) 
                    .tail(10)) 
     
    fig = px.bar( 
        x=top_countries.values, 
        y=top_countries.index, 
        orientation='h', 
        color=top_countries.values, 
        color_continuous_scale=[[0, colors['light']], [1, colors['primary']]] 
    ) 
     
    fig.update_traces( 
        hovertemplate='<b>%{y}</b><br>Avg Salary: $%{x:,.0f}<extra></extra>' 
    ) 
     
    fig.update_layout( 
        xaxis_title="Average Salary (USD)", 
        yaxis_title="Country", 
        showlegend=False, 
        margin=dict(t=20, b=20, l=20, r=20), 
        height=300, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        coloraxis_showscale=False 
    ) 
     
    return fig 

def create_key_insights(df, colors): 
    """Create key insights section""" 
     
    avg_salary = df['salary_in_usd'].mean() 
    max_salary = df['salary_in_usd'].max() 
    min_salary = df['salary_in_usd'].min() 
     
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean() 
    top_job = df['job_title'].value_counts().index[0] 
    top_country = df['company_location'].value_counts().index[0] 
     
    insights = [ 
        { 
            'icon': '💰', 
            'title': 'Rentang Gaji', 
            'text': f'Gaji berkisar dari ${min_salary:,.0f} hingga ${max_salary:,.0f} dengan rata-rata ${avg_salary:,.0f}' 
        }, 
        { 
            'icon': '📈', 
            'title': 'Premium Pengalaman', 
            'text': f'Level senior mendapat gaji {(exp_salary.get("SE", 0) / exp_salary.get("EN", 1)):.1f}x lebih tinggi dari level pemula' 
        }, 
        { 
            'icon': '🌟', 
            'title': 'Jabatan Teratas', 
            'text': f'{top_job} adalah posisi paling umum dengan {df[df["job_title"] == top_job].shape[0]} lowongan' 
        }, 
        { 
            'icon': '🌍', 
            'title': 'Negara Terkemuka', 
            'text': f'{top_country} memimpin dengan {df[df["company_location"] == top_country].shape[0]} perusahaan' 
        } 
    ] 
     
    insight_cards = [] 
    for insight in insights: 
        insight_cards.append( 
            dbc.Col([ 
                html.Div([ 
                    html.Div([ 
                        html.Span(insight['icon'], className="fs-3 me-3"), 
                        html.Div([ 
                            html.H6(insight['title'], className="fw-bold mb-1"), 
                            html.P(insight['text'], className="mb-0 small") 
                        ]) 
                    ], className="d-flex align-items-center") 
                ], className="p-3 rounded border-start border-3", 
                   style={'borderColor': colors['primary'] + '!important', 
                          'backgroundColor': colors['light'] + '20'}) 
            ], md=6, className="mb-3") 
        ) 
     
    return dbc.Row(insight_cards)

def create_download_section(colors): 
    """Create download section""" 
    return dbc.Card([ 
        dbc.CardHeader([ 
            html.H4([ 
                html.I(className="fas fa-download me-3", style={'color': 'white'}), 
                "Download & Export" 
            ], className="mb-0 text-center") 
        ], style={'background': colors['gradient'], 'color': 'white'}), 
        dbc.CardBody([ 
            dbc.Row([ 
                dbc.Col([ 
                    html.Div([ 
                        html.H5("📥 Quick Downloads", className="text-primary fw-bold mb-4 text-center"), 
                         
                        dbc.Row([ 
                            dbc.Col([ 
                                dbc.Card([ 
                                    dbc.CardBody([ 
                                        html.Div([ 
                                            html.I(className="fas fa-file-csv fa-2x text-success mb-3"), 
                                            html.H6("Dataset (CSV)", className="fw-bold mb-2"), 
                                            html.P("Unduh dataset lengkap", className="text-muted small mb-3"), 
                                            dbc.Button([ 
                                                html.I(className="fas fa-download me-2"), 
                                                "Unduh CSV" 
                                            ], color="success", size="sm", className="w-100", id="btn-download-csv-home")
                                        ], className="text-center") 
                                    ]) 
                                ], className="h-100 shadow-sm border-0", 
                                   style={'borderTop': f'3px solid {colors["secondary"]}'}) 
                            ], md=4, className="mb-3"), 
                             
                            dbc.Col([ 
                                dbc.Card([ 
                                    dbc.CardBody([ 
                                        html.Div([ 
                                            html.I(className="fas fa-chart-line fa-2x text-info mb-3"), 
                                            html.H6("Laporan Analisis", className="fw-bold mb-2"), 
                                            html.P("Ringkasan insight & temuan", className="text-muted small mb-3"), 
                                            dbc.Button([ 
                                                html.I(className="fas fa-file-pdf me-2"), 
                                                "Unduh PDF" 
                                            ], color="info", size="sm", className="w-100") 
                                        ], className="text-center") 
                                    ]) 
                                ], className="h-100 shadow-sm border-0", 
                                   style={'borderTop': f'3px solid {colors["accent"]}'}) 
                            ], md=4, className="mb-3"), 
                             
                            dbc.Col([ 
                                dbc.Card([ 
                                    dbc.CardBody([ 
                                        html.Div([ 
                                            html.I(className="fas fa-robot fa-2x text-warning mb-3"), 
                                            html.H6("Model ML", className="fw-bold mb-2"), 
                                            html.P("Model prediksi yang sudah dilatih", className="text-muted small mb-3"), 
                                            dbc.Button([ 
                                                html.I(className="fas fa-download me-2"), 
                                                "Unduh Model" 
                                            ], color="warning", size="sm", className="w-100") 
                                        ], className="text-center") 
                                    ]) 
                                ], className="h-100 shadow-sm border-0", 
                                   style={'borderTop': f'3px solid {colors["dark"]}'}) 
                            ], md=4, className="mb-3") 
                        ]) 
                    ]) 
                ], md=10, className="mx-auto"), 
            ]), 
             
            html.Hr(), 
             
            html.Div([ 
                html.P([ 
                    html.I(className="fas fa-info-circle me-2 text-primary"), 
                    "Semua file unduhan tersedia dalam format yang kompatibel dengan alat analisis populer." 
                ], className="text-center text-muted mb-0 small") 
            ]) 
        ]) 
    ], className="shadow-sm border-0 mb-4")

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

# Callback for integrated prediction in home page
@app.callback(
    Output('app-prediction-results', 'children'),
    Input('app-predict-button', 'n_clicks'),
    [
        State('app-work-year-dropdown', 'value'),
        State('app-experience-dropdown', 'value'),
        State('app-employment-dropdown', 'value'),
        State('app-job-title-dropdown', 'value'),
        State('app-location-dropdown', 'value'),
        State('app-company-size-dropdown', 'value'),
        State('app-remote-slider', 'value')
    ],
    prevent_initial_call=True
)
def predict_salary_integrated(n_clicks, work_year, experience, employment, job_title, location, company_size, remote_ratio):
    """Predict salary for integrated form in home page"""
    if n_clicks is None:
        return html.Div([
            html.Div([
                html.I(className="fas fa-robot fa-4x text-muted mb-3"),
                html.H5("Siap Memprediksi", className="text-muted"),
                html.P("Isi parameter di sebelah kiri dan klik 'Prediksi Gaji' untuk memulai.")
            ], className="text-center py-5")
        ])
    
    if model is None:
        # Demo prediction when model is not available
        import random
        base_salary = random.randint(60000, 180000)
        
        # Adjust based on experience
        exp_multiplier = {'EN': 0.7, 'MI': 1.0, 'SE': 1.3, 'EX': 1.8}
        predicted_salary = base_salary * exp_multiplier.get(experience, 1.0)
        
        predicted_salary = int(predicted_salary)
    else:
        try:
            # Create input DataFrame for real model prediction
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
            predicted_salary = int(model.predict(input_data)[0])
        except Exception as e:
            return dbc.Alert([
                html.H5("Prediksi Error", className="alert-heading"),
                html.P(f"Error: {str(e)}"),
                html.P("Menggunakan prediksi demo.")
            ], color="warning")
    
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
            html.I(className="fas fa-shield-alt me-2 text-success"),
            html.Small("Prediksi berdasarkan Random Forest model dengan akurasi tinggi" if model else "Demo prediksi - latih model untuk hasil akurat",
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