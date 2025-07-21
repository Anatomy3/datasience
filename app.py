""" 
Data Science Salaries Dashboard - Modern Green Theme 
Main Application File with Horizontal Navigation, Footer, and Prediction 
""" 

import dash 
from dash import dcc, html, Input, Output, State, callback_context, ALL 
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
        def layout(df, colors, lang='id'): 
            return create_home_layout(df, colors, lang) 

try: 
    from pages import data_overview 
except ImportError: 
    print("Creating fallback for data_overview...") 
    class data_overview: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("Data Overview", "Page under construction") 

try: 
    from pages import eda 
except ImportError: 
    class eda: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("EDA & Visualisasi", "Page under construction")

try:
    from pages import eda_advanced
except ImportError:
    class eda_advanced:
        @staticmethod
        def layout(df, colors, lang='id'):
            return create_simple_layout("Advanced EDA", "Page under construction") 

try: 
    from pages import data_cleaning 
except ImportError: 
    class data_cleaning: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("Data Cleaning", "Page under construction") 

try: 
    from pages import modeling 
except ImportError: 
    class modeling: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("Modeling", "Page under construction")

try:
    from pages import modeling_advanced
except ImportError:
    class modeling_advanced:
        @staticmethod
        def layout(df, colors, lang='id'):
            return create_simple_layout("Advanced Modeling", "Page under construction") 

try:
    from pages import pca_analysis
except ImportError:
    class pca_analysis:
        @staticmethod
        def layout(df, colors, lang='id'):
            return create_simple_layout("PCA Analysis", "Page under construction")

try: 
    from pages import prediction 
except ImportError: 
    print("Creating fallback for prediction...") 
    class prediction: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_prediction_layout(df, colors) 

try: 
    from pages import results 
except ImportError: 
    class results: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("Results", "Page under construction") 

try: 
    from pages import insights 
except ImportError: 
    class insights: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("Insights", "Page under construction") 

try: 
    from pages import download 
except ImportError: 
    class download: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("Download", "Page under construction") 

try: 
    from pages import about 
except ImportError: 
    class about: 
        @staticmethod 
        def layout(df, colors, lang='id'): 
            return create_simple_layout("About", "Page under construction") 

 

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

# Load model for prediction - Updated path and error handling
try:
    import joblib
    # Try different possible model file paths
    model_paths = ['data/model.pkl', 'data/trained_model.pkl']
    model = None
    
    for path in model_paths:
        try:
            model = joblib.load(path)
            print(f"Model loaded successfully from {path}!")
            break
        except FileNotFoundError:
            continue
    
    if model is None:
        print("No model file found. Prediction will show demo data.")
        
except (ImportError, ModuleNotFoundError):
    print("Joblib not available. Prediction will show demo data.")
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

# Language configurations
LANGUAGES = {
    'id': {
        'name': 'Indonesia',
        'flag': '🇮🇩',
        'code': 'ID'
    },
    'en': {
        'name': 'English', 
        'flag': '🇺🇸',
        'code': 'EN'
    }
}

# Text translations
TEXTS = {
    'navbar': {
        'beranda': {'id': 'Beranda', 'en': 'Home'},
        'data': {'id': 'Data', 'en': 'Data'},
        'analisis': {'id': 'Analisis', 'en': 'Analysis'},
        'tools': {'id': 'Tools', 'en': 'Tools'},
        'about': {'id': 'About', 'en': 'About'},
        'data_overview': {'id': 'Overview Dataset', 'en': 'Data Overview'},
        'data_cleaning': {'id': 'Pembersihan Data', 'en': 'Data Cleaning'},
        'eda': {'id': 'Analisis Eksploratif', 'en': 'Exploratory Analysis'},
        'eda_advanced': {'id': 'Analisis Statistik', 'en': 'Statistical Analysis'},
        'pca': {'id': 'Analisis PCA', 'en': 'PCA Analysis'},
        'modeling': {'id': 'Machine Learning', 'en': 'Machine Learning'},
        'modeling_advanced': {'id': 'Model Comparison', 'en': 'Advanced Modeling'},
        'results': {'id': 'Visualisasi Hasil', 'en': 'Results Visualization'},
        'prediction': {'id': 'Prediksi Gaji', 'en': 'Salary Prediction'},
        'download': {'id': 'Download Data', 'en': 'Download Data'}
    },
    'hero': {
        'title': {'id': 'Dashboard Analisis Gaji Data Science Global', 'en': 'Global Data Science Salary Analytics Dashboard'},
        'subtitle': {'id': 'Platform analytics untuk menganalisis tren gaji data science global. Berdasarkan dataset komprehensif dari 3,755+ profesional di berbagai negara dan perusahaan teknologi dunia.', 
                    'en': 'Analytics platform for analyzing global data science salary trends. Based on comprehensive dataset from 3,755+ professionals across various countries and tech companies worldwide.'},
        'btn_start': {'id': 'Mulai Analisis', 'en': 'Start Analysis'},
        'btn_predict': {'id': 'Prediksi Gaji', 'en': 'Predict Salary'}
    },
    'home': {
        'quick_stats': {'id': 'Statistik Cepat', 'en': 'Quick Statistics'},
        'avg_salary': {'id': 'Rata-rata Gaji', 'en': 'Average Salary'},
        'total_records': {'id': 'Total Records', 'en': 'Total Records'},
        'countries': {'id': 'Negara', 'en': 'Countries'},
        'job_roles': {'id': 'Posisi Kerja', 'en': 'Job Roles'},
        'cta_title': {'id': 'Siap Memulai Analisis?', 'en': 'Ready to Start Analysis?'},
        'cta_desc': {'id': 'Jelajahi insights mendalam dari data gaji data science dan buat keputusan karir yang lebih baik.', 'en': 'Explore deep insights from data science salary data and make better career decisions.'},
        'cta_start': {'id': 'Mulai Sekarang', 'en': 'Start Now'},
        'cta_download': {'id': 'Download Data', 'en': 'Download Data'}
    },
    'eda': {
        'title': {'id': 'Analisis Data Eksploratif', 'en': 'Exploratory Data Analysis'},
        'subtitle': {'id': 'Visualisasi dan analisis mendalam dari dataset gaji data science', 'en': 'Deep visualization and analysis of data science salary dataset'},
        'overview': {'id': 'Ringkasan Dataset', 'en': 'Dataset Overview'},
        'distributions': {'id': 'Distribusi Data', 'en': 'Data Distributions'},
        'correlations': {'id': 'Analisis Korelasi', 'en': 'Correlation Analysis'},
        'insights': {'id': 'Temuan Penting', 'en': 'Key Insights'},
        'export_analysis': {'id': 'Ekspor Analisis', 'en': 'Export Analysis'}
    },
    'modeling': {
        'title': {'id': 'Machine Learning & Prediksi', 'en': 'Machine Learning & Prediction'},
        'subtitle': {'id': 'Model prediksi gaji menggunakan algoritma machine learning', 'en': 'Salary prediction models using machine learning algorithms'},
        'model_training': {'id': 'Pelatihan Model', 'en': 'Model Training'},
        'model_evaluation': {'id': 'Evaluasi Model', 'en': 'Model Evaluation'},
        'feature_importance': {'id': 'Pentingnya Fitur', 'en': 'Feature Importance'},
        'predictions': {'id': 'Prediksi', 'en': 'Predictions'},
        'performance_metrics': {'id': 'Metrik Performa', 'en': 'Performance Metrics'}
    },
    'pca': {
        'title': {'id': 'Analisis PCA & Reduksi Dimensi', 'en': 'PCA Analysis & Dimensionality Reduction'},
        'subtitle': {'id': 'Principal Component Analysis untuk reduksi dimensi dan ekstraksi fitur', 'en': 'Principal Component Analysis for dimensionality reduction and feature extraction'},
        'configuration': {'id': 'Konfigurasi PCA', 'en': 'PCA Configuration'},
        'variance_analysis': {'id': 'Analisis Varians', 'en': 'Variance Analysis'},
        'components': {'id': 'Komponen Utama', 'en': 'Principal Components'},
        'clustering': {'id': 'Clustering', 'en': 'Clustering'},
        'insights_recommendations': {'id': 'Insight & Rekomendasi', 'en': 'Insights & Recommendations'}
    },
    'insights': {
        'title': {'id': 'Insight & Rekomendasi', 'en': 'Insights & Recommendations'},
        'subtitle': {'id': 'Temuan kunci, insight yang actionable, dan rekomendasi strategis', 'en': 'Key findings, actionable insights, and strategic recommendations'},
        'executive_summary': {'id': 'Ringkasan Eksekutif', 'en': 'Executive Summary'},
        'key_insights': {'id': 'Insight Utama', 'en': 'Key Insights'},
        'recommendations': {'id': 'Rekomendasi Strategis', 'en': 'Strategic Recommendations'},
        'action_plan': {'id': 'Rencana Aksi', 'en': 'Action Plan'},
        'market_analysis': {'id': 'Analisis Pasar', 'en': 'Market Analysis'}
    },
    'about': {
        'title': {'id': 'Tentang Project & Tim', 'en': 'About Project & Team'},
        'subtitle': {'id': 'Mengenal project DataSalary Analytics dan tim pengembang', 'en': 'Get to know DataSalary Analytics project and development team'},
        'team_title': {'id': 'Tim Pengembang', 'en': 'Development Team'},
        'project_info': {'id': 'Informasi Project', 'en': 'Project Information'},
        'technologies': {'id': 'Teknologi yang Digunakan', 'en': 'Technologies Used'},
        'timeline': {'id': 'Timeline Project', 'en': 'Project Timeline'}
    },
    'common': {
        'loading': {'id': 'Memuat...', 'en': 'Loading...'},
        'error': {'id': 'Terjadi kesalahan', 'en': 'An error occurred'},
        'no_data': {'id': 'Tidak ada data', 'en': 'No data available'},
        'download': {'id': 'Unduh', 'en': 'Download'},
        'export': {'id': 'Ekspor', 'en': 'Export'},
        'analyze': {'id': 'Analisis', 'en': 'Analyze'},
        'predict': {'id': 'Prediksi', 'en': 'Predict'},
        'view_details': {'id': 'Lihat Detail', 'en': 'View Details'},
        'back': {'id': 'Kembali', 'en': 'Back'},
        'next': {'id': 'Selanjutnya', 'en': 'Next'},
        'previous': {'id': 'Sebelumnya', 'en': 'Previous'}
    }
} 

# App configuration 
app.title = "Data Salary Dashboard - Analisis Gaji Data Science Global" 
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

# Helper functions for dropdown options - Updated based on real dataset
def get_job_title_options(df):
    """Get top job title options from dataset by frequency"""
    job_counts = df['job_title'].value_counts()
    top_jobs = job_counts.head(20)  # Top 20 most common jobs
    return [{'label': f"{title} ({count})", 'value': title} for title, count in top_jobs.items()]

def get_location_options(df):
    """Get top location options from dataset by frequency"""
    location_counts = df['company_location'].value_counts()
    top_locations = location_counts.head(20)  # Top 20 most common locations
    
    # Country name mapping for better UX
    country_names = {
        'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada', 'ES': 'Spain',
        'IN': 'India', 'DE': 'Germany', 'FR': 'France', 'BR': 'Brazil',
        'PT': 'Portugal', 'GR': 'Greece', 'AU': 'Australia', 'NL': 'Netherlands',
        'MX': 'Mexico', 'IE': 'Ireland', 'SG': 'Singapore', 'JP': 'Japan',
        'AT': 'Austria', 'TR': 'Turkey', 'PL': 'Poland', 'NG': 'Nigeria'
    }
    
    return [{'label': f"{country_names.get(code, code)} ({count})", 'value': code} 
            for code, count in top_locations.items()]

# Helper functions for labels
def get_experience_label(exp):
    mapping = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
    return mapping.get(exp, exp)

def get_employment_label(emp):
    mapping = {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}
    return mapping.get(emp, emp)

# Job Categories Functions
def create_job_categories_section(df, colors):
    """
    Create job categories section with simple design like government data portal
    """
    # Define job categories based on actual dataset analysis with images
    categories = [
        {"name": "Data Engineering", "image": "/assets/dataengineer.png", "color": "#f59e0b", "desc": "1,092 posisi"},
        {"name": "Data Science", "image": "/assets/datasience.png", "color": "#10b981", "desc": "880 posisi"},
        {"name": "Data Analysis", "image": "/assets/dataanalisis.jpg", "color": "#3b82f6", "desc": "761 posisi"},
        {"name": "Machine Learning", "image": "/assets/machinelearning.jpg", "color": "#8b5cf6", "desc": "533 posisi"},
        {"name": "Management", "image": "/assets/manajemn.jpg", "color": "#ef4444", "desc": "198 posisi"},
        {"name": "Data Architecture", "image": "/assets/dataarsitektur.jpg", "color": "#6366f1", "desc": "130 posisi"},
        {"name": "Research", "image": "/assets/research.jpg", "color": "#06b6d4", "desc": "119 posisi"},
        {"name": "Business Intelligence", "image": "/assets/BusinessIntelligence.jpg", "color": "#84cc16", "desc": "42 posisi"}
    ]
    
    category_items = []
    for i, category in enumerate(categories):
        category_items.append(
            dbc.Col([
                create_simple_category_item(category["name"], category["image"], category["color"], category["desc"])
            ], lg=3, md=4, sm=6, xs=6, className="mb-2")
        )
    
    return dbc.Row(category_items, className="justify-content-center")


def create_simple_category_item(name, image, color, desc=""):
    """
    Create simple category item like government portal - with image and name
    """
    return html.A([
        html.Div([
            # Image Container
            html.Div([
                html.Img(
                    src=image,
                    style={
                        'width': '100%',
                        'height': '100%',
                        'objectFit': 'cover',
                        'borderRadius': '20px'
                    }
                )
            ], style={
                'width': '200px',
                'height': '200px',
                'borderRadius': '20px',
                'margin': '0 auto 10px auto',
                'transition': 'all 0.3s ease',
                'border': f'3px solid {color}',
                'overflow': 'hidden'
            }, className="category-icon"),
            
            # Category Name
            html.H6(name, 
                   className="fw-bold text-center mb-1", 
                   style={
                       'color': '#374151',
                       'fontSize': '0.9rem',
                       'transition': 'color 0.3s ease'
                   }),
            
            # Position Count
            html.P(desc,
                  className="text-center mb-0 small text-muted",
                  style={'fontSize': '0.75rem'})
        ], style={
            'padding': '15px 10px',
            'borderRadius': '15px',
            'transition': 'all 0.3s ease',
            'cursor': 'pointer'
        }, className="simple-category-item")
    ], 
    href=f"/eda?filter={name.lower().replace(' ', '_')}", 
    style={'textDecoration': 'none'},
    className="category-link")

# Team Section Function - Moved to About page

# Built-in Home Layout Function with Integrated Prediction Form
def create_home_layout(df, colors, lang='id'): 
    """Built-in home layout with integrated prediction form""" 
    return dbc.Container([ 
        # Job Categories Explorer Section
        html.Div([
            # Section Header
            html.Div([
                html.H2("Jelajahi Kategori Pekerjaan Data Science", className="section-title"),
                html.P("Temukan berbagai jalur karir di bidang data science dengan analisis gaji dan peluang", className="section-subtitle")
            ], className="text-center mb-5"),
            
            # Job Categories Grid
            create_job_categories_section(df, colors),
            
        ], className="job-categories-section mb-5"),

        # Data Overview Section (Government Portal Style)
        create_data_overview_section(df, colors, lang),
         
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
                                        {'label': 'Entry Level (320 records)', 'value': 'EN'},
                                        {'label': 'Mid Level (805 records)', 'value': 'MI'},
                                        {'label': 'Senior Level (2,516 records)', 'value': 'SE'},
                                        {'label': 'Executive Level (114 records)', 'value': 'EX'}
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
                                        {'label': 'Full-time (3,718 records)', 'value': 'FT'},
                                        {'label': 'Part-time (17 records)', 'value': 'PT'},
                                        {'label': 'Contract (10 records)', 'value': 'CT'},
                                        {'label': 'Freelance (10 records)', 'value': 'FL'}
                                    ],
                                    value='FT',
                                    className="mb-3"
                                )
                            ], md=6)
                        ]),
                        
                        # Row 2: Job Title & Company Size
                        dbc.Row([
                            dbc.Col([
                                html.Label([
                                    "Job Title ",
                                    html.Small("(for reference only)", className="text-muted")
                                ], className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='app-job-title-dropdown',
                                    options=get_job_title_options(df),
                                    value='Data Engineer',  # Most common job in dataset
                                    className="mb-3"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Company Size", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='app-company-size-dropdown',
                                    options=[
                                        {'label': 'Small (148 records)', 'value': 'S'},
                                        {'label': 'Medium (3,153 records)', 'value': 'M'},
                                        {'label': 'Large (454 records)', 'value': 'L'}
                                    ],
                                    value='M',
                                    className="mb-3"
                                )
                            ], md=6)
                        ]),
                        
                        # Row 3: Location & Work Year
                        dbc.Row([
                            dbc.Col([
                                html.Label([
                                    "Company Location ",
                                    html.Small("(for reference only)", className="text-muted")
                                ], className="fw-bold mb-2"),
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
                                        {'label': '2020 (76 records)', 'value': 2020},
                                        {'label': '2021 (230 records)', 'value': 2021},
                                        {'label': '2022 (1,664 records)', 'value': 2022},
                                        {'label': '2023 (1,785 records)', 'value': 2023}
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
                                        0: '0% (1,923)',
                                        50: '50% (189)', 
                                        100: '100% (1,643)'
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
         
 
                 
        # Call to Action Section 
        dbc.Row([ 
            dbc.Col([ 
                html.Hr(className="my-5"), 
                create_cta_section(colors, lang) 
            ]) 
        ]),
         
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

def create_data_overview_section(df, colors, lang='id'):
    """Create data overview section with government portal style - text left, charts right"""
    
    return html.Div([
        dbc.Container([
            dbc.Row([
                # Left Column - Text & Description (like Open Data Jabar)
                dbc.Col([
                    html.Div([
                        html.H2("Data Overview", 
                               className="text-white fw-bold mb-4",
                               style={'fontSize': '2.8rem', 'lineHeight': '1.2'}),
                        
                        html.P("Temukan insights mendalam dari data gaji data science global", 
                               className="text-light mb-4", 
                               style={'fontSize': '1.2rem', 'opacity': '0.9'}),
                        
                        # Feature descriptions like government portal
                        create_overview_feature("Dataset lengkap dan akurat", 
                                              "Analisis komprehensif dari 3,755+ data gaji profesional data science global dari berbagai perusahaan teknologi terkemuka"),
                        
                        create_overview_feature("Cakupan global yang luas", 
                                              "Data dari 57 negara dan 93 posisi berbeda memberikan perspektif internasional yang menyeluruh"),
                        
                        create_overview_feature("Visualisasi interaktif", 
                                              "Charts dan grafik yang mudah dipahami untuk membantu analisis trend dan pattern gaji"),
                        
                        create_overview_feature("Data mutakhir", 
                                              "Informasi terkini dari tahun 2020-2023 dengan fokus pada trend industri teknologi")
                    ], className="overview-content")
                ], md=6, className="d-flex align-items-center"),
                
                # Right Column - Charts, Stats & Table with Parallax Spacing
                dbc.Col([
                    # Quick Stats Grid with extra spacing
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    create_mini_stat_card("3,755", "Total Records", "fas fa-database", colors)
                                ], className="parallax-element")
                            ], md=6, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    create_mini_stat_card("11", "Fitur Data", "fas fa-columns", colors)
                                ], className="parallax-element")
                            ], md=6, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    create_mini_stat_card("57", "Negara", "fas fa-globe", colors)
                                ], className="parallax-element")
                            ], md=6, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    create_mini_stat_card("93", "Job Roles", "fas fa-briefcase", colors)
                                ], className="parallax-element")
                            ], md=6, className="mb-3")
                        ])
                    ], className="mb-4"),
                    
                    # Charts with normal spacing
                    html.Div([
                        html.Div([
                            create_large_chart("Experience Distribution", 
                                              create_experience_chart(df, colors),
                                              "fas fa-chart-pie")
                        ], className="parallax-element mb-4"),
                        
                        html.Div([
                            create_large_chart("Top Countries Salary",
                                              create_country_chart(df, colors), 
                                              "fas fa-globe")
                        ], className="parallax-element mb-4")
                    ])
                ], md=6)
            ], align="center")
        ], fluid=True, className="py-5")
    ], style={
        'background': f'linear-gradient(135deg, {colors["primary"]} 0%, {colors["dark"]} 100%)',
        'margin': '60px 0',
        'padding': '60px 0'
    }, className="data-overview-section")

def create_overview_feature(title, description):
    """Create feature point like government portal"""
    return html.Div([
        html.H5(title, className="text-white fw-bold mb-2"),
        html.P(description, className="text-light mb-4", 
               style={'fontSize': '1rem', 'lineHeight': '1.6', 'opacity': '0.85'})
    ], className="overview-feature mb-4")

def create_mini_stat_card(value, label, icon, colors):
    """Create mini statistics card"""
    return html.Div([
        html.Div([
            html.I(className=f"{icon} fa-lg", style={'color': colors['accent']}),
            html.Div([
                html.H6(value, className="text-white fw-bold mb-0"),
                html.Small(label, className="text-light", style={'fontSize': '0.8rem'})
            ], className="ms-2")
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '12px',
            'background': 'rgba(255, 255, 255, 0.1)',
            'borderRadius': '8px',
            'border': '1px solid rgba(255, 255, 255, 0.2)',
            'transition': 'all 0.3s ease'
        }, className="mini-stat-card")
    ])

def create_compact_chart(title, figure, icon):
    """Create compact chart for right column with parallax effect"""
    return html.Div([
        html.Div([
            html.H6([
                html.I(className=f"{icon} me-2", style={'color': '#34d399'}),
                title
            ], className="text-white fw-bold mb-2", style={'fontSize': '0.9rem'}),
            dcc.Graph(
                figure=figure,
                config={'displayModeBar': False},
                style={'height': '200px'}
            )
        ], style={
            'background': 'rgba(255, 255, 255, 0.08)',
            'borderRadius': '10px',
            'padding': '15px',
            'border': '1px solid rgba(255, 255, 255, 0.15)'
        })
    ], className="compact-chart-container parallax-element")

def create_large_chart(title, figure, icon):
    """Create large chart with more space for parallax effect"""
    return html.Div([
        html.Div([
            html.H5([
                html.I(className=f"{icon} me-2", style={'color': '#34d399'}),
                title
            ], className="text-white fw-bold mb-3"),
            dcc.Graph(
                figure=figure,
                config={'displayModeBar': False},
                style={'height': '350px'}  # Larger chart
            )
        ], style={
            'background': 'rgba(255, 255, 255, 0.1)',
            'borderRadius': '15px',
            'padding': '25px',
            'border': '1px solid rgba(255, 255, 255, 0.2)',
            'boxShadow': '0 10px 30px rgba(0, 0, 0, 0.2)'
        })
    ], className="large-chart-container")

def create_cta_section(colors, lang='id'): 
    """Create simple call-to-action section""" 
    return dbc.Card([ 
        dbc.CardBody([ 
            dbc.Row([ 
                dbc.Col([ 
                    html.Div([ 
                        html.H3(get_text('home', 'cta_title', lang), className="fw-bold text-center mb-3"), 
                        html.P(get_text('home', 'cta_desc', lang), 
                               className="text-center text-muted mb-4"), 
                        html.Div([ 
                            dbc.Button([ 
                                html.I(className="fas fa-rocket me-2"), 
                                get_text('home', 'cta_start', lang)
                            ], color="primary", size="lg", href="/data-overview", className="me-3"), 
                            dbc.Button([ 
                                html.I(className="fas fa-download me-2"), 
                                get_text('home', 'cta_download', lang)
                            ], color="outline-primary", size="lg", href="/download") 
                        ], className="text-center") 
                    ]) 
                ], md=8, className="mx-auto") 
            ]) 
        ], className="py-5") 
    ], className="shadow-sm border-0", 
       style={'background': f'linear-gradient(135deg, {colors["primary"]}10, {colors["light"]}20)'})

def create_navbar(lang='id', current_path='/'):
    """Create horizontal navigation bar with dropdowns"""
    
    # Text translations
    navbar_text = {
        'id': {
            'brand': 'DS Salaries Dashboard',
            'home': 'Beranda',
            'data': 'Data', 
            'data_overview': 'Data Overview',
            'data_cleaning': 'Data Cleaning',
            'analysis': 'Analisis',
            'eda': 'EDA & Visualisasi', 
            'modeling': 'Modeling',
            'results': 'Results',
            'tools': 'Tools',
            'prediction': 'Prediction',
            'download': 'Download',
            'info': 'Info',
            'insights': 'Insights',
            'about': 'About',
            'bantuan': 'Bantuan'
        },
        'en': {
            'brand': 'DS Salaries Dashboard',
            'home': 'Home',
            'data': 'Data',
            'data_overview': 'Data Overview', 
            'data_cleaning': 'Data Cleaning',
            'analysis': 'Analysis',
            'eda': 'EDA & Visualization',
            'modeling': 'Modeling',
            'results': 'Results', 
            'tools': 'Tools',
            'prediction': 'Prediction',
            'download': 'Download',
            'info': 'Info',
            'insights': 'Insights',
            'about': 'About',
            'bantuan': 'Help'
        }
    }
    
    t = navbar_text.get(lang, navbar_text['id'])
    
    # Navigation items with dropdowns
    nav_items = [
        # Home link
        dbc.NavItem(
            dbc.NavLink(
                t['home'], 
                href="/", 
                active="exact"
            )
        ),
        
        # Data dropdown
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem(t['data_overview'], href="/data-overview"),
                dbc.DropdownMenuItem(t['data_cleaning'], href="/data-cleaning"),
            ],
            nav=True,
            in_navbar=True,
            label=t['data']
        ),
        
        # Analysis dropdown  
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem(t['eda'], href="/eda"),
                dbc.DropdownMenuItem(t['modeling'], href="/modeling"),
                dbc.DropdownMenuItem(t['results'], href="/results"),
            ],
            nav=True,
            in_navbar=True,
            label=t['analysis']
        ),
        
        # Tools dropdown
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem(t['prediction'], href="/prediction"),
                dbc.DropdownMenuItem(t['download'], href="/download"),
            ],
            nav=True,
            in_navbar=True,
            label=t['tools']
        ),
        
        # Info dropdown
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem(t['insights'], href="/insights"),
                dbc.DropdownMenuItem(t['about'], href="/about"),
                dbc.DropdownMenuItem(t['bantuan'], href="/bantuan"),
            ],
            nav=True,
            in_navbar=True,
            label=t['info']
        ),
    ]
    
    # Language switcher
    language_switcher = dbc.NavItem([
        dbc.ButtonGroup([
            dbc.Button(
                "ID",
                id="lang-id-btn",
                size="sm",
                outline=True,
                color="light" if lang != 'id' else "primary",
                active=lang == 'id'
            ),
            dbc.Button(
                "EN", 
                id="lang-en-btn",
                size="sm",
                outline=True,
                color="light" if lang != 'en' else "primary",
                active=lang == 'en'
            ),
        ], size="sm")
    ])
    
    # Add language switcher to nav items
    nav_items.append(language_switcher)
    
    # Create navbar
    navbar = dbc.Navbar([
        dbc.Container([
            # Brand section
            dbc.NavbarBrand([
                html.I(className="fas fa-chart-line me-2"),
                t['brand']
            ], 
            href="/", 
            style={
                'fontWeight': 'bold', 
                'fontSize': '1.5rem',
                'color': 'white'
            }),
            
            # Mobile toggle
            dbc.NavbarToggler(id="navbar-toggler"),
            
            # Collapsible navigation
            dbc.Collapse([
                dbc.Nav(nav_items, className="ms-auto", navbar=True)
            ], 
            id="navbar-collapse", 
            navbar=True)
        ])
    ],
    color="primary",
    dark=True, 
    sticky="top",
    className="mb-4")
    
    return navbar 

# Hero Section Component   
def create_hero(lang='id'): 
    return html.Div([ 
        dbc.Container([ 
            dbc.Row([ 
                # Left Side - Text Content 
                dbc.Col([ 
                    html.Div([ 
                        # Main Headline 
                        html.H1(
                            get_text('hero', 'title', lang),
                            style={ 
                                'fontSize': '3.2rem', 
                                'fontWeight': '800', 
                                'lineHeight': '1.1', 
                                'marginBottom': '1.5rem', 
                                'color': '#1f2937' 
                            }, className="highlight-title"
                        ), 
                         
                        # Subtitle 
                        html.P(
                            get_text('hero', 'subtitle', lang),
                            className="hero-subtitle"
                        ), 
                         
                        # CTA Buttons 
                        html.Div([ 
                            dbc.Button([ 
                                html.I(className="fas fa-play me-2"), 
                                get_text('hero', 'btn_start', lang)
                            ],  
                            color="primary",  
                            href="/data-overview", 
                            className="hero-btn-primary me-3"), 
                             
                            dbc.Button([ 
                                get_text('hero', 'btn_predict', lang),
                                html.I(className="fas fa-arrow-right ms-2") 
                            ],  
                            href="/prediction",
                            className="hero-btn-secondary") 
                        ], className="hero-cta") 
                         
                    ], className="hero-text") 
                ], md=6, className="d-flex align-items-center", style={'minHeight': '500px'}), 
                 
                # Right Side - Visual Content 
                dbc.Col([ 
                    html.Div([ 
                        # Main Image Container 
                        html.Div([ 
                            # Text above image with two-tone styling
                            html.Div([
                                html.H3([
                                    html.Span("HOW MUCH DO", style={
                                        'color': '#1f2937',
                                        'fontWeight': '800',
                                        'fontSize': '1.8rem',
                                        'display': 'block',
                                        'marginBottom': '5px'
                                    }),
                                    html.Span("DATA SCIENTISTS MAKE?", style={
                                        'color': '#10b981',
                                        'fontWeight': '800', 
                                        'fontSize': '1.8rem',
                                        'display': 'block'
                                    })
                                ], style={
                                    'textAlign': 'center',
                                    'lineHeight': '1.2',
                                    'marginBottom': '20px',
                                    'fontFamily': 'Inter, sans-serif',
                                    'textTransform': 'uppercase',
                                    'letterSpacing': '1px'
                                })
                            ], className="hero-image-title"),
                            html.Img(
                                src="/assets/1.png", 
                                style={
                                    'width': '100%',
                                    'height': 'auto',
                                    'maxWidth': '600px',
                                    'borderRadius': '0px',
                                    'transition': 'transform 0.3s ease'
                                },
                                className="hero-image"
                            )
                        ], className="hero-image-container"), 
                         
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
                ], md=6, className="d-flex align-items-stretch", style={'minHeight': '500px'}) 
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

# Helper function to get text based on language
def get_text(category, key, lang='id'):
    """Get translated text based on language"""
    try:
        return TEXTS[category][key][lang]
    except KeyError:
        return TEXTS[category][key]['id']  # Fallback to Indonesian

# Main Layout with Footer Integration 
app.layout = html.Div([ 
    dcc.Location(id='url', refresh=False), 
    dcc.Download(id="download-dataframe-csv"), 
    dcc.Store(id='language-store', data='id'),  # Default to Indonesian
    html.Div(id='navbar-container'),
    html.Div(id='page-content'), 
    create_footer(COLORS)
], style={'backgroundColor': COLORS['gray'], 'minHeight': '100vh'}) 

# Callback for navbar update based on language and location
@app.callback(
    Output('navbar-container', 'children'),
    [Input('language-store', 'data'),
     Input('url', 'pathname')]
)
def update_navbar(lang, pathname):
    current_path = pathname if pathname else '/'
    return create_navbar(lang, current_path)

# Callback for mobile navbar toggle
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback for language switching using pattern matching
@app.callback(
    Output('language-store', 'data'),
    [Input({'type': 'lang-switch', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def switch_language(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return 'id'
    
    # Get which button was clicked
    button_info = ctx.triggered[0]['prop_id']
    if 'lang-switch' in button_info:
        if '"index":"en"' in button_info:
            return 'en'
        elif '"index":"id"' in button_info:
            return 'id'
    
    return 'id'

# Callback for page routing
@app.callback( 
    Output('page-content', 'children'), 
    [Input('url', 'pathname'),
     Input('language-store', 'data')], 
    prevent_initial_call=False 
) 
def display_page(pathname, lang): 
    try: 
        # Ensure lang has a default value
        if lang is None:
            lang = 'id'
            
        if pathname == '/' or pathname is None: 
            return html.Div([ 
                create_hero(lang), 
                home.layout(df, COLORS, lang) 
            ]) 
        elif pathname == '/data-overview': 
            return data_overview.layout(df, COLORS, lang) 
        elif pathname == '/eda': 
            return eda.layout(df, COLORS, lang)
        elif pathname == '/eda-advanced':
            return eda_advanced.layout(df, COLORS, lang) 
        elif pathname == '/pca-analysis':
            return pca_analysis.layout(df, COLORS, lang)
        elif pathname == '/cleaning': 
            return data_cleaning.layout(df, COLORS, lang) 
        elif pathname == '/modeling': 
            return modeling.layout(df, COLORS, lang)
        elif pathname == '/modeling-advanced':
            return modeling_advanced.layout(df, COLORS, lang) 
        elif pathname == '/prediction': 
            return prediction.layout(df, COLORS, lang) 
        elif pathname == '/results': 
            return results.layout(df, COLORS, lang) 
        elif pathname == '/insights': 
            return insights.layout(df, COLORS, lang) 
        elif pathname == '/download': 
            return download.layout(df, COLORS, lang) 
        elif pathname == '/about': 
            return about.layout(df, COLORS, lang) 
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
    """Predict salary for integrated form in home page - Updated for real model"""
    if n_clicks is None:
        return html.Div([
            html.Div([
                html.I(className="fas fa-robot fa-4x text-muted mb-3"),
                html.H5("Siap Memprediksi", className="text-muted"),
                html.P("Isi parameter di sebelah kiri dan klik 'Prediksi Gaji' untuk memulai.")
            ], className="text-center py-5")
        ])
    
    if model is None:
        # Enhanced demo prediction based on real dataset patterns
        # Base salary from actual dataset averages
        experience_avg = {
            'EN': 122000,  # Entry level average from dataset
            'MI': 136000,  # Mid level average 
            'SE': 162000,  # Senior level average
            'EX': 198000   # Executive level average
        }
        
        size_multiplier = {'S': 0.85, 'M': 1.0, 'L': 1.15}
        employment_multiplier = {'FT': 1.0, 'PT': 0.7, 'CT': 0.8, 'FL': 0.6}
        
        # Get base from experience level
        base_salary = experience_avg.get(experience, 150000)
        
        # Apply multipliers
        predicted_salary = int(base_salary * 
                             size_multiplier.get(company_size, 1.0) * 
                             employment_multiplier.get(employment, 1.0))
        
        # Note: job_title and location not used in actual model
        
    else:
        try:
            # Prepare input for real model - 5 features only as per trained model
            # Model expects: [work_year, experience_encoded, company_size_encoded, employment_encoded, remote_ratio]
            
            # Encode categorical variables exactly as in training
            experience_mapping = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
            company_size_mapping = {'S': 0, 'M': 1, 'L': 2}
            employment_mapping = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}
            
            # Create feature array in correct order
            features = [
                work_year,
                experience_mapping.get(experience, 2),  # Default to SE if unknown
                company_size_mapping.get(company_size, 1),  # Default to M if unknown  
                employment_mapping.get(employment, 0),  # Default to FT if unknown
                remote_ratio
            ]
            
            # Make prediction
            predicted_salary = int(model.predict([features])[0])
            
        except Exception as e:
            # Fallback to demo prediction if model fails
            print(f"Model prediction failed: {e}")
            experience_avg = {'EN': 122000, 'MI': 136000, 'SE': 162000, 'EX': 198000}
            base_salary = experience_avg.get(experience, 150000)
            size_multiplier = {'S': 0.85, 'M': 1.0, 'L': 1.15}
            predicted_salary = int(base_salary * size_multiplier.get(company_size, 1.0))
    
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
            html.I(className=f"fas fa-{'shield-alt' if model else 'exclamation-triangle'} me-2 {'text-success' if model else 'text-warning'}"),
            html.Small("Prediksi berdasarkan Random Forest model dengan akurasi tinggi" if model else "⚠️ DEMO MODE: Prediksi deterministik - Upload model.pkl untuk prediksi ML yang akurat",
                      className="text-muted fw-bold" if not model else "text-muted")
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
        <!-- Custom favicon - original design -->
        <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPCEtLSBPcmlnaW5hbCBkZXNpZ24gLSBkYXNoYm9hcmQgdGhlbWUgLS0+CjxyZWN0IHg9IjQiIHk9IjQiIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgcng9IjQiIGZpbGw9IiMxMGI5ODEiLz4KPCEtLSBEYXNoYm9hcmQgYmFycyAtLT4KPHJlY3QgeD0iOCIgeT0iMjAiIHdpZHRoPSIyIiBoZWlnaHQ9IjQiIGZpbGw9IndoaXRlIiByeD0iMSIvPgo8cmVjdCB4PSIxMiIgeT0iMTgiIHdpZHRoPSIyIiBoZWlnaHQ9IjYiIGZpbGw9IndoaXRlIiByeD0iMSIvPgo8cmVjdCB4PSIxNiIgeT0iMTQiIHdpZHRoPSIyIiBoZWlnaHQ9IjEwIiBmaWxsPSJ3aGl0ZSIgcng9IjEiLz4KPHJlY3QgeD0iMjAiIHk9IjE2IiB3aWR0aD0iMiIgaGVpZ2h0PSI4IiBmaWxsPSJ3aGl0ZSIgcng9IjEiLz4KPCEtLSBEb2xsYXIgc3ltYm9sIC0tPgo8Y2lyY2xlIGN4PSIyNCIgY3k9IjgiIHI9IjQiIGZpbGw9IiMzNGQzOTkiLz4KPHRleHQgeD0iMjEuNSIgeT0iMTEiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSI2IiBmb250LXdlaWdodD0iNzAwIiBmaWxsPSJ3aGl0ZSI+JDwvdGV4dD4KPC9zdmc+">
        {%css%} 
        <style> 
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: #f3f4f6;
                padding-top: 75px !important; /* Space for fixed navbar */
            }
            
            .navbar-fixed {
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                z-index: 1030 !important;
                backdrop-filter: blur(10px) !important;
                background: rgba(255, 255, 255, 0.95) !important;
                transition: all 0.3s ease !important;
            }
            
            .navbar-fixed:hover {
                background: rgba(255, 255, 255, 0.98) !important;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
            }
            
            /* Logo hover animations */
            .navbar-brand {
                transition: all 0.3s ease !important;
            }
            
            .navbar-brand:hover {
                transform: scale(1.05) !important;
            }
            
            .navbar-brand img {
                transition: all 0.3s ease !important;
            }
            
            .navbar-brand:hover img {
                filter: drop-shadow(0 4px 8px rgba(16, 185, 129, 0.3)) !important;
            }
            
            /* Minimal page content spacing */
            #page-content {
                margin-top: 5px !important;
            }
            
            /* Hero Image Styling */
            .hero-image {
                display: block !important;
                margin: 0 auto !important;
                border: none !important;
                box-shadow: none !important;
            }
            
            .hero-image:hover {
                transform: scale(1.02) !important;
            }
            
            .hero-image-container {
                position: relative !important;
                padding: 0 !important;
                height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: space-between !important;
                align-items: center !important;
            }
            
            .hero-image-title {
                margin-top: 0 !important;
                flex-shrink: 0 !important;
            }
            
            /* Align right column content */
            .hero-visual {
                height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                position: relative !important;
                padding: 20px 0 !important;
            }
            
            /* Hero section improvements */
            .hero-content {
                min-height: 500px !important;
                align-items: stretch !important;
            }
            
            .hero-gradient {
                padding: 60px 0 !important;
            }
            
            /* Simple Hero Styles */
            .hero-gradient {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                padding: 60px 0;
            }
            
            /* Statistics positioning */
            .hero-stats {
                position: absolute;
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 15px 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            } 
             
            .nav-link-modern { 
                font-weight: 500 !important; 
                color: #374151 !important; 
                transition: all 0.4s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
                position: relative !important;
                overflow: hidden !important;
                border-radius: 12px !important;
                padding: 0.5rem 0.8rem !important;
            } 
            
            .nav-link-modern::before {
                content: '' !important;
                position: absolute !important;
                top: 0 !important;
                left: -100% !important;
                width: 100% !important;
                height: 100% !important;
                background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.1), transparent) !important;
                transition: left 0.4s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
            }
            
            .nav-link-modern:hover::before {
                left: 100% !important;
            }
             
            .nav-link-modern:hover { 
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.1)) !important; 
                color: #059669 !important; 
                transform: translateY(-3px) scale(1.02) !important;
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15) !important;
            } 
             
            .nav-link-modern.active { 
                background: linear-gradient(135deg, #10b981, #34d399) !important; 
                color: white !important; 
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
                transform: translateY(-3px) scale(1.02) !important;
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
                transition: all 0.4s cubic-bezier(0.4, 0.0, 0.2, 1) !important; 
                border-radius: 12px !important; 
                padding: 0.5rem 0.8rem !important; 
                text-decoration: none !important; 
                border: none !important; 
                background: transparent !important;
                position: relative !important;
                overflow: hidden !important;
            } 
            
            .nav-dropdown .dropdown-toggle::before {
                content: '' !important;
                position: absolute !important;
                top: 0 !important;
                left: -100% !important;
                width: 100% !important;
                height: 100% !important;
                background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.1), transparent) !important;
                transition: left 0.4s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
            }
            
            .nav-dropdown .dropdown-toggle:hover::before {
                left: 100% !important;
            }
             
            .nav-dropdown .dropdown-toggle:hover { 
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.1)) !important; 
                color: #059669 !important; 
                transform: translateY(-3px) scale(1.02) !important;
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15) !important;
            } 
            
            .nav-dropdown .dropdown-toggle.show,
            .nav-dropdown-active .dropdown-toggle {
                background: linear-gradient(135deg, #10b981, #34d399) !important; 
                color: white !important; 
                transform: translateY(-3px) scale(1.02) !important;
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
            }
            
            .nav-dropdown-active .dropdown-toggle:hover {
                background: linear-gradient(135deg, #059669, #10b981) !important; 
                color: white !important; 
                transform: translateY(-4px) scale(1.03) !important;
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.5) !important;
            }
             
            .nav-dropdown .dropdown-toggle:focus { 
                box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important; 
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.1)) !important; 
                color: #059669 !important; 
                outline: none !important;
            } 
             
            .nav-dropdown .dropdown-menu { 
                border: none !important; 
                border-radius: 16px !important; 
                box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15), 0 10px 25px rgba(0, 0, 0, 0.1) !important; 
                padding: 0.75rem !important; 
                margin-top: 0.75rem !important;
                backdrop-filter: blur(10px) !important;
                background: rgba(255, 255, 255, 0.95) !important;
                animation: dropdownSlideIn 0.3s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
            } 
            
            @keyframes dropdownSlideIn {
                from {
                    opacity: 0 !important;
                    transform: translateY(-10px) scale(0.95) !important;
                }
                to {
                    opacity: 1 !important;
                    transform: translateY(0) scale(1) !important;
                }
            }
             
            .nav-dropdown .dropdown-item { 
                border-radius: 10px !important; 
                padding: 0.75rem 1.25rem !important; 
                transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1) !important; 
                color: #374151 !important; 
                font-weight: 500 !important;
                position: relative !important;
                overflow: hidden !important;
            } 
            
            .nav-dropdown .dropdown-item::before {
                content: '' !important;
                position: absolute !important;
                top: 0 !important;
                left: -100% !important;
                width: 100% !important;
                height: 100% !important;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
                transition: left 0.4s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
            }
            
            .nav-dropdown .dropdown-item:hover::before {
                left: 100% !important;
            }
             
            .nav-dropdown .dropdown-item:hover { 
                background: linear-gradient(135deg, #10b981, #34d399) !important; 
                color: white !important; 
                transform: translateX(8px) scale(1.02) !important;
                box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
            } 
             
            .nav-dropdown .dropdown-item:focus { 
                background: linear-gradient(135deg, #10b981, #34d399) !important; 
                color: white !important; 
                outline: none !important;
                box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
            } 
             
            /* Language Switcher Professional Styles */
            .lang-switcher-container {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .lang-toggle-btn {
                position: relative !important;
                background: transparent !important;
                border: none !important;
                outline: none !important;
                color: inherit !important;
                font-family: 'Inter', sans-serif !important;
                letter-spacing: 0.5px !important;
                text-transform: uppercase !important;
                transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
                border-radius: 18px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                line-height: 1 !important;
            }
            
            .lang-toggle-btn:hover {
                transform: scale(1.05) !important;
            }
            
            .lang-toggle-btn:active {
                transform: scale(0.95) !important;
            }
            
            .lang-slider {
                position: absolute !important;
                top: 2px !important;
                left: 2px !important;
                width: calc(50% - 2px) !important;
                height: calc(100% - 4px) !important;
                background: linear-gradient(135deg, #10b981, #34d399) !important;
                border-radius: 18px !important;
                transition: transform 0.3s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
                box-shadow: 0 2px 12px rgba(16, 185, 129, 0.4) !important;
            }
            
            /* Hover effects for the container */
            .lang-switcher-container:hover .lang-slider {
                box-shadow: 0 4px 20px rgba(16, 185, 129, 0.6) !important;
            }
            
            /* Smooth container hover */
            .lang-switcher-container > div {
                transition: all 0.3s ease !important;
            }
            
            .lang-switcher-container:hover > div {
                transform: scale(1.02) !important;
            }

            /* Job Categories Section Styles - Transparent Background */
            .job-categories-section {
                padding: 20px 10px;
                background: transparent;
                margin: 20px 0;
            }
            
            .simple-category-item {
                transition: all 0.3s ease !important;
            }
            
            .simple-category-item:hover {
                background: rgba(255, 255, 255, 0.9) !important;
                transform: translateY(-5px) !important;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
            }
            
            .category-icon {
                transition: all 0.3s ease !important;
            }
            
            .simple-category-item:hover .category-icon {
                transform: scale(1.1) !important;
                box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2) !important;
            }
            
            .category-link {
                text-decoration: none !important;
            }
            
            .category-link:hover h6 {
                color: #10b981 !important;
            }
            
            /* Job category image hover effects */
            .category-card:hover img {
                transform: scale(1.1) !important;
                transition: transform 0.3s ease !important;
            }
            
            .category-card img {
                transition: transform 0.3s ease !important;
            }
            
            /* Data Overview Section - Government Portal Style */
            .data-overview-section {
                position: relative !important;
                border-radius: 20px !important;
                width: 100vw !important;
                margin-left: calc(-50vw + 50%) !important;
                padding-left: calc(50vw - 50%) !important;
                padding-right: calc(50vw - 50%) !important;
                contain: layout !important;
            }
            
            /* Fix layout issues on mobile */
            @media (max-width: 768px) {
                .data-overview-section {
                    width: 100% !important;
                    margin-left: 0 !important;
                    padding-left: 15px !important;
                    padding-right: 15px !important;
                    border-radius: 15px !important;
                }
            }
            
            .overview-content {
                animation: fadeInLeft 1s ease-out !important;
            }
            
            .overview-feature {
                transition: all 0.3s ease !important;
                padding: 15px 0 !important;
                border-left: 3px solid transparent !important;
                padding-left: 15px !important;
            }
            
            .overview-feature:hover {
                border-left-color: #34d399 !important;
                padding-left: 25px !important;
                background: rgba(255, 255, 255, 0.05) !important;
                border-radius: 8px !important;
            }
            
            .mini-stat-card:hover {
                background: rgba(255, 255, 255, 0.15) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            }
            
            @keyframes fadeInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-30px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            /* Simple Parallax Effects - Working Version */
            .data-overview-section {
                position: relative !important;
                overflow: visible !important; /* Changed from hidden to visible */
            }
            
            /* Parallax elements - simple and effective */
            .parallax-element {
                will-change: transform !important;
                transition: none !important; /* Remove transitions for smooth parallax */
                position: relative !important;
                z-index: 1 !important;
            }
            
            /* Performance optimizations for charts and cards */
            .mini-stat-card,
            .compact-chart-container,
            .large-chart-container {
                transform: translateZ(0) !important; /* GPU acceleration */
                backface-visibility: hidden !important;
            }
            
            .mini-stat-card {
                transition: background-color 0.3s ease, box-shadow 0.3s ease !important;
            }
            
            .mini-stat-card:hover {
                background: rgba(255, 255, 255, 0.15) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            }
            
            /* Mobile optimizations */
            @media (max-width: 768px) {
                .parallax-element {
                    transform: none !important;
                }
            }
            
            /* Reduced motion preferences */
            @media (prefers-reduced-motion: reduce) {
                .parallax-element {
                    transform: none !important;
                }
            }
            
            
            /* Section Titles Enhancement */
            .section-title {
                background: linear-gradient(135deg, #10b981, #059669) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                font-weight: 800 !important;
                font-size: 2.5rem !important;
                margin-bottom: 0.5rem !important;
            }
            
            .section-subtitle {
                color: #6b7280 !important;
                font-size: 1.1rem !important;
                font-weight: 500 !important;
                max-width: 600px !important;
                margin: 0 auto !important;
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
            
            /* Language Switcher Styles */
            .highlight-title {
                background: linear-gradient(135deg, #10b981, #059669);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            /* Language Button Styling */
            .lang-btn {
                font-weight: 600 !important;
                font-size: 12px !important;
                min-width: 45px !important;
                border-radius: 6px !important;
                transition: all 0.3s ease !important;
            }
            
            .btn-outline-primary.lang-btn {
                background: transparent !important;
                border-color: #10b981 !important;
                color: #10b981 !important;
            }
            
            .btn-outline-primary.lang-btn:hover {
                background: #10b981 !important;
                border-color: #10b981 !important;
                color: white !important;
                transform: translateY(-1px);
            }
            
            .btn-primary.lang-btn {
                background: #10b981 !important;
                border-color: #10b981 !important;
                color: white !important;
                box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
            }
            
            .btn-group-sm > .btn {
                border-radius: 6px !important;
            }
            
            .btn-group .btn + .btn {
                margin-left: 2px !important;
                border-left: none !important;
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
        
        <!-- Bootstrap JavaScript for Dropdown Functionality -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        
        <!-- Custom Dropdown Enhancement -->
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Force reinitialize dropdowns after Dash updates
                function initDropdowns() {
                    setTimeout(function() {
                        var dropdownElementList = [].slice.call(document.querySelectorAll('.dropdown-toggle'));
                        dropdownElementList.forEach(function(dropdown) {
                            if (!dropdown.hasAttribute('data-bs-toggle')) {
                                dropdown.setAttribute('data-bs-toggle', 'dropdown');
                            }
                        });
                        
                        // Manual dropdown toggle for better control
                        dropdownElementList.forEach(function(dropdown, index) {
                            dropdown.addEventListener('click', function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                
                                var menu = dropdown.nextElementSibling;
                                if (menu && menu.classList.contains('dropdown-menu')) {
                                    // Close all other dropdowns
                                    document.querySelectorAll('.dropdown-menu.show').forEach(function(otherMenu) {
                                        if (otherMenu !== menu) {
                                            otherMenu.classList.remove('show');
                                            otherMenu.parentElement.classList.remove('show');
                                        }
                                    });
                                    
                                    // Toggle current dropdown
                                    menu.classList.toggle('show');
                                    dropdown.parentElement.classList.toggle('show');
                                }
                            });
                        });
                    }, 100);
                }
                
                // Initialize on page load
                initDropdowns();
                
                // Reinitialize after Dash updates
                var observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.addedNodes.length > 0) {
                            initDropdowns();
                        }
                    });
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
                
                // Close dropdowns when clicking outside
                document.addEventListener('click', function(e) {
                    if (!e.target.closest('.dropdown')) {
                        document.querySelectorAll('.dropdown-menu.show').forEach(function(menu) {
                            menu.classList.remove('show');
                            menu.parentElement.classList.remove('show');
                        });
                    }
                });
            });
        </script>
        
        <script>
            // Simple and effective parallax effect
            function initSimpleParallax() {
                console.log('Initializing simple parallax...');
                
                function updateParallax() {
                    const scrollTop = window.pageYOffset;
                    const parallaxElements = document.querySelectorAll('.parallax-element');
                    
                    if (parallaxElements.length === 0) {
                        console.log('No parallax elements found');
                        return;
                    }
                    
                    parallaxElements.forEach((element, index) => {
                        // More visible parallax: saat scroll down, element bergerak up dengan kecepatan berbeda
                        const speed = 0.3 + (index * 0.1); // Kecepatan yang lebih terlihat
                        const yPos = scrollTop * speed * -0.5; // Setengah dari scroll speed, berlawanan arah
                        
                        // Apply transform
                        element.style.transform = `translateY(${yPos}px)`;
                        element.style.opacity = '1';
                        
                        // Debug info for first element
                        if (index === 0 && scrollTop % 100 < 10) {
                            console.log(`Parallax: scrollTop=${scrollTop}, yPos=${yPos}, element=${element.className}`);
                        }
                    });
                }
                
                // Throttled scroll listener
                let ticking = false;
                function requestTick() {
                    if (!ticking) {
                        requestAnimationFrame(updateParallax);
                        ticking = true;
                        setTimeout(() => { ticking = false; }, 16); // ~60fps
                    }
                }
                
                // Add scroll listener
                window.addEventListener('scroll', requestTick, { passive: true });
                
                // Initial call
                updateParallax();
                
                console.log('Parallax initialized with', document.querySelectorAll('.parallax-element').length, 'elements');
            }
            
            // Initialize when page loads
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initSimpleParallax);
            } else {
                initSimpleParallax();
            }
            
            // Re-initialize after Dash updates
            window.addEventListener('load', function() {
                setTimeout(initSimpleParallax, 1000);
            });
            
            // Additional fallback for Dash apps
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        // Check if parallax elements were added
                        const hasParallaxElements = Array.from(mutation.addedNodes).some(node => 
                            node.nodeType === 1 && (
                                node.classList?.contains('parallax-element') || 
                                node.querySelector?.('.parallax-element')
                            )
                        );
                        
                        if (hasParallaxElements) {
                            console.log('New parallax elements detected, re-initializing...');
                            setTimeout(initSimpleParallax, 500);
                        }
                    }
                });
            });
            
            // Observe the entire document
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        </script>
    </body> 
</html> 
''' 

if __name__ == '__main__': 
    import os
    port = int(os.environ.get('PORT', 8051))
    app.run_server(debug=True, host='127.0.0.1', port=port)