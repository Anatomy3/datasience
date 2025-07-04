""" 
Home Page Layout 
Dashboard overview, quick statistics, and prediction section 
""" 

import dash_bootstrap_components as dbc 
from dash import html, dcc, Input, Output, State, callback
import plotly.express as px 
import plotly.graph_objects as go 
import pandas as pd
import joblib
import numpy as np

# Load the trained model for home page predictions
try:
    model = joblib.load('data/model.pkl')
    print("Model loaded successfully for home page!")
except FileNotFoundError:
    print("Model file not found. Prediction will not work.")
    model = None

# =====================================================================
#  BAGIAN BARU: Fungsi untuk Membuat Section 'Tentang Tim' Sesuai Gaya
# =====================================================================
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


def layout(df, colors): 
    """Create home page layout""" 
     
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
         
        # SALARY PREDICTION SECTION - UPDATED WITH INTEGRATED FORM
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
                                    id='home-experience-dropdown',
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
                                    id='home-employment-dropdown',
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
                                    id='home-job-title-dropdown',
                                    options=get_job_title_options(df),
                                    value='Data Scientist',
                                    className="mb-3"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Company Size", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='home-company-size-dropdown',
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
                                    id='home-location-dropdown',
                                    options=get_location_options(df),
                                    value='US',
                                    className="mb-3"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Work Year", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='home-work-year-dropdown',
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
                                    id='home-remote-slider',
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
                                id='home-predict-button',
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
                        html.Div(id='home-prediction-results', children=[
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
        #  BAGIAN TIM DIPINDAHKAN KE SINI (POSISI AKHIR)
        # =====================================================================
        create_team_section_styled(colors),
         
    ], fluid=True, className="py-4")

# Helper functions for dropdown options
def get_job_title_options(df):
    """Get job title options from dataset"""
    job_titles = sorted(df['job_title'].unique())
    return [{'label': title, 'value': title} for title in job_titles[:20]]  # Top 20

def get_location_options(df):
    """Get location options from dataset"""
    locations = sorted(df['company_location'].unique())
    return [{'label': loc, 'value': loc} for loc in locations]

# Callback for home page prediction
@callback(
    Output('home-prediction-results', 'children'),
    Input('home-predict-button', 'n_clicks'),
    [
        State('home-work-year-dropdown', 'value'),
        State('home-experience-dropdown', 'value'),
        State('home-employment-dropdown', 'value'),
        State('home-job-title-dropdown', 'value'),
        State('home-location-dropdown', 'value'),
        State('home-company-size-dropdown', 'value'),
        State('home-remote-slider', 'value')
    ],
    prevent_initial_call=True
)
def predict_salary_home(n_clicks, work_year, experience, employment, job_title, location, company_size, remote_ratio):
    """Predict salary based on input parameters for home page"""
    if n_clicks is None or model is None:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Model tidak tersedia. Pastikan model sudah dilatih."
            ], color="warning")
        ])
    
    try:
        # Create input DataFrame (sesuaikan dengan fitur yang digunakan model Anda)
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
        predicted_salary = model.predict(input_data)[0]
        
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
                html.Small("Prediksi berdasarkan Random Forest model dengan akurasi 100% kategori",
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
        
    except Exception as e:
        return dbc.Alert([
            html.H5("Prediksi Error", className="alert-heading"),
            html.P(f"Error: {str(e)}"),
            html.P("Silakan periksa file model dan parameter input.")
        ], color="danger")

# Helper functions for labels
def get_experience_label(exp):
    mapping = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
    return mapping.get(exp, exp)

def get_employment_label(emp):
    mapping = {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}
    return mapping.get(emp, emp)

def create_stat_card(icon, value, label, color): 
    """Create animated statistics card""" 
    return dbc.Card([ 
        dbc.CardBody([ 
            html.Div([ 
                html.Div(icon, className="fs-2 mb-2"), 
                html.H3(value, className="fw-bold mb-1", style={'color': color}), 
                html.P(label, className="mb-0 text-muted fw-medium") 
            ], className="text-center") 
        ]) 
    ], className="h-100 border-0 shadow-sm card-hover", 
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
    ], className="h-100 border-0 shadow-sm card-hover", 
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