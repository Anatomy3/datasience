""" 
Home Page Layout 
Dashboard overview, quick statistics, and prediction section 
""" 

import dash_bootstrap_components as dbc 
from dash import html, dcc 
import plotly.express as px 
import plotly.graph_objects as go 

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
         
        # Salary Prediction Section (DIPINDAH KE SINI - SETELAH CHARTS) 
        dbc.Row([ 
            dbc.Col([ 
                html.H3([ 
                    html.I(className="fas fa-crystal-ball me-3", style={'color': colors['primary']}), 
                    "Prediksi Gaji Data Sience" 
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
                             
                            # Right Side - Preview/Demo (UPDATED) 
                            dbc.Col([ 
                                html.Div([ 
                                    # Image preview - larger and touch bottom (NO "Sample Prediction" text) 
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
                                       style={ 
                                           'height': '100%',  
                                           'display': 'flex',  
                                           'alignItems': 'flex-end', 
                                           'justifyContent': 'center' 
                                       }) 
                                     
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
        ]) 
         
    ], fluid=True, className="py-4") 

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
     
    # Calculate key metrics 
    avg_salary = df['salary_in_usd'].mean() 
    max_salary = df['salary_in_usd'].max() 
    min_salary = df['salary_in_usd'].min() 
     
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean() 
    top_job = df['job_title'].value_counts().index[0] 
    top_country = df['company_location'].value_counts().index[0] 
     
    insights = [ 
        { 
            'icon': '💰', 
            'title': 'Salary Range', 
            'text': f'Gaji berkisar dari ${min_salary:,.0f} hingga ${max_salary:,.0f} dengan rata-rata ${avg_salary:,.0f}' 
        }, 
        { 
            'icon': '📈', 
            'title': 'Experience Premium', 
            'text': f'Senior level mendapat gaji {(exp_salary.get("SE", 0) / exp_salary.get("EN", 1)):.1f}x lebih tinggi dari entry level' 
        }, 
        { 
            'icon': '🌟', 
            'title': 'Top Job Title', 
            'text': f'{top_job} adalah posisi paling umum dengan {df[df["job_title"] == top_job].shape[0]} openings' 
        }, 
        { 
            'icon': '🌍', 
            'title': 'Leading Country', 
            'text': f'{top_country} memimpin dengan {df[df["company_location"] == top_country].shape[0]} companies' 
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
                         
                        # Download Cards 
                        dbc.Row([ 
                            dbc.Col([ 
                                dbc.Card([ 
                                    dbc.CardBody([ 
                                        html.Div([ 
                                            html.I(className="fas fa-file-csv fa-2x text-success mb-3"), 
                                            html.H6("Dataset (CSV)", className="fw-bold mb-2"), 
                                            html.P("Download complete dataset", className="text-muted small mb-3"), 
                                            dbc.Button([ 
                                                html.I(className="fas fa-download me-2"), 
                                                "Download CSV" 
                                            ], color="success", size="sm", className="w-100", id="btn-download-csv-home") # <-- ID DITAMBAHKAN DI SINI
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
                                            html.H6("Analysis Report", className="fw-bold mb-2"), 
                                            html.P("Summary insights & findings", className="text-muted small mb-3"), 
                                            dbc.Button([ 
                                                html.I(className="fas fa-file-pdf me-2"), 
                                                "Download PDF" 
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
                                            html.H6("ML Model", className="fw-bold mb-2"), 
                                            html.P("Trained prediction model", className="text-muted small mb-3"), 
                                            dbc.Button([ 
                                                html.I(className="fas fa-download me-2"), 
                                                "Download Model" 
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
             
            # Additional Info 
            html.Div([ 
                html.P([ 
                    html.I(className="fas fa-info-circle me-2 text-primary"), 
                    "Semua file download tersedia dalam format yang kompatibel dengan tools analisis populer" 
                ], className="text-center text-muted mb-0 small") 
            ]) 
        ]) 
    ], className="shadow-sm border-0 mb-4")