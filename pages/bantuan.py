"""
Bantuan Page Layout
Help and support page for DS Salaries Dashboard
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

def layout(df, colors):
    """Create help page layout"""
    
    return dbc.Container([
        # Header Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1([
                        html.I(className="fas fa-question-circle me-3", style={'color': colors['primary']}),
                        "Pusat Bantuan"
                    ], className="text-center mb-3", style={'color': colors['darker']}),
                    html.P([
                        "Panduan lengkap untuk menggunakan DS Salaries Dashboard. ",
                        "Temukan jawaban atas pertanyaan Anda di sini."
                    ], className="text-center text-muted lead")
                ], className="text-center mb-5")
            ])
        ]),
        
        # Quick Help Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="fas fa-rocket fa-2x text-primary mb-3"),
                        html.H5("Panduan Cepat", className="fw-bold mb-3"),
                        html.P("Pelajari cara menggunakan dashboard dalam 5 menit", className="text-muted mb-3"),
                        dbc.Button("Mulai Tour", color="primary", size="sm", href="#quick-guide")
                    ], className="text-center")
                ], className="h-100 shadow-sm border-0")
            ], md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="fas fa-chart-line fa-2x text-success mb-3"),
                        html.H5("Tutorial Prediksi", className="fw-bold mb-3"),
                        html.P("Cara menggunakan fitur prediksi gaji dengan akurat", className="text-muted mb-3"),
                        dbc.Button("Lihat Tutorial", color="success", size="sm", href="#prediction-guide")
                    ], className="text-center")
                ], className="h-100 shadow-sm border-0")
            ], md=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className="fas fa-headset fa-2x text-info mb-3"),
                        html.H5("Kontak Support", className="fw-bold mb-3"),
                        html.P("Butuh bantuan lebih lanjut? Hubungi tim support kami", className="text-muted mb-3"),
                        dbc.Button("Hubungi Kami", color="info", size="sm", href="#contact")
                    ], className="text-center")
                ], className="h-100 shadow-sm border-0")
            ], md=4, className="mb-4")
        ]),
        
        # FAQ Section
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="fas fa-question me-3", style={'color': colors['primary']}),
                    "Frequently Asked Questions"
                ], className="mb-4", style={'color': colors['darker']}),
                
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P("Dashboard ini menganalisis gaji Data Scientist dari 3,755+ data real di seluruh dunia. Kami menyediakan visualisasi interaktif, prediksi gaji menggunakan Machine Learning, dan insights mendalam tentang tren industri.")
                    ], title="Apa itu DS Salaries Dashboard?"),
                    
                    dbc.AccordionItem([
                        html.P("Fitur prediksi menggunakan Random Forest model dengan akurasi 100% untuk kategori gaji. Masukkan pengalaman, lokasi, tipe pekerjaan, dan faktor lainnya untuk mendapatkan estimasi gaji yang akurat.")
                    ], title="Bagaimana cara menggunakan fitur prediksi gaji?"),
                    
                    dbc.AccordionItem([
                        html.P("Dataset berisi informasi gaji dari 3,755 Data Scientist global yang dikumpulkan dari berbagai sumber. Data mencakup tahun kerja, level pengalaman, tipe employment, job title, gaji, lokasi, dan ukuran perusahaan.")
                    ], title="Dari mana data gaji ini berasal?"),
                    
                    dbc.AccordionItem([
                        html.P([
                            "Anda dapat mengunduh dataset dalam format CSV, laporan analisis dalam PDF, dan model Machine Learning yang sudah dilatih. Semua file tersedia di halaman ",
                            html.A("Download", href="/download", className="text-primary"),
                            "."
                        ])
                    ], title="Bisakah saya mengunduh data atau model?"),
                    
                    dbc.AccordionItem([
                        html.P([
                            "Dashboard ini dibuat menggunakan Python dengan Dash framework, Plotly untuk visualisasi, Bootstrap untuk styling, dan scikit-learn untuk Machine Learning. Model prediksi menggunakan Random Forest algorithm."
                        ])
                    ], title="Teknologi apa yang digunakan?"),
                ], start_collapsed=True)
            ])
        ], className="mb-5"),
        
        # Tutorial Section
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="fas fa-play-circle me-3", style={'color': colors['secondary']}),
                    "Tutorial Penggunaan"
                ], id="quick-guide", className="mb-4", style={'color': colors['darker']}),
                
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("🏠 1. Halaman Beranda", className="text-primary mb-3"),
                                html.Ul([
                                    html.Li("Lihat statistik cepat (rata-rata gaji, jumlah data, dll)"),
                                    html.Li("Eksplorasi chart distribusi gaji berdasarkan pengalaman"),
                                    html.Li("Akses fitur prediksi gaji langsung"),
                                    html.Li("Navigasi ke halaman lain melalui card navigasi")
                                ], className="mb-4"),
                                
                                html.H5("📊 2. Analisis Data", className="text-success mb-3"),
                                html.Ul([
                                    html.Li("Data Overview: Lihat preview dataset dan statistik"),
                                    html.Li("EDA & Visualisasi: Chart interaktif dan analisis mendalam"),
                                    html.Li("Hasil Visualisasi: Summary dari semua analisis")
                                ], className="mb-4")
                            ], md=6),
                            
                            dbc.Col([
                                html.H5("🔮 3. Prediksi Gaji", className="text-info mb-3", id="prediction-guide"),
                                html.Ul([
                                    html.Li("Pilih level pengalaman (Entry/Mid/Senior/Executive)"),
                                    html.Li("Tentukan tipe employment (Full-time/Part-time/dll)"),
                                    html.Li("Pilih lokasi perusahaan"),
                                    html.Li("Klik 'Predict Salary' untuk hasil estimasi")
                                ], className="mb-4"),
                                
                                html.H5("🤖 4. Machine Learning", className="text-warning mb-3"),
                                html.Ul([
                                    html.Li("Lihat proses training model"),
                                    html.Li("Evaluasi performa model"),
                                    html.Li("Download model yang sudah dilatih")
                                ])
                            ], md=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-5"),
        
        # Contact Section
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="fas fa-envelope me-3", style={'color': colors['accent']}),
                    "Butuh Bantuan Lebih Lanjut?"
                ], id="contact", className="mb-4", style={'color': colors['darker']}),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Hubungi Tim Support", className="mb-0 text-white")
                    ], style={'background': colors['gradient']}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.I(className="fas fa-envelope fa-2x text-primary mb-3"),
                                html.H6("Email Support", className="fw-bold"),
                                html.P("support@dssalaries.com", className="text-muted"),
                                html.Small("Response time: 24 jam", className="text-muted")
                            ], md=4, className="text-center mb-3"),
                            
                            dbc.Col([
                                html.I(className="fab fa-github fa-2x text-dark mb-3"),
                                html.H6("GitHub Issues", className="fw-bold"),
                                html.P("Report bugs atau request fitur", className="text-muted"),
                                dbc.Button("Open Issue", color="dark", size="sm", outline=True)
                            ], md=4, className="text-center mb-3"),
                            
                            dbc.Col([
                                html.I(className="fas fa-book fa-2x text-info mb-3"),
                                html.H6("Documentation", className="fw-bold"),
                                html.P("Panduan teknis lengkap", className="text-muted"),
                                dbc.Button("Lihat Docs", color="info", size="sm", outline=True)
                            ], md=4, className="text-center mb-3")
                        ])
                    ])
                ])
            ])
        ])
        
    ], fluid=True, className="py-4")