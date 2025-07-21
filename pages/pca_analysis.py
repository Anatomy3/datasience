"""
PCA Analysis & Dimensionality Reduction Page
Comprehensive Principal Component Analysis dengan explained variance dan visualizations
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def layout(df, colors, lang='id'):
    """Create PCA analysis page layout"""
    from app import get_text
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-compress-alt me-3", style={'color': colors['primary']}),
                            get_text('pca', 'title', lang)
                        ], className="display-5 fw-bold mb-3"),
                        html.P(get_text('pca', 'subtitle', lang),
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Dataset Overview for PCA
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-database me-2"),
                                "Dataset Overview for PCA"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_dataset_overview_pca(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4"),
            
            # PCA Configuration
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-cogs me-2"),
                                "PCA Configuration"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Number of Components:", className="fw-bold mb-2"),
                                    dcc.Slider(
                                        id='pca-components',
                                        min=2, max=10, step=1, value=5,
                                        marks={i: str(i) for i in range(2, 11)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Scaling Method:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='pca-scaling',
                                        options=[
                                            {'label': 'Standard Scaling', 'value': 'standard'},
                                            {'label': 'Min-Max Scaling', 'value': 'minmax'},
                                            {'label': 'No Scaling', 'value': 'none'}
                                        ],
                                        value='standard',
                                        clearable=False
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label(" ", className="fw-bold mb-2 d-block"),
                                    dbc.Button(
                                        [html.I(className="fas fa-play me-2"), "Run PCA Analysis"],
                                        id='run-pca-btn',
                                        color="primary",
                                        className="w-100"
                                    )
                                ], md=4)
                            ])
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4"),
            
            # PCA Results
            dbc.Row([
                dbc.Col([
                    html.Div(id='pca-results')
                ])
            ]),
            
            # Explained Variance Analysis
            dbc.Row([
                dbc.Col([
                    html.Div(id='explained-variance-analysis')
                ])
            ]),
            
            # Principal Components Visualization
            dbc.Row([
                dbc.Col([
                    html.Div(id='pca-visualization')
                ])
            ]),
            
            # Feature Contribution Analysis
            dbc.Row([
                dbc.Col([
                    html.Div(id='feature-contribution')
                ])
            ]),
            
            # Clustering on Principal Components
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-layer-group me-2"),
                                "Clustering on Principal Components"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Number of Clusters:", className="fw-bold mb-2"),
                                    dcc.Slider(
                                        id='cluster-count',
                                        min=2, max=8, step=1, value=3,
                                        marks={i: str(i) for i in range(2, 9)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label(" ", className="fw-bold mb-2 d-block"),
                                    dbc.Button(
                                        [html.I(className="fas fa-sitemap me-2"), "Apply Clustering"],
                                        id='apply-clustering-btn',
                                        color="success",
                                        className="w-100"
                                    )
                                ], md=6)
                            ])
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4"),
            
            # Clustering Results
            dbc.Row([
                dbc.Col([
                    html.Div(id='clustering-results')
                ])
            ]),
            
            # PCA Insights and Recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-lightbulb me-2"),
                                "PCA Insights & Recommendations"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div(id='pca-insights')
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_dataset_overview_pca(df, colors):
    """Create dataset overview for PCA analysis"""
    # Prepare numeric features for PCA
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Create encoded features for demonstration
    df_encoded = df.copy()
    
    # Encode categorical variables
    categorical_mappings = {
        'experience_level': {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4},
        'employment_type': {'PT': 1, 'FL': 2, 'CT': 3, 'FT': 4},
        'company_size': {'S': 1, 'M': 2, 'L': 3}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_encoded.columns:
            df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
    
    # Count total features available for PCA
    all_numeric_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    total_features = len(all_numeric_features)
    
    # Create feature summary cards
    feature_cards = dbc.Row([
        dbc.Col([
            create_feature_summary_card("📊", total_features, "Total Numeric Features", 
                                      "success" if total_features >= 5 else "warning")
        ], md=3, className="mb-3"),
        
        dbc.Col([
            create_feature_summary_card("🔢", len(numeric_features), "Original Numeric", "info")
        ], md=3, className="mb-3"),
        
        dbc.Col([
            create_feature_summary_card("🏷️", len(categorical_features), "Categorical (Encoded)", "secondary")
        ], md=3, className="mb-3"),
        
        dbc.Col([
            create_feature_summary_card("✅", "Ready" if total_features >= 5 else "Limited", 
                                      "PCA Feasibility", "success" if total_features >= 5 else "warning")
        ], md=3, className="mb-3")
    ])
    
    # Feature list
    feature_list = html.Div([
        html.H5("Available Features for PCA Analysis:", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H6("Original Numeric Features:", className="fw-bold text-primary mb-2"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.I(className="fas fa-hashtag me-2 text-primary"),
                        feature
                    ]) for feature in numeric_features
                ])
            ], md=6),
            dbc.Col([
                html.H6("Encoded Categorical Features:", className="fw-bold text-secondary mb-2"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.I(className="fas fa-tag me-2 text-secondary"),
                        f"{col}_encoded"
                    ]) for col in categorical_mappings.keys() if col in df.columns
                ])
            ], md=6)
        ])
    ])
    
    return html.Div([
        feature_cards,
        html.Hr(),
        feature_list,
        html.Hr(),
        dbc.Alert([
            html.H6("PCA Suitability Assessment", className="alert-heading"),
            html.P(f"Dataset contains {total_features} numeric features, which is {'suitable' if total_features >= 5 else 'limited'} for meaningful PCA analysis.", className="mb-2"),
            html.P("PCA will help identify the most important dimensions and reduce dataset complexity while preserving variance.", className="mb-0")
        ], color="success" if total_features >= 5 else "warning")
    ])

def create_feature_summary_card(icon, value, label, color):
    """Create feature summary card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, className="fs-2 mb-2"),
                html.H4(str(value), className="fw-bold mb-1"),
                html.P(label, className="mb-0 text-muted")
            ], className="text-center")
        ])
    ], className=f"border-{color} shadow-sm", style={'borderWidth': '2px'})

# PCA Analysis Callbacks
@callback(
    [Output('pca-results', 'children'),
     Output('explained-variance-analysis', 'children'),
     Output('pca-visualization', 'children'),
     Output('feature-contribution', 'children')],
    [Input('run-pca-btn', 'n_clicks')],
    [State('pca-components', 'value'),
     State('pca-scaling', 'value')]
)
def run_pca_analysis(n_clicks, n_components, scaling_method):
    if n_clicks is None:
        return [html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Click 'Run PCA Analysis' to perform Principal Component Analysis on the dataset."
            ], color="info")
        ])] * 4
    
    # Load data
    from utils.data_loader import load_data
    df = load_data()
    colors = {
        'primary': '#10b981',
        'secondary': '#34d399',
        'accent': '#6ee7b7',
        'light': '#a7f3d0',
        'dark': '#059669'
    }
    
    # Prepare features for PCA
    features, feature_names = prepare_features_for_pca(df)
    
    # Apply scaling
    if scaling_method == 'standard':
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    elif scaling_method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    
    # Store PCA results globally for clustering
    import joblib
    import os
    os.makedirs('data', exist_ok=True)
    joblib.dump({
        'pca': pca,
        'pca_result': pca_result,
        'feature_names': feature_names,
        'scaler': scaler if scaling_method != 'none' else None,
        'scaling_method': scaling_method
    }, 'data/pca_results.pkl')
    
    # Create visualizations
    pca_results_card = create_pca_results_summary(pca, n_components, scaling_method, colors)
    variance_analysis = create_explained_variance_analysis(pca, colors)
    pca_viz = create_pca_visualization(pca_result, pca, colors)
    feature_contrib = create_feature_contribution_analysis(pca, feature_names, colors)
    
    return pca_results_card, variance_analysis, pca_viz, feature_contrib

@callback(
    [Output('clustering-results', 'children'),
     Output('pca-insights', 'children')],
    [Input('apply-clustering-btn', 'n_clicks')],
    [State('cluster-count', 'value')]
)
def apply_clustering_on_pca(n_clicks, n_clusters):
    if n_clicks is None:
        return html.Div(), create_default_pca_insights()
    
    try:
        # Load PCA results
        import joblib
        pca_data = joblib.load('data/pca_results.pkl')
        pca_result = pca_data['pca_result']
        pca = pca_data['pca']
        
        colors = {
            'primary': '#10b981',
            'secondary': '#34d399',
            'accent': '#6ee7b7',
            'light': '#a7f3d0',
            'dark': '#059669'
        }
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_result)
        
        # Create clustering visualization
        clustering_viz = create_clustering_visualization(pca_result, cluster_labels, colors)
        
        # Create comprehensive insights
        insights = create_comprehensive_pca_insights(pca, pca_result, cluster_labels, n_clusters)
        
        return clustering_viz, insights
        
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Please run PCA analysis first before applying clustering."
            ], color="warning")
        ]), create_default_pca_insights()

# Helper Functions
def prepare_features_for_pca(df):
    """Prepare features for PCA analysis"""
    df_pca = df.copy()
    
    # Encode categorical variables
    categorical_mappings = {
        'experience_level': {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4},
        'employment_type': {'PT': 1, 'FL': 2, 'CT': 3, 'FT': 4},
        'company_size': {'S': 1, 'M': 2, 'L': 3}
    }
    
    feature_names = []
    
    # Add numeric features
    numeric_features = df_pca.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_features:
        feature_names.append(col)
    
    # Add encoded categorical features
    for col, mapping in categorical_mappings.items():
        if col in df_pca.columns:
            df_pca[f'{col}_encoded'] = df_pca[col].map(mapping)
            feature_names.append(f'{col}_encoded')
    
    # Select all numeric features (original + encoded)
    all_features = df_pca.select_dtypes(include=[np.number]).columns.tolist()
    features = df_pca[all_features].values
    
    return features, all_features

def create_pca_results_summary(pca, n_components, scaling_method, colors):
    """Create PCA results summary"""
    total_variance = sum(pca.explained_variance_ratio_) * 100
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-chart-pie me-2"),
                "PCA Analysis Results"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-compress fa-2x text-primary mb-3"),
                            html.H4(f"{n_components}", className="fw-bold"),
                            html.P("Components", className="mb-0")
                        ])
                    ], className="text-center shadow-sm border-0")
                ], md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-percentage fa-2x text-success mb-3"),
                            html.H4(f"{total_variance:.1f}%", className="fw-bold"),
                            html.P("Total Variance", className="mb-0")
                        ])
                    ], className="text-center shadow-sm border-0")
                ], md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-balance-scale fa-2x text-info mb-3"),
                            html.H4(scaling_method.title(), className="fw-bold"),
                            html.P("Scaling Method", className="mb-0")
                        ])
                    ], className="text-center shadow-sm border-0")
                ], md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-chart-line fa-2x text-warning mb-3"),
                            html.H4(f"{100-total_variance:.1f}%", className="fw-bold"),
                            html.P("Information Loss", className="mb-0")
                        ])
                    ], className="text-center shadow-sm border-0")
                ], md=3)
            ])
        ])
    ], className="shadow-sm border-0 mb-4")

def create_explained_variance_analysis(pca, colors):
    """Create explained variance analysis"""
    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)
    
    # Create variance plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Individual Explained Variance', 'Cumulative Explained Variance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Individual variance
    fig.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(len(explained_var))],
            y=explained_var,
            name='Explained Variance',
            marker_color=colors['primary']
        ),
        row=1, col=1
    )
    
    # Cumulative variance
    fig.add_trace(
        go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumulative_var))],
            y=cumulative_var,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color=colors['secondary'], width=3),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Add 80% and 95% lines for cumulative variance
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="80% Threshold", row=1, col=2)
    fig.add_hline(y=95, line_dash="dash", line_color="orange", 
                  annotation_text="95% Threshold", row=1, col=2)
    
    fig.update_layout(
        height=400,
        title_text="Principal Components Explained Variance Analysis",
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create variance table
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_var))],
        'Explained Variance (%)': [f"{var:.2f}" for var in explained_var],
        'Cumulative Variance (%)': [f"{cum:.2f}" for cum in cumulative_var]
    })
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-chart-area me-2"),
                "Explained Variance Analysis"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.Hr(),
            html.H5("Variance Breakdown", className="mb-3"),
            dash_table.DataTable(
                data=variance_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in variance_df.columns],
                style_cell={'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'}
            )
        ])
    ], className="shadow-sm border-0 mb-4")

def create_pca_visualization(pca_result, pca, colors):
    """Create PCA visualization"""
    # 2D visualization (first two components)
    fig_2d = px.scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        title='PCA: First Two Principal Components',
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
        color_discrete_sequence=[colors['primary']]
    )
    
    fig_2d.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 3D visualization if we have at least 3 components
    fig_3d = None
    if pca_result.shape[1] >= 3:
        fig_3d = px.scatter_3d(
            x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
            title='PCA: First Three Principal Components',
            labels={
                'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                'z': f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)'
            },
            color_discrete_sequence=[colors['secondary']]
        )
        
        fig_3d.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    components = [
        dbc.Col([
            dcc.Graph(figure=fig_2d, config={'displayModeBar': False})
        ], md=6 if fig_3d else 12)
    ]
    
    if fig_3d:
        components.append(
            dbc.Col([
                dcc.Graph(figure=fig_3d, config={'displayModeBar': False})
            ], md=6)
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-cube me-2"),
                "Principal Components Visualization"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row(components)
        ])
    ], className="shadow-sm border-0 mb-4")

def create_feature_contribution_analysis(pca, feature_names, colors):
    """Create feature contribution analysis"""
    # Get the components matrix
    components = pca.components_
    
    # Create heatmap of feature contributions
    fig = px.imshow(
        components,
        x=feature_names,
        y=[f'PC{i+1}' for i in range(len(components))],
        color_continuous_scale='RdBu',
        title='Feature Contributions to Principal Components',
        aspect='auto'
    )
    
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create top contributors for first few components
    top_contributors = []
    for i, component in enumerate(components[:3]):  # First 3 components
        abs_contributions = np.abs(component)
        top_indices = np.argsort(abs_contributions)[-5:][::-1]  # Top 5
        
        contrib_data = []
        for idx in top_indices:
            contrib_data.append({
                'Feature': feature_names[idx],
                'Contribution': component[idx],
                'Abs Contribution': abs_contributions[idx]
            })
        
        top_contributors.append({
            'component': f'PC{i+1}',
            'data': contrib_data
        })
    
    # Create contribution tables
    contribution_tables = []
    for contrib in top_contributors:
        df_contrib = pd.DataFrame(contrib['data'])
        
        contribution_tables.append(
            dbc.Col([
                html.H6(f"Top Contributors - {contrib['component']}", className="fw-bold mb-2"),
                dash_table.DataTable(
                    data=df_contrib.round(3).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_contrib.columns],
                    style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '12px'},
                    style_header={'backgroundColor': colors['secondary'], 'color': 'white', 'fontWeight': 'bold'}
                )
            ], md=4, className="mb-3")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-weight-hanging me-2"),
                "Feature Contribution Analysis"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.Hr(),
            html.H5("Top Feature Contributors by Component", className="mb-3"),
            dbc.Row(contribution_tables)
        ])
    ], className="shadow-sm border-0 mb-4")

def create_clustering_visualization(pca_result, cluster_labels, colors):
    """Create clustering visualization on PCA space"""
    # 2D clustering plot
    fig_2d = px.scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        color=cluster_labels,
        title='K-Means Clustering on PCA Space (2D)',
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component', 'color': 'Cluster'},
        color_continuous_scale='viridis'
    )
    
    fig_2d.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 3D clustering plot if available
    fig_3d = None
    if pca_result.shape[1] >= 3:
        fig_3d = px.scatter_3d(
            x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
            color=cluster_labels,
            title='K-Means Clustering on PCA Space (3D)',
            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Cluster'},
            color_continuous_scale='viridis'
        )
        
        fig_3d.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    # Cluster statistics
    unique_clusters = np.unique(cluster_labels)
    cluster_stats = []
    
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_size = np.sum(cluster_mask)
        cluster_percentage = (cluster_size / len(cluster_labels)) * 100
        
        cluster_stats.append({
            'Cluster': f'Cluster {cluster}',
            'Size': cluster_size,
            'Percentage': f'{cluster_percentage:.1f}%'
        })
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    components = [
        dbc.Col([
            dcc.Graph(figure=fig_2d, config={'displayModeBar': False})
        ], md=6 if fig_3d else 12)
    ]
    
    if fig_3d:
        components.append(
            dbc.Col([
                dcc.Graph(figure=fig_3d, config={'displayModeBar': False})
            ], md=6)
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-layer-group me-2"),
                "Clustering Results on PCA Space"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row(components),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H5("Cluster Statistics", className="mb-3"),
                    dash_table.DataTable(
                        data=cluster_stats_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in cluster_stats_df.columns],
                        style_cell={'textAlign': 'center', 'padding': '10px'},
                        style_header={'backgroundColor': colors['accent'], 'color': 'white', 'fontWeight': 'bold'}
                    )
                ], md=6),
                dbc.Col([
                    html.H5("Clustering Insights", className="mb-3"),
                    dbc.Alert([
                        html.H6("Key Findings:", className="alert-heading"),
                        html.Ul([
                            html.Li(f"Dataset clustered into {len(unique_clusters)} distinct groups"),
                            html.Li(f"Largest cluster contains {max(cluster_stats, key=lambda x: x['Size'])['Percentage']} of data"),
                            html.Li("Clusters show clear separation in PCA space"),
                            html.Li("PCA dimensions preserve cluster structure effectively")
                        ])
                    ], color="info")
                ], md=6)
            ])
        ])
    ], className="shadow-sm border-0 mb-4")

def create_comprehensive_pca_insights(pca, pca_result, cluster_labels, n_clusters):
    """Create comprehensive PCA insights"""
    explained_variance = pca.explained_variance_ratio_ * 100
    total_variance = sum(explained_variance)
    
    insights = []
    
    # Dimensionality reduction insights
    insights.append(dbc.Alert([
        html.H6("📉 Dimensionality Reduction Analysis", className="alert-heading"),
        html.P(f"PCA successfully reduced dataset complexity while preserving {total_variance:.1f}% of original variance.", className="mb-2"),
        html.P(f"First component captures {explained_variance[0]:.1f}% of variance, indicating strong primary pattern in data.", className="mb-0")
    ], color="success"))
    
    # Variance distribution insights
    if explained_variance[0] > 50:
        insights.append(dbc.Alert([
            html.H6("🎯 Dominant Pattern Detected", className="alert-heading"),
            html.P(f"First principal component explains {explained_variance[0]:.1f}% of variance, suggesting a strong underlying pattern dominates the dataset.", className="mb-0")
        ], color="info"))
    
    # Component balance insights
    variance_balance = np.std(explained_variance[:3]) if len(explained_variance) >= 3 else 0
    if variance_balance < 5:
        insights.append(dbc.Alert([
            html.H6("⚖️ Balanced Components", className="alert-heading"),
            html.P("Principal components show relatively balanced variance distribution, indicating multiple important dimensions in the data.", className="mb-0")
        ], color="info"))
    
    # Clustering insights
    insights.append(dbc.Alert([
        html.H6("🔍 Clustering Analysis", className="alert-heading"),
        html.P(f"K-means clustering identified {n_clusters} distinct groups in the PCA space.", className="mb-2"),
        html.P("This suggests natural segmentation exists in the salary data patterns.", className="mb-0")
    ], color="primary"))
    
    # Recommendations
    recommendations = dbc.Alert([
        html.H6("💡 Recommendations for Further Analysis", className="alert-heading"),
        html.Ul([
            html.Li(f"Use first {min(3, len(explained_variance))} components for visualization and modeling (capture {sum(explained_variance[:3]):.1f}% variance)"),
            html.Li("Consider the identified clusters for salary segmentation analysis"),
            html.Li("Investigate feature contributions to understand what drives salary variation"),
            html.Li("Apply PCA-transformed features in machine learning models for potentially better performance")
        ], className="mb-0")
    ], color="warning")
    
    return html.Div([
        html.H5("Comprehensive PCA Analysis Insights", className="mb-4"),
        html.Div(insights),
        recommendations
    ])

def create_default_pca_insights():
    """Create default PCA insights when clustering is not applied"""
    return html.Div([
        html.H5("PCA Analysis Benefits", className="mb-4"),
        dbc.Alert([
            html.H6("🔬 Why Use PCA for Salary Data?", className="alert-heading"),
            html.Ul([
                html.Li("Reduces dataset complexity while preserving important patterns"),
                html.Li("Helps identify the most important factors affecting salary"),
                html.Li("Enables better visualization of high-dimensional data"),
                html.Li("Can improve machine learning model performance"),
                html.Li("Reveals hidden relationships between features")
            ])
        ], color="info"),
        
        dbc.Alert([
            html.H6("📊 Next Steps", className="alert-heading"),
            html.P("Run the PCA analysis to discover the principal components and apply clustering to identify salary patterns.", className="mb-0")
        ], color="primary")
    ])