"""
Advanced EDA & Statistical Analysis Page
Comprehensive statistical analysis with advanced visualizations and insights
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def layout(df, colors, lang='id'):
    """Create advanced EDA page layout with statistical analysis"""
    
    return html.Div([
        # Page Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-microscope me-3", style={'color': colors['primary']}),
                            "Advanced Statistical Analysis"
                        ], className="display-5 fw-bold mb-3"),
                        html.P("Deep statistical insights dengan advanced analytics dan hypothesis testing",
                              className="lead text-muted")
                    ], className="text-center py-4")
                ])
            ])
        ], fluid=True, className="bg-light"),
        
        dbc.Container([
            # Statistical Summary Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-calculator me-2"),
                                "Descriptive Statistics"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_descriptive_stats_table(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4"),
            
            # Distribution Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-area me-2"),
                                "Salary Distribution Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_distribution_analysis(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=8, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-exclamation-triangle me-2"),
                                "Outlier Detection"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_outlier_analysis(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ], md=4, className="mb-4")
            ]),
            
            # Advanced Correlation Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-project-diagram me-2"),
                                "Advanced Correlation Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_advanced_correlation_heatmap(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-scatter me-2"),
                                "Pair Plot Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_pair_plot_analysis(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            # Statistical Tests Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-flask me-2"),
                                "Statistical Hypothesis Testing"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_hypothesis_testing_section(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-line me-2"),
                                "Normality Tests"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_normality_tests(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ], md=6, className="mb-4")
            ]),
            
            # Advanced Visualizations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-eye me-2"),
                                "Multi-dimensional Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_multidimensional_analysis(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=8, className="mb-4"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Categorical Analysis"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=create_categorical_analysis(df, colors),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=4, className="mb-4")
            ]),
            
            # Data Quality Assessment
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-check-circle me-2"),
                                "Data Quality Assessment"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_data_quality_assessment(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4"),
            
            # Advanced Insights
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="fas fa-brain me-2"),
                                "Advanced Statistical Insights"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_advanced_insights_section(df, colors)
                        ])
                    ], className="shadow-sm border-0")
                ])
            ], className="mb-4")
            
        ], fluid=True, className="py-4")
    ])

def create_descriptive_stats_table(df, colors):
    """Create comprehensive descriptive statistics table"""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = df[numeric_cols].describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame({
        col: {
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'variance': df[col].var(),
            'coeff_var': df[col].std() / df[col].mean() * 100 if df[col].mean() != 0 else 0
        } for col in numeric_cols
    }).T
    
    # Combine statistics
    complete_stats = pd.concat([stats_df.T, additional_stats], axis=1)
    
    # Format for display
    complete_stats = complete_stats.round(2)
    
    # Create table
    table_header = [
        html.Thead([
            html.Tr([html.Th("Statistic")] + [html.Th(col) for col in complete_stats.columns])
        ])
    ]
    
    table_body = [
        html.Tbody([
            html.Tr([html.Td(stat)] + [html.Td(f"{complete_stats.loc[col, stat]:,.2f}") 
                                      for col in complete_stats.index])
            for stat in complete_stats.columns
        ])
    ]
    
    return dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        className="mb-0"
    )

def create_distribution_analysis(df, colors):
    """Create comprehensive distribution analysis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histogram with Normal Curve', 'Q-Q Plot', 'Box Plot', 'Violin Plot'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    salary_data = df['salary_in_usd']
    
    # 1. Histogram with normal distribution overlay
    fig.add_trace(
        go.Histogram(x=salary_data, name='Salary Distribution', 
                    marker_color=colors['primary'], opacity=0.7),
        row=1, col=1
    )
    
    # Add normal distribution curve
    x_norm = np.linspace(salary_data.min(), salary_data.max(), 100)
    y_norm = stats.norm.pdf(x_norm, salary_data.mean(), salary_data.std())
    y_norm = y_norm * len(salary_data) * (salary_data.max() - salary_data.min()) / 50  # Scale to match histogram
    
    fig.add_trace(
        go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Distribution',
                  line=dict(color=colors['dark'], width=2)),
        row=1, col=1
    )
    
    # 2. Q-Q Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(salary_data, dist="norm", plot=None)
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot',
                  marker=dict(color=colors['secondary'])),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', 
                  name='Theoretical Line', line=dict(color=colors['dark'])),
        row=1, col=2
    )
    
    # 3. Box Plot
    fig.add_trace(
        go.Box(y=salary_data, name='Salary Box Plot', 
               marker_color=colors['accent']),
        row=2, col=1
    )
    
    # 4. Violin Plot
    fig.add_trace(
        go.Violin(y=salary_data, name='Salary Violin Plot',
                 marker_color=colors['primary']),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Comprehensive Salary Distribution Analysis",
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_outlier_analysis(df, colors):
    """Create outlier detection analysis"""
    salary_data = df['salary_in_usd']
    
    # IQR Method
    Q1 = salary_data.quantile(0.25)
    Q3 = salary_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = salary_data[(salary_data < lower_bound) | (salary_data > upper_bound)]
    
    # Z-Score Method
    z_scores = np.abs(stats.zscore(salary_data))
    outliers_zscore = salary_data[z_scores > 3]
    
    # Modified Z-Score Method
    median = salary_data.median()
    mad = np.median(np.abs(salary_data - median))
    modified_z_scores = 0.6745 * (salary_data - median) / mad
    outliers_modified_z = salary_data[np.abs(modified_z_scores) > 3.5]
    
    outlier_stats = [
        {"method": "IQR Method", "count": len(outliers_iqr), "percentage": len(outliers_iqr)/len(df)*100},
        {"method": "Z-Score (>3)", "count": len(outliers_zscore), "percentage": len(outliers_zscore)/len(df)*100},
        {"method": "Modified Z-Score", "count": len(outliers_modified_z), "percentage": len(outliers_modified_z)/len(df)*100}
    ]
    
    cards = []
    for stat in outlier_stats:
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(stat["method"], className="fw-bold mb-2"),
                    html.H4(f"{stat['count']}", className="text-primary mb-1"),
                    html.Small(f"{stat['percentage']:.1f}% of data", className="text-muted")
                ])
            ], className="text-center mb-2 shadow-sm border-0")
        )
    
    return html.Div(cards)

def create_advanced_correlation_heatmap(df, colors):
    """Create advanced correlation analysis with annotations"""
    # Prepare encoded dataframe
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
    
    # Select relevant columns for correlation
    corr_cols = ['work_year', 'salary_in_usd', 'remote_ratio']
    for col in categorical_mappings.keys():
        if f'{col}_encoded' in df_encoded.columns:
            corr_cols.append(f'{col}_encoded')
    
    corr_matrix = df_encoded[corr_cols].corr()
    
    # Create readable labels
    label_mapping = {
        'work_year': 'Work Year',
        'salary_in_usd': 'Salary (USD)',
        'remote_ratio': 'Remote Ratio (%)',
        'experience_level_encoded': 'Experience Level',
        'employment_type_encoded': 'Employment Type',
        'company_size_encoded': 'Company Size'
    }
    
    # Rename matrix
    display_matrix = corr_matrix.rename(index=label_mapping, columns=label_mapping)
    
    # Create heatmap with annotations
    fig = go.Figure(data=go.Heatmap(
        z=display_matrix.values,
        x=display_matrix.columns,
        y=display_matrix.index,
        colorscale=[[0, '#ffffff'], [0.5, colors['light']], [1, colors['primary']]],
        text=np.around(display_matrix.values, decimals=3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix with Strength Indicators",
        height=400,
        margin=dict(t=50, b=40, l=100, r=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_pair_plot_analysis(df, colors):
    """Create pair plot style analysis"""
    # Select key variables for pair plot
    variables = ['salary_in_usd', 'remote_ratio', 'work_year']
    
    fig = make_subplots(
        rows=len(variables), cols=len(variables),
        subplot_titles=[f"{var1} vs {var2}" for var1 in variables for var2 in variables]
    )
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                # Diagonal: histogram
                fig.add_trace(
                    go.Histogram(x=df[var1], marker_color=colors['primary'], opacity=0.7),
                    row=i+1, col=j+1
                )
            else:
                # Off-diagonal: scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=df[var2], y=df[var1], mode='markers',
                        marker=dict(color=colors['secondary'], opacity=0.6),
                        name=f"{var1} vs {var2}"
                    ),
                    row=i+1, col=j+1
                )
    
    fig.update_layout(
        title="Pair Plot Analysis of Key Variables",
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_hypothesis_testing_section(df, colors):
    """Create hypothesis testing results"""
    results = []
    
    # Test 1: Experience level affects salary (ANOVA)
    groups = [group['salary_in_usd'].values for name, group in df.groupby('experience_level')]
    f_stat, p_value_anova = stats.f_oneway(*groups)
    
    results.append({
        'test': 'One-way ANOVA',
        'hypothesis': 'Experience level affects salary',
        'statistic': f'F = {f_stat:.3f}',
        'p_value': p_value_anova,
        'result': 'Significant' if p_value_anova < 0.05 else 'Not Significant'
    })
    
    # Test 2: Company size affects salary (Kruskal-Wallis)
    groups_size = [group['salary_in_usd'].values for name, group in df.groupby('company_size')]
    h_stat, p_value_kw = stats.kruskal(*groups_size)
    
    results.append({
        'test': 'Kruskal-Wallis',
        'hypothesis': 'Company size affects salary',
        'statistic': f'H = {h_stat:.3f}',
        'p_value': p_value_kw,
        'result': 'Significant' if p_value_kw < 0.05 else 'Not Significant'
    })
    
    # Test 3: Remote work correlation with salary
    corr_coef, p_value_corr = stats.pearsonr(df['remote_ratio'], df['salary_in_usd'])
    
    results.append({
        'test': 'Pearson Correlation',
        'hypothesis': 'Remote ratio correlates with salary',
        'statistic': f'r = {corr_coef:.3f}',
        'p_value': p_value_corr,
        'result': 'Significant' if p_value_corr < 0.05 else 'Not Significant'
    })
    
    # Create results table
    test_cards = []
    for result in results:
        color = 'success' if result['result'] == 'Significant' else 'warning'
        test_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(result['test'], className="fw-bold mb-2"),
                    html.P(result['hypothesis'], className="small mb-2"),
                    html.P(result['statistic'], className="mb-2"),
                    html.P(f"p-value: {result['p_value']:.6f}", className="small mb-2"),
                    dbc.Badge(result['result'], color=color)
                ])
            ], className="mb-2 shadow-sm border-0")
        )
    
    return html.Div(test_cards)

def create_normality_tests(df, colors):
    """Create normality test results"""
    salary_data = df['salary_in_usd']
    
    # Shapiro-Wilk test (for smaller samples)
    if len(salary_data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(salary_data)
    else:
        shapiro_stat, shapiro_p = None, None
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(salary_data, 'norm', args=(salary_data.mean(), salary_data.std()))
    
    # Anderson-Darling test
    ad_stat, ad_critical, ad_significance = stats.anderson(salary_data, dist='norm')
    
    tests = []
    
    if shapiro_stat is not None:
        tests.append({
            'test': 'Shapiro-Wilk',
            'statistic': f'{shapiro_stat:.6f}',
            'p_value': shapiro_p,
            'result': 'Normal' if shapiro_p > 0.05 else 'Not Normal'
        })
    
    tests.append({
        'test': 'Kolmogorov-Smirnov',
        'statistic': f'{ks_stat:.6f}',
        'p_value': ks_p,
        'result': 'Normal' if ks_p > 0.05 else 'Not Normal'
    })
    
    tests.append({
        'test': 'Anderson-Darling',
        'statistic': f'{ad_stat:.6f}',
        'p_value': 'See critical values',
        'result': 'Normal' if ad_stat < ad_critical[2] else 'Not Normal'  # 5% significance level
    })
    
    test_cards = []
    for test in tests:
        color = 'success' if test['result'] == 'Normal' else 'danger'
        test_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(test['test'], className="fw-bold mb-2"),
                    html.P(f"Statistic: {test['statistic']}", className="small mb-1"),
                    html.P(f"p-value: {test['p_value']}", className="small mb-2"),
                    dbc.Badge(test['result'], color=color)
                ])
            ], className="mb-2 shadow-sm border-0")
        )
    
    return html.Div(test_cards)

def create_multidimensional_analysis(df, colors):
    """Create multidimensional analysis chart"""
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=df['work_year'],
        y=df['remote_ratio'],
        z=df['salary_in_usd'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['salary_in_usd'],
            colorscale=[[0, colors['light']], [1, colors['primary']]],
            opacity=0.8,
            colorbar=dict(title="Salary (USD)")
        ),
        text=df['job_title'],
        hovertemplate='<b>Year:</b> %{x}<br><b>Remote:</b> %{y}%<br><b>Salary:</b> $%{z:,.0f}<br><b>Job:</b> %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='3D Analysis: Year, Remote Ratio, and Salary',
        scene=dict(
            xaxis_title='Work Year',
            yaxis_title='Remote Ratio (%)',
            zaxis_title='Salary (USD)'
        ),
        height=500,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_categorical_analysis(df, colors):
    """Create categorical variable analysis"""
    # Sunburst chart showing hierarchical relationships
    df_sample = df.copy()
    
    fig = go.Figure(go.Sunburst(
        labels=['All Data'] + 
                [f'{exp} Experience' for exp in df['experience_level'].unique()] +
                [f'{exp}-{size}' for exp in df['experience_level'].unique() 
                 for size in df['company_size'].unique()],
        parents=[''] + 
                 ['All Data'] * len(df['experience_level'].unique()) +
                 [f'{exp} Experience' for exp in df['experience_level'].unique()
                  for size in df['company_size'].unique()],
        values=[len(df)] +
               [len(df[df['experience_level'] == exp]) for exp in df['experience_level'].unique()] +
               [len(df[(df['experience_level'] == exp) & (df['company_size'] == size)])
                for exp in df['experience_level'].unique() 
                for size in df['company_size'].unique()],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Hierarchical Analysis: Experience Level → Company Size",
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_data_quality_assessment(df, colors):
    """Create comprehensive data quality assessment"""
    # Calculate quality metrics
    total_rows = len(df)
    
    quality_metrics = {
        'completeness': {
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (total_rows * len(df.columns))) * 100,
            'complete_rows': total_rows - df.isnull().any(axis=1).sum()
        },
        'uniqueness': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / total_rows) * 100,
            'unique_rows': total_rows - df.duplicated().sum()
        },
        'validity': {
            'negative_salaries': (df['salary_in_usd'] < 0).sum() if 'salary_in_usd' in df.columns else 0,
            'invalid_remote_ratio': ((df['remote_ratio'] < 0) | (df['remote_ratio'] > 100)).sum() if 'remote_ratio' in df.columns else 0,
            'future_years': (df['work_year'] > 2024).sum() if 'work_year' in df.columns else 0
        }
    }
    
    # Create quality dashboard
    cards = []
    
    # Completeness card
    completeness_score = (quality_metrics['completeness']['complete_rows'] / total_rows) * 100
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-check-circle fa-2x mb-3", style={'color': colors['primary']}),
                        html.H4(f"{completeness_score:.1f}%", className="fw-bold"),
                        html.P("Data Completeness", className="mb-2"),
                        html.Small(f"{quality_metrics['completeness']['complete_rows']:,} complete rows", className="text-muted")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0")
        ], md=4, className="mb-3")
    )
    
    # Uniqueness card
    uniqueness_score = (quality_metrics['uniqueness']['unique_rows'] / total_rows) * 100
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-fingerprint fa-2x mb-3", style={'color': colors['secondary']}),
                        html.H4(f"{uniqueness_score:.1f}%", className="fw-bold"),
                        html.P("Data Uniqueness", className="mb-2"),
                        html.Small(f"{quality_metrics['uniqueness']['duplicate_rows']} duplicates found", className="text-muted")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0")
        ], md=4, className="mb-3")
    )
    
    # Validity card
    total_invalid = sum(quality_metrics['validity'].values())
    validity_score = ((total_rows - total_invalid) / total_rows) * 100
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-shield-alt fa-2x mb-3", style={'color': colors['accent']}),
                        html.H4(f"{validity_score:.1f}%", className="fw-bold"),
                        html.P("Data Validity", className="mb-2"),
                        html.Small(f"{total_invalid} invalid values", className="text-muted")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0")
        ], md=4, className="mb-3")
    )
    
    # Overall quality score
    overall_score = (completeness_score + uniqueness_score + validity_score) / 3
    overall_card = dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-medal fa-3x mb-3", style={'color': colors['dark']}),
                html.H2(f"{overall_score:.1f}%", className="fw-bold text-primary"),
                html.H5("Overall Data Quality Score", className="mb-3"),
                dbc.Progress(value=overall_score, color="success" if overall_score >= 90 else "warning" if overall_score >= 70 else "danger"),
                html.P(get_quality_recommendation(overall_score), className="mt-3 small text-muted")
            ], className="text-center")
        ])
    ], className="shadow-sm border-0 mb-3")
    
    return html.Div([
        dbc.Row(cards),
        overall_card
    ])

def get_quality_recommendation(score):
    """Get data quality recommendation based on score"""
    if score >= 95:
        return "Excellent data quality! Dataset is ready for analysis and modeling."
    elif score >= 85:
        return "Good data quality. Minor improvements may enhance analysis accuracy."
    elif score >= 70:
        return "Fair data quality. Consider data cleaning to improve reliability."
    else:
        return "Poor data quality. Significant data cleaning and validation required."

def create_advanced_insights_section(df, colors):
    """Create advanced statistical insights"""
    insights = []
    
    # Salary distribution insights
    salary_data = df['salary_in_usd']
    skewness = salary_data.skew()
    kurtosis = salary_data.kurtosis()
    
    insights.append({
        'title': 'Distribution Shape Analysis',
        'content': f'Salary distribution shows skewness of {skewness:.2f} ({"right" if skewness > 0 else "left"} skewed) and kurtosis of {kurtosis:.2f} ({"heavy" if kurtosis > 0 else "light"} tails).',
        'icon': '📊',
        'color': colors['primary']
    })
    
    # Correlation insights
    if 'remote_ratio' in df.columns:
        corr_remote_salary = df['remote_ratio'].corr(df['salary_in_usd'])
        insights.append({
            'title': 'Remote Work Impact',
            'content': f'Remote work ratio shows {"positive" if corr_remote_salary > 0 else "negative"} correlation ({corr_remote_salary:.3f}) with salary, suggesting {"higher" if corr_remote_salary > 0 else "lower"} pay for remote positions.',
            'icon': '🏠',
            'color': colors['secondary']
        })
    
    # Experience level impact
    exp_salary_std = df.groupby('experience_level')['salary_in_usd'].std()
    most_variable_exp = exp_salary_std.idxmax()
    
    insights.append({
        'title': 'Salary Variability by Experience',
        'content': f'{most_variable_exp} level shows highest salary variability (σ = ${exp_salary_std[most_variable_exp]:,.0f}), indicating diverse compensation packages.',
        'icon': '📈',
        'color': colors['accent']
    })
    
    # Geographic insights
    if 'company_location' in df.columns:
        country_counts = df['company_location'].value_counts()
        top_3_countries = country_counts.head(3).index.tolist()
        concentration = country_counts.head(3).sum() / len(df) * 100
        
        insights.append({
            'title': 'Geographic Concentration',
            'content': f'Top 3 countries ({", ".join(top_3_countries)}) represent {concentration:.1f}% of all positions, showing high geographic concentration.',
            'icon': '🌍',
            'color': colors['dark']
        })
    
    # Create insight cards
    insight_cards = []
    for insight in insights:
        insight_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span(insight['icon'], className="fs-2 me-3"),
                            html.Div([
                                html.H6(insight['title'], className="fw-bold mb-2"),
                                html.P(insight['content'], className="mb-0 small")
                            ])
                        ], className="d-flex align-items-start")
                    ])
                ], className="h-100 shadow-sm border-0",
                   style={'borderLeft': f'4px solid {insight["color"]}'})
            ], md=6, className="mb-3")
        )
    
    return dbc.Row(insight_cards)