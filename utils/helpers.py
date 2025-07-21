"""
Helper Functions
Utility functions for formatting, calculations, and common operations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

def format_currency(amount, currency='USD'):
    """Format number as currency"""
    if pd.isna(amount):
        return 'N/A'
    
    if currency == 'USD':
        return f"${amount:,.0f}"
    else:
        return f"{amount:,.0f} {currency}"

def format_number(number, decimal_places=0):
    """Format number with thousand separators"""
    if pd.isna(number):
        return 'N/A'
    
    if decimal_places == 0:
        return f"{number:,.0f}"
    else:
        return f"{number:,.{decimal_places}f}"

def format_percentage(value, decimal_places=1):
    """Format value as percentage"""
    if pd.isna(value):
        return 'N/A'
    
    return f"{value:.{decimal_places}f}%"

def calculate_percentile(df, column, value):
    """Calculate percentile of a value in a column"""
    if column not in df.columns:
        return None
    
    percentile = (df[column] <= value).mean() * 100
    return percentile

def get_experience_label(exp_code):
    """Get human-readable experience level label"""
    mapping = {
        'EN': 'Entry Level',
        'MI': 'Mid Level', 
        'SE': 'Senior Level',
        'EX': 'Executive Level'
    }
    return mapping.get(exp_code, exp_code)

def get_company_size_label(size_code):
    """Get human-readable company size label"""
    mapping = {
        'S': 'Small',
        'M': 'Medium',
        'L': 'Large'
    }
    return mapping.get(size_code, size_code)

def get_employment_label(emp_code):
    """Get human-readable employment type label"""
    mapping = {
        'FT': 'Full Time',
        'PT': 'Part Time',
        'CT': 'Contract',
        'FL': 'Freelance'
    }
    return mapping.get(emp_code, emp_code)

def get_remote_label(remote_ratio):
    """Get human-readable remote work label"""
    if remote_ratio == 100:
        return 'Fully Remote'
    elif remote_ratio == 50:
        return 'Hybrid'
    elif remote_ratio == 0:
        return 'On-site'
    else:
        return f'{remote_ratio}% Remote'

def create_summary_stats(df, column):
    """Create summary statistics for a column"""
    if column not in df.columns:
        return None
    
    series = df[column]
    
    if series.dtype in ['int64', 'float64']:
        # Numerical column
        stats = {
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'mode': series.mode().iloc[0] if not series.mode().empty else None,
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100
        }
    else:
        # Categorical column
        value_counts = series.value_counts()
        stats = {
            'count': len(series),
            'unique_count': series.nunique(),
            'most_frequent': value_counts.index[0] if not value_counts.empty else None,
            'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'least_frequent': value_counts.index[-1] if not value_counts.empty else None,
            'least_frequent_count': value_counts.iloc[-1] if not value_counts.empty else 0,
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'value_counts': value_counts.to_dict()
        }
    
    return stats

def calculate_salary_insights(df):
    """Calculate comprehensive salary insights"""
    if 'salary_in_usd' not in df.columns:
        return None
    
    salary_col = df['salary_in_usd']
    
    insights = {
        'basic_stats': create_summary_stats(df, 'salary_in_usd'),
        'percentiles': {
            'p10': salary_col.quantile(0.10),
            'p25': salary_col.quantile(0.25),
            'p50': salary_col.quantile(0.50),
            'p75': salary_col.quantile(0.75),
            'p90': salary_col.quantile(0.90),
            'p95': salary_col.quantile(0.95),
            'p99': salary_col.quantile(0.99)
        }
    }
    
    # Experience level insights
    if 'experience_level' in df.columns:
        exp_insights = {}
        for exp in df['experience_level'].unique():
            exp_data = df[df['experience_level'] == exp]['salary_in_usd']
            exp_insights[exp] = {
                'mean': exp_data.mean(),
                'median': exp_data.median(),
                'count': len(exp_data),
                'min': exp_data.min(),
                'max': exp_data.max()
            }
        insights['by_experience'] = exp_insights
    
    # Company size insights
    if 'company_size' in df.columns:
        size_insights = {}
        for size in df['company_size'].unique():
            size_data = df[df['company_size'] == size]['salary_in_usd']
            size_insights[size] = {
                'mean': size_data.mean(),
                'median': size_data.median(),
                'count': len(size_data),
                'min': size_data.min(),
                'max': size_data.max()
            }
        insights['by_company_size'] = size_insights
    
    # Remote work insights
    if 'remote_ratio' in df.columns:
        remote_insights = {}
        for ratio in df['remote_ratio'].unique():
            remote_data = df[df['remote_ratio'] == ratio]['salary_in_usd']
            remote_insights[ratio] = {
                'mean': remote_data.mean(),
                'median': remote_data.median(),
                'count': len(remote_data),
                'min': remote_data.min(),
                'max': remote_data.max()
            }
        insights['by_remote_ratio'] = remote_insights
    
    return insights

def export_to_csv(df, filename=None):
    """Export DataFrame to CSV format"""
    if filename is None:
        filename = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Create CSV string
    csv_string = df.to_csv(index=False)
    
    # Encode for download
    csv_bytes = csv_string.encode('utf-8')
    csv_b64 = base64.b64encode(csv_bytes).decode()
    
    return {
        'content': csv_b64,
        'filename': filename,
        'type': 'text/csv',
        'size': len(csv_bytes)
    }

def export_to_excel(df, filename=None, sheet_name='Data'):
    """Export DataFrame to Excel format"""
    if filename is None:
        filename = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    excel_bytes = output.getvalue()
    excel_b64 = base64.b64encode(excel_bytes).decode()
    
    return {
        'content': excel_b64,
        'filename': filename,
        'type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'size': len(excel_bytes)
    }

def export_to_json(data, filename=None):
    """Export data to JSON format"""
    if filename is None:
        filename = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    if isinstance(data, pd.DataFrame):
        json_string = data.to_json(orient='records', indent=2)
    else:
        import json
        json_string = json.dumps(data, indent=2, default=str)
    
    json_bytes = json_string.encode('utf-8')
    json_b64 = base64.b64encode(json_bytes).decode()
    
    return {
        'content': json_b64,
        'filename': filename,
        'type': 'application/json',
        'size': len(json_bytes)
    }

def create_download_link(export_data):
    """Create download link for exported data"""
    return f"data:{export_data['type']};base64,{export_data['content']}"

def validate_data_quality(df):
    """Validate data quality and return report"""
    report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'data_types': df.dtypes.to_dict(),
        'missing_values': {},
        'duplicate_records': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'quality_score': 0
    }
    
    # Check missing values
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    for col in df.columns:
        report['missing_values'][col] = {
            'count': missing_counts[col],
            'percentage': missing_percentages[col]
        }
    
    # Calculate quality score (0-100)
    completeness_score = (1 - missing_counts.sum() / (len(df) * len(df.columns))) * 100
    uniqueness_score = (1 - report['duplicate_records'] / len(df)) * 100 if len(df) > 0 else 100
    
    report['quality_score'] = (completeness_score + uniqueness_score) / 2
    report['completeness_score'] = completeness_score
    report['uniqueness_score'] = uniqueness_score
    
    # Quality assessment
    if report['quality_score'] >= 90:
        report['quality_level'] = 'Excellent'
    elif report['quality_score'] >= 75:
        report['quality_level'] = 'Good'
    elif report['quality_score'] >= 60:
        report['quality_level'] = 'Fair'
    else:
        report['quality_level'] = 'Poor'
    
    return report

def create_color_palette(base_color='#10b981', num_colors=5):
    """Create color palette based on base color"""
    import colorsys
    
    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    rgb = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert to HSV
    hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    
    # Generate palette
    colors = []
    for i in range(num_colors):
        # Vary saturation and value
        s = max(0.3, hsv[1] - (i * 0.15))
        v = min(1.0, hsv[2] + (i * 0.1))
        
        rgb_new = colorsys.hsv_to_rgb(hsv[0], s, v)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb_new[0] * 255),
            int(rgb_new[1] * 255), 
            int(rgb_new[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def calculate_growth_rate(df, value_column, time_column, periods=None):
    """Calculate growth rate over time"""
    if periods is None:
        periods = sorted(df[time_column].unique())
    
    growth_rates = {}
    
    for i in range(1, len(periods)):
        current_period = periods[i]
        previous_period = periods[i-1]
        
        current_value = df[df[time_column] == current_period][value_column].mean()
        previous_value = df[df[time_column] == previous_period][value_column].mean()
        
        if previous_value != 0:
            growth_rate = ((current_value - previous_value) / previous_value) * 100
            growth_rates[f"{previous_period}_to_{current_period}"] = growth_rate
    
    return growth_rates

def create_benchmark_comparison(df, benchmark_column, value_column, user_value):
    """Create benchmark comparison"""
    if benchmark_column not in df.columns or value_column not in df.columns:
        return None
    
    comparison = {}
    
    for benchmark in df[benchmark_column].unique():
        benchmark_data = df[df[benchmark_column] == benchmark][value_column]
        
        comparison[benchmark] = {
            'mean': benchmark_data.mean(),
            'median': benchmark_data.median(),
            'percentile_25': benchmark_data.quantile(0.25),
            'percentile_75': benchmark_data.quantile(0.75),
            'user_percentile': (benchmark_data <= user_value).mean() * 100,
            'vs_mean_diff': user_value - benchmark_data.mean(),
            'vs_mean_pct': ((user_value - benchmark_data.mean()) / benchmark_data.mean()) * 100 if benchmark_data.mean() != 0 else 0
        }
    
    return comparison

def generate_insights_text(insights_data):
    """Generate human-readable insights text"""
    insights_text = []
    
    if 'salary_growth' in insights_data:
        growth = insights_data['salary_growth']
        if growth > 5:
            insights_text.append(f"Salary trends show strong growth of {growth:.1f}% year-over-year.")
        elif growth > 0:
            insights_text.append(f"Salary trends show modest growth of {growth:.1f}% year-over-year.")
        else:
            insights_text.append("Salary trends show declining compensation patterns.")
    
    if 'remote_premium' in insights_data:
        premium = insights_data['remote_premium']
        if premium > 10:
            insights_text.append(f"Remote work offers significant premium of {premium:.1f}% over on-site positions.")
        elif premium > 0:
            insights_text.append(f"Remote work offers modest premium of {premium:.1f}% over on-site positions.")
        else:
            insights_text.append("Remote work shows no significant salary advantage.")
    
    if 'top_paying_role' in insights_data:
        role = insights_data['top_paying_role']
        insights_text.append(f"Highest paying role is {role['title']} with average salary of {format_currency(role['salary'])}.")
    
    return insights_text

def create_data_story(df):
    """Create data storytelling elements"""
    story_elements = {
        'headline': generate_headline(df),
        'key_stats': extract_key_stats(df),
        'trends': identify_trends(df),
        'insights': generate_story_insights(df),
        'recommendations': generate_recommendations(df)
    }
    
    return story_elements

def generate_headline(df):
    """Generate compelling headline from data"""
    avg_salary = df['salary_in_usd'].mean()
    max_salary = df['salary_in_usd'].max()
    countries = df['company_location'].nunique()
    
    return f"Data Scientists Earn ${avg_salary:,.0f} on Average Across {countries} Countries, with Top Earners Reaching ${max_salary:,.0f}"

def extract_key_stats(df):
    """Extract key statistics for storytelling"""
    return {
        'total_professionals': len(df),
        'average_salary': df['salary_in_usd'].mean(),
        'salary_range': (df['salary_in_usd'].min(), df['salary_in_usd'].max()),
        'countries_covered': df['company_location'].nunique(),
        'job_titles': df['job_title'].nunique(),
        'remote_percentage': (df['remote_ratio'] == 100).mean() * 100
    }

def identify_trends(df):
    """Identify key trends in the data"""
    trends = {}
    
    # Experience level trend
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean()
    trends['experience_multiplier'] = exp_salary.max() / exp_salary.min()
    
    # Remote work trend
    if 'remote_ratio' in df.columns:
        remote_avg = df.groupby('remote_ratio')['salary_in_usd'].mean()
        trends['remote_advantage'] = remote_avg.get(100, 0) / remote_avg.get(0, 1) - 1
    
    # Year-over-year trend
    if 'work_year' in df.columns and df['work_year'].nunique() > 1:
        yearly_avg = df.groupby('work_year')['salary_in_usd'].mean()
        years = sorted(yearly_avg.index)
        if len(years) >= 2:
            trends['yoy_growth'] = (yearly_avg[years[-1]] / yearly_avg[years[-2]] - 1) * 100
    
    return trends

def generate_story_insights(df):
    """Generate insights for data storytelling"""
    insights = []
    
    # Top paying countries
    top_countries = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(3)
    insights.append(f"Top paying countries: {', '.join(top_countries.index[:3])}")
    
    # Experience premium
    exp_salary = df.groupby('experience_level')['salary_in_usd'].mean()
    if 'SE' in exp_salary.index and 'EN' in exp_salary.index:
        multiplier = exp_salary['SE'] / exp_salary['EN']
        insights.append(f"Senior professionals earn {multiplier:.1f}x more than entry level")
    
    # Company size impact
    size_salary = df.groupby('company_size')['salary_in_usd'].mean()
    if len(size_salary) > 1:
        max_size = size_salary.idxmax()
        min_size = size_salary.idxmin()
        difference = size_salary.max() - size_salary.min()
        insights.append(f"Large companies pay ${difference:,.0f} more than small companies on average")
    
    return insights

def generate_recommendations(df):
    """Generate actionable recommendations"""
    recommendations = []
    
    # For job seekers
    recommendations.append({
        'audience': 'Job Seekers',
        'recommendations': [
            'Target senior-level positions for maximum salary potential',
            'Consider remote opportunities for salary premium',
            'Focus on high-demand specializations like ML Engineering',
            'Negotiate based on market data and location factors'
        ]
    })
    
    # For employers
    recommendations.append({
        'audience': 'Employers', 
        'recommendations': [
            'Offer competitive remote work options',
            'Benchmark salaries against market leaders',
            'Invest in employee skill development',
            'Create clear progression paths with transparent compensation'
        ]
    })
    
    return recommendations