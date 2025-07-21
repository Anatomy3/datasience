"""
Helper Functions Library
Utility functions for data processing, formatting, and common operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import base64
import io
import json
from typing import Dict, List, Any, Union, Optional
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

# Color scheme
COLORS = {
    'primary': '#10b981',
    'secondary': '#34d399',
    'accent': '#6ee7b7',
    'success': '#059669',
    'info': '#06b6d4',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'light': '#f0fdf4',
    'white': '#ffffff',
    'text': '#1f2937',
    'gray': '#6b7280'
}

# =============================================================================
# DATA PROCESSING HELPERS
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division by zero
    
    Returns:
        float: Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value (float): Original value
        new_value (float): New value
    
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return 0 if new_value == 0 else float('inf')
    
    return ((new_value - old_value) / old_value) * 100

def clean_numeric_column(series: pd.Series, fill_method: str = 'median') -> pd.Series:
    """
    Clean numeric column by handling missing values and outliers
    
    Args:
        series (pd.Series): Numeric series to clean
        fill_method (str): Method to fill missing values ('median', 'mean', 'mode', 'zero')
    
    Returns:
        pd.Series: Cleaned series
    """
    cleaned = series.copy()
    
    # Handle missing values
    if fill_method == 'median':
        cleaned = cleaned.fillna(cleaned.median())
    elif fill_method == 'mean':
        cleaned = cleaned.fillna(cleaned.mean())
    elif fill_method == 'mode':
        mode_val = cleaned.mode()
        cleaned = cleaned.fillna(mode_val[0] if len(mode_val) > 0 else 0)
    elif fill_method == 'zero':
        cleaned = cleaned.fillna(0)
    
    return cleaned

def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        series (pd.Series): Numeric series
        factor (float): IQR multiplier factor
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names (lowercase, replace spaces with underscores)
    
    Args:
        df (pd.DataFrame): DataFrame to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    df_copy = df.copy()
    df_copy.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '_', col.lower().strip().replace(' ', '_'))
        for col in df_copy.columns
    ]
    return df_copy

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data summary
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
    
    Returns:
        Dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'total_records': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
        'duplicates': df.duplicated().sum(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    
    # Add numeric summaries
    numeric_cols = summary['numeric_columns']
    if numeric_cols:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Add categorical summaries
    categorical_cols = summary['categorical_columns']
    if categorical_cols:
        summary['categorical_summary'] = {
            col: df[col].value_counts().head(10).to_dict()
            for col in categorical_cols
        }
    
    return summary

# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def format_currency(value: float, currency: str = 'USD', precision: int = 0) -> str:
    """
    Format number as currency
    
    Args:
        value (float): Numeric value
        currency (str): Currency code
        precision (int): Decimal places
    
    Returns:
        str: Formatted currency string
    """
    if pd.isna(value):
        return 'N/A'
    
    if currency == 'USD':
        symbol = '$'
    elif currency == 'EUR':
        symbol = '€'
    elif currency == 'GBP':
        symbol = '£'
    else:
        symbol = currency + ' '
    
    if abs(value) >= 1_000_000:
        return f"{symbol}{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{symbol}{value/1_000:.1f}K"
    else:
        return f"{symbol}{value:,.{precision}f}"

def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format number as percentage
    
    Args:
        value (float): Numeric value (0-100 range)
        precision (int): Decimal places
    
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(value):
        return 'N/A'
    
    return f"{value:.{precision}f}%"

def format_number(value: float, precision: int = 0, unit: str = '') -> str:
    """
    Format number with thousand separators and units
    
    Args:
        value (float): Numeric value
        precision (int): Decimal places
        unit (str): Unit suffix
    
    Returns:
        str: Formatted number string
    """
    if pd.isna(value):
        return 'N/A'
    
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B{unit}"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M{unit}"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K{unit}"
    else:
        return f"{value:,.{precision}f}{unit}"

def format_date(date_value: Union[str, datetime], format_str: str = '%Y-%m-%d') -> str:
    """
    Format date consistently
    
    Args:
        date_value: Date value to format
        format_str (str): Format string
    
    Returns:
        str: Formatted date string
    """
    if pd.isna(date_value):
        return 'N/A'
    
    try:
        if isinstance(date_value, str):
            date_obj = pd.to_datetime(date_value)
        else:
            date_obj = date_value
        
        return date_obj.strftime(format_str)
    except:
        return str(date_value)

def truncate_text(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """
    Truncate text to specified length
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix for truncated text
    
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): Required column names
    
    Returns:
        Dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for completely empty columns
    empty_columns = [col for col in df.columns if df[col].isnull().all()]
    if empty_columns:
        validation_results['warnings'].append(f'Completely empty columns: {empty_columns}')
    
    # Check for high missing value percentage
    high_missing = [
        col for col in df.columns 
        if df[col].isnull().sum() / len(df) > 0.5
    ]
    if high_missing:
        validation_results['warnings'].append(f'Columns with >50% missing values: {high_missing}')
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['warnings'].append(f'Found {duplicate_count} duplicate rows')
    
    # Check data types
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        validation_results['info'].append(f'Object columns (may need encoding): {object_columns}')
    
    return validation_results

def is_numeric_column(series: pd.Series, threshold: float = 0.8) -> bool:
    """
    Check if a column can be converted to numeric
    
    Args:
        series (pd.Series): Series to check
        threshold (float): Minimum proportion of convertible values
    
    Returns:
        bool: True if column can be numeric
    """
    try:
        converted = pd.to_numeric(series, errors='coerce')
        valid_ratio = converted.notna().sum() / len(series)
        return valid_ratio >= threshold
    except:
        return False

# =============================================================================
# CHART HELPERS
# =============================================================================

def create_trend_indicator(current_value: float, previous_value: float) -> html.Div:
    """
    Create trend indicator with icon and color
    
    Args:
        current_value (float): Current period value
        previous_value (float): Previous period value
    
    Returns:
        html.Div: Trend indicator component
    """
    if previous_value == 0:
        change_pct = 0
    else:
        change_pct = ((current_value - previous_value) / previous_value) * 100
    
    if change_pct > 0:
        icon = 'fas fa-arrow-up'
        color = COLORS['success']
        text = f'+{change_pct:.1f}%'
    elif change_pct < 0:
        icon = 'fas fa-arrow-down'
        color = COLORS['danger']
        text = f'{change_pct:.1f}%'
    else:
        icon = 'fas fa-minus'
        color = COLORS['gray']
        text = '0%'
    
    return html.Div([
        html.I(className=icon, style={'color': color, 'marginRight': '4px'}),
        html.Span(text, style={'color': color, 'fontWeight': '600'})
    ], className="d-flex align-items-center")

def generate_color_palette(n_colors: int, base_color: str = COLORS['primary']) -> List[str]:
    """
    Generate color palette based on base color
    
    Args:
        n_colors (int): Number of colors needed
        base_color (str): Base color hex code
    
    Returns:
        List[str]: List of hex color codes
    """
    if n_colors <= len(COLORS):
        return list(COLORS.values())[:n_colors]
    
    # Generate variations of the base color
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
             COLORS['info'], COLORS['success'], COLORS['warning']]
    
    while len(colors) < n_colors:
        colors.extend(colors)
    
    return colors[:n_colors]

# =============================================================================
# FILE HELPERS
# =============================================================================

def export_dataframe_to_excel(df: pd.DataFrame, filename: str = None) -> bytes:
    """
    Export DataFrame to Excel bytes
    
    Args:
        df (pd.DataFrame): DataFrame to export
        filename (str): Optional filename
    
    Returns:
        bytes: Excel file content
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # Add summary sheet
        summary_df = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Data Types'],
            'Value': [
                len(df),
                len(df.columns),
                df.isnull().sum().sum(),
                ', '.join(df.dtypes.unique().astype(str))
            ]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

def create_download_link(content: bytes, filename: str, label: str = "Download") -> html.A:
    """
    Create download link for file content
    
    Args:
        content (bytes): File content
        filename (str): Download filename
        label (str): Link label
    
    Returns:
        html.A: Download link component
    """
    b64_content = base64.b64encode(content).decode()
    href = f"data:application/octet-stream;base64,{b64_content}"
    
    return html.A(
        dbc.Button([
            html.I(className="fas fa-download me-2"),
            label
        ], color="primary", size="sm"),
        href=href,
        download=filename,
        style={'textDecoration': 'none'}
    )

# =============================================================================
# UI HELPERS
# =============================================================================

def create_loading_spinner(text: str = "Loading...") -> html.Div:
    """
    Create loading spinner component
    
    Args:
        text (str): Loading text
    
    Returns:
        html.Div: Loading spinner component
    """
    return html.Div([
        dbc.Spinner(
            color="primary",
            size="lg",
            spinner_style={'width': '3rem', 'height': '3rem'}
        ),
        html.P(text, className="mt-3 text-muted text-center")
    ], className="text-center p-5")

def create_alert_message(message: str, type: str = "info", dismissible: bool = True) -> dbc.Alert:
    """
    Create alert message component
    
    Args:
        message (str): Alert message
        type (str): Alert type (success, warning, danger, info)
        dismissible (bool): Whether alert can be dismissed
    
    Returns:
        dbc.Alert: Alert component
    """
    icon_map = {
        'success': 'fas fa-check-circle',
        'warning': 'fas fa-exclamation-triangle',
        'danger': 'fas fa-times-circle',
        'info': 'fas fa-info-circle'
    }
    
    return dbc.Alert([
        html.I(className=f"{icon_map.get(type, 'fas fa-info-circle')} me-2"),
        message
    ], color=type, dismissable=dismissible)

def create_tooltip_text(text: str, tooltip: str) -> html.Span:
    """
    Create text with tooltip
    
    Args:
        text (str): Display text
        tooltip (str): Tooltip text
    
    Returns:
        html.Span: Text with tooltip
    """
    return html.Span([
        text,
        html.I(
            className="fas fa-question-circle ms-1",
            style={'color': COLORS['gray'], 'fontSize': '0.8rem'},
            id=f"tooltip-{hash(text)}"
        ),
        dbc.Tooltip(tooltip, target=f"tooltip-{hash(text)}")
    ])

# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def calculate_confidence_interval(data: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for data
    
    Args:
        data (pd.Series): Numeric data
        confidence (float): Confidence level (0-1)
    
    Returns:
        Dict: Confidence interval bounds
    """
    from scipy import stats
    
    mean = data.mean()
    std_err = stats.sem(data.dropna())
    interval = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return {
        'mean': mean,
        'lower_bound': mean - interval,
        'upper_bound': mean + interval,
        'margin_of_error': interval
    }

def perform_basic_statistics(series: pd.Series) -> Dict[str, float]:
    """
    Calculate basic statistics for a series
    
    Args:
        series (pd.Series): Numeric series
    
    Returns:
        Dict: Basic statistics
    """
    stats = {
        'count': series.count(),
        'mean': series.mean(),
        'median': series.median(),
        'mode': series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
        'std': series.std(),
        'var': series.var(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    
    return stats

# Export all functions
__all__ = [
    # Data processing
    'safe_divide', 'calculate_percentage_change', 'clean_numeric_column',
    'detect_outliers_iqr', 'standardize_column_names', 'get_data_summary',
    
    # Formatting
    'format_currency', 'format_percentage', 'format_number', 'format_date', 'truncate_text',
    
    # Validation
    'validate_dataframe', 'is_numeric_column',
    
    # Chart helpers
    'create_trend_indicator', 'generate_color_palette',
    
    # File helpers
    'export_dataframe_to_excel', 'create_download_link',
    
    # UI helpers
    'create_loading_spinner', 'create_alert_message', 'create_tooltip_text',
    
    # Statistical helpers
    'calculate_confidence_interval', 'perform_basic_statistics',
    
    # Constants
    'COLORS'
]