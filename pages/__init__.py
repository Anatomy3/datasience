"""
Pages module initialization
All page layouts for DS Salaries Dashboard
"""

# Import all page modules for easy access
from . import (
    home,
    data_overview, 
    eda,
    data_cleaning,
    modeling,
    results,
    insights,
    download,
    about
)

__all__ = [
    'home',
    'data_overview',
    'eda', 
    'data_cleaning',
    'modeling',
    'results',
    'insights',
    'download',
    'about'
]