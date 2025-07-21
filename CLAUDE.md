# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Data Science Salaries Dashboard** built with Python Dash. It's a comprehensive web application for analyzing data science salary trends globally, featuring interactive visualizations, machine learning predictions, and extensive data analysis capabilities.

## Key Technologies & Dependencies

- **Framework**: Dash (Python web app framework)
- **UI Components**: Dash Bootstrap Components (dbc)
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, seaborn, matplotlib
- **Machine Learning**: scikit-learn, joblib
- **File Handling**: openpyxl for Excel exports

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Quick setup (runs setup script)
python setup.py
```

### Running the Application
```bash
# Start the dashboard server
python app.py

# Default URL: http://127.0.0.1:8050
```

### Data Requirements
- Place `ds_salaries.csv` in the `data/` folder
- Optionally place `model.pkl` (trained ML model) in the `data/` folder
- If files are missing, the app creates sample fallback data

## Architecture Overview

### File Structure
```
├── app.py                  # Main application entry point
├── setup.py               # Project setup and dependency checker
├── requirements.txt       # Python dependencies
├── data/                  # Data files (CSV, model.pkl)
├── assets/                # Static assets (CSS, images)
├── components/            # Reusable UI components
│   ├── cards.py          # Card components
│   ├── charts.py         # Chart components
│   ├── footer.py         # Footer component
│   ├── helpers.py        # Utility functions
│   └── navbar.py         # Navigation component
├── pages/                 # Page layouts
│   ├── home.py           # Homepage with integrated prediction
│   ├── data_overview.py  # Dataset overview and statistics
│   ├── eda.py            # Exploratory data analysis
│   ├── data_cleaning.py  # Data cleaning workflows
│   ├── modeling.py       # ML model training/evaluation
│   ├── prediction.py     # Salary prediction interface
│   ├── results.py        # Analysis results
│   ├── insights.py       # Key insights and findings
│   ├── download.py       # Export functionality
│   ├── about.py          # About page
│   └── bantuan.py        # Help/support page
└── utils/                 # Utility modules
    ├── data_loader.py    # Data loading and management
    ├── data_processor.py # Data processing utilities
    └── helpers.py        # General helper functions
```

### Core Application Flow

1. **app.py**: Main application file that:
   - Initializes Dash app with Bootstrap themes
   - Imports page modules with fallback mechanisms
   - Handles routing between pages
   - Contains integrated prediction functionality on homepage
   - Manages global state and callbacks

2. **Data Layer** (`utils/data_loader.py`):
   - `DataLoader` class for centralized data management
   - Handles CSV loading with fallback to sample data
   - Model loading with error handling
   - Data quality reporting and filtering

3. **Component Layer** (`components/`):
   - Modular UI components for reuse across pages
   - `helpers.py` contains extensive utility functions for data processing, formatting, validation, and UI components
   - Color scheme and styling constants

4. **Page Layer** (`pages/`):
   - Each page is a separate module with its own layout function
   - Home page includes integrated salary prediction form
   - Follows consistent structure: layout(df, colors) function

### Key Features Architecture

#### 1. Navigation System
- Horizontal navbar with dropdown menus
- Multi-level navigation structure organized by function:
  - Data (Overview, Cleaning)
  - Analysis (EDA, Modeling, Results)
  - Tools (Prediction, Download)
  - Info (Insights, About)

#### 2. Prediction System
- **Integrated on Homepage**: Quick prediction form with real-time results
- **Dedicated Prediction Page**: More detailed prediction interface
- Model handling with graceful fallbacks when model.pkl is missing
- Input validation and error handling

#### 3. Data Processing Pipeline
- Standardized data loading through `DataLoader` class
- Data validation and quality reporting
- Outlier detection using IQR method
- Missing value handling with multiple strategies

#### 4. Visualization System
- Consistent color scheme defined in COLORS dictionary
- Responsive charts using plotly
- Chart helper functions for common visualizations
- Interactive elements with hover templates

## Development Patterns

### Adding New Pages
1. Create new file in `pages/` directory
2. Implement `layout(df, colors)` function
3. Add import and routing in `app.py`
4. Add navigation link in navbar component

### Working with Data
- Use `utils.data_loader.load_data()` for quick data access
- Use `DataLoader` class for advanced data operations
- Apply data validation using `validate_dataframe()` helper
- Use formatting helpers from `components.helpers` for consistent display

### UI Development
- Follow Bootstrap components pattern using dbc
- Use predefined color scheme from COLORS dictionary
- Leverage helper functions for common UI patterns
- Ensure responsive design with Bootstrap grid system

### Machine Learning Integration
- Models should be saved as `data/model.pkl` using joblib
- Use try/catch blocks for model loading
- Provide fallback functionality when models are unavailable
- Input features should match training data structure

## Error Handling Strategy

The application implements graceful degradation:
- Missing data files → fallback to sample data
- Missing model files → demo predictions or disabled features
- Missing page modules → fallback layouts with construction messages
- Import errors → try/catch blocks with informative messages

## Styling and Theming

- **CSS Framework**: Bootstrap 5
- **Color Scheme**: Modern emerald green theme
- **Typography**: Inter font family
- **Custom CSS**: Embedded in app.index_string with hover effects and animations
- **Icons**: Font Awesome 6.4.0

## Performance Considerations

- Large datasets should be loaded once and cached
- Use `prevent_initial_call=True` for callbacks that shouldn't run on page load
- Consider pagination for large data displays
- Optimize plotly figures with appropriate config settings