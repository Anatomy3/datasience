"""
Navbar Component
Modern navigation bar with responsive design
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

# Color scheme
COLORS = {
    'primary': '#10b981',
    'secondary': '#34d399',
    'accent': '#6ee7b7',
    'success': '#059669',
    'info': '#06b6d4',
    'warning': '#f59e0b',
    'light': '#f0fdf4',
    'white': '#ffffff',
    'text': '#1f2937',
    'gray': '#6b7280',
    'gradient': 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)'
}

def create_navbar(dataset_info=None):
    """
    Create modern navigation bar
    
    Args:
        dataset_info (dict): Information about the dataset (records count, etc.)
    
    Returns:
        dbc.Navbar: Navigation bar component
    """
    
    # Default dataset info if none provided
    if dataset_info is None:
        dataset_info = {
            'total_records': 'N/A',
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
    
    return dbc.Navbar([
        dbc.Container([
            # Brand section with logo and title
            dbc.Row([
                dbc.Col([
                    dbc.NavbarBrand([
                        # Logo/Icon
                        html.I(
                            className="fas fa-chart-line me-3", 
                            style={
                                'color': COLORS['white'], 
                                'fontSize': '28px',
                                'filter': 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))'
                            }
                        ),
                        # Brand text with gradient effect
                        html.Span(
                            "DS Salaries", 
                            className="fw-bold", 
                            style={
                                'fontSize': '24px', 
                                'letterSpacing': '-0.5px',
                                'textShadow': '0 2px 4px rgba(0,0,0,0.2)'
                            }
                        )
                    ], 
                    href="/", 
                    className="d-flex align-items-center",
                    style={'textDecoration': 'none'})
                ], width="auto"),
            ], align="center", className="g-0 w-100 justify-content-between"),
            
            # Right side information section
            dbc.Row([
                dbc.Col([
                    # Dataset info badge
                    dbc.Badge([
                        html.I(className="fas fa-database me-2"),
                        f"{dataset_info['total_records']} records"
                    ], 
                    color="light", 
                    text_color="dark", 
                    className="me-3 px-3 py-2",
                    style={
                        'borderRadius': '12px',
                        'fontWeight': '500',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    }),
                    
                    # Last updated badge
                    dbc.Badge([
                        html.I(className="fas fa-calendar-alt me-2"), 
                        dataset_info['last_updated']
                    ], 
                    color="light", 
                    text_color="dark", 
                    className="px-3 py-2",
                    style={
                        'borderRadius': '12px',
                        'fontWeight': '500',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    })
                ], width="auto")
            ], align="center")
        ], fluid=True)
    ],
    style={
        'background': COLORS['gradient'],
        'boxShadow': '0 4px 20px rgba(16, 185, 129, 0.15)',
        'borderBottom': f'3px solid {COLORS["accent"]}',
        'backdropFilter': 'blur(10px)',
        'position': 'sticky',
        'top': '0',
        'zIndex': '1030'
    },
    dark=True,
    className="mb-0")

def create_mobile_navbar_toggle():
    """Create mobile navigation toggle button"""
    return dbc.Button([
        html.I(className="fas fa-bars")
    ],
    id="mobile-nav-toggle",
    color="light",
    outline=True,
    className="d-md-none",
    style={
        'borderRadius': '8px',
        'border': '2px solid rgba(255,255,255,0.3)'
    })

def create_breadcrumb(current_page, breadcrumb_items=None):
    """
    Create breadcrumb navigation
    
    Args:
        current_page (str): Current page name
        breadcrumb_items (list): List of breadcrumb items
    
    Returns:
        dbc.Breadcrumb: Breadcrumb component
    """
    
    if breadcrumb_items is None:
        breadcrumb_items = [
            {"label": "Home", "href": "/", "active": False},
            {"label": current_page, "active": True}
        ]
    
    items = []
    for item in breadcrumb_items:
        if item.get("active", False):
            items.append(dbc.BreadcrumbItem(item["label"], active=True))
        else:
            items.append(dbc.BreadcrumbItem(
                item["label"], 
                href=item["href"],
                style={'color': COLORS['primary']}
            ))
    
    return html.Div([
        dbc.Breadcrumb(
            items,
            style={
                'backgroundColor': 'transparent',
                'padding': '0.5rem 0',
                'marginBottom': '1rem'
            }
        )
    ])

def create_page_header(title, description, icon=None, actions=None):
    """
    Create consistent page header
    
    Args:
        title (str): Page title
        description (str): Page description
        icon (str): FontAwesome icon class
        actions (list): List of action buttons
    
    Returns:
        html.Div: Page header component
    """
    
    header_content = []
    
    # Title with optional icon
    title_elements = []
    if icon:
        title_elements.append(
            html.I(className=f"{icon} me-3", style={'color': COLORS['primary']})
        )
    title_elements.append(title)
    
    header_content.append(
        html.H1(
            title_elements,
            className="mb-2",
            style={
                'color': COLORS['text'],
                'fontWeight': '700',
                'letterSpacing': '-0.5px'
            }
        )
    )
    
    # Description
    if description:
        header_content.append(
            html.P(
                description,
                className="lead mb-4",
                style={
                    'color': COLORS['gray'],
                    'fontSize': '1.15rem',
                    'lineHeight': '1.6'
                }
            )
        )
    
    # Actions
    if actions:
        header_content.append(
            html.Div(
                actions,
                className="d-flex gap-2 mb-4"
            )
        )
    
    return html.Div(
        header_content,
        className="mb-4"
    )

def create_notification_toast(message, type="info", duration=5000):
    """
    Create notification toast
    
    Args:
        message (str): Toast message
        type (str): Toast type (success, warning, danger, info)
        duration (int): Auto-hide duration in milliseconds
    
    Returns:
        dbc.Toast: Toast component
    """
    
    icon_map = {
        'success': 'fas fa-check-circle',
        'warning': 'fas fa-exclamation-triangle',
        'danger': 'fas fa-times-circle',
        'info': 'fas fa-info-circle'
    }
    
    color_map = {
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'danger': COLORS['danger'],
        'info': COLORS['info']
    }
    
    return dbc.Toast([
        html.Div([
            html.I(
                className=icon_map.get(type, 'fas fa-info-circle'),
                style={
                    'color': color_map.get(type, COLORS['info']),
                    'fontSize': '1.2rem',
                    'marginRight': '0.5rem'
                }
            ),
            html.Span(message, style={'color': COLORS['text']})
        ], className="d-flex align-items-center")
    ],
    id="notification-toast",
    header="Notification",
    dismissable=True,
    duration=duration,
    style={
        'position': 'fixed',
        'top': '100px',
        'right': '20px',
        'zIndex': '9999',
        'minWidth': '300px',
        'border': f'1px solid {color_map.get(type, COLORS["info"])}',
        'borderRadius': '12px',
        'boxShadow': '0 8px 25px rgba(0,0,0,0.15)'
    })

def create_search_bar(placeholder="Search...", search_id="global-search"):
    """
    Create global search bar
    
    Args:
        placeholder (str): Search placeholder text
        search_id (str): HTML ID for the search input
    
    Returns:
        html.Div: Search bar component
    """
    
    return html.Div([
        dbc.InputGroup([
            dbc.Input(
                id=search_id,
                placeholder=placeholder,
                type="search",
                style={
                    'border': f'2px solid {COLORS["accent"]}',
                    'borderRadius': '12px 0 0 12px',
                    'boxShadow': 'none'
                }
            ),
            dbc.Button([
                html.I(className="fas fa-search")
            ],
            color="primary",
            style={
                'borderRadius': '0 12px 12px 0',
                'background': COLORS['gradient'],
                'border': 'none'
            })
        ])
    ], className="mb-3")

def create_quick_stats_bar(stats):
    """
    Create quick statistics bar
    
    Args:
        stats (list): List of stat dictionaries with 'label', 'value', 'icon'
    
    Returns:
        html.Div: Quick stats component
    """
    
    stat_items = []
    for stat in stats:
        stat_items.append(
            html.Div([
                html.I(
                    className=stat.get('icon', 'fas fa-chart-bar'),
                    style={
                        'color': COLORS['primary'],
                        'fontSize': '1.2rem',
                        'marginRight': '0.5rem'
                    }
                ),
                html.Span([
                    html.Strong(stat['value'], style={'color': COLORS['text']}),
                    html.Span(f" {stat['label']}", style={'color': COLORS['gray']})
                ])
            ],
            className="d-flex align-items-center me-4"
            )
        )
    
    return html.Div([
        html.Div(
            stat_items,
            className="d-flex flex-wrap align-items-center p-3"
        )
    ],
    style={
        'backgroundColor': COLORS['light'],
        'borderRadius': '12px',
        'border': f'1px solid {COLORS["accent"]}',
        'marginBottom': '1rem'
    })

def create_action_toolbar(actions):
    """
    Create action toolbar with buttons
    
    Args:
        actions (list): List of action dictionaries
    
    Returns:
        html.Div: Action toolbar component
    """
    
    action_buttons = []
    for action in actions:
        button_props = {
            'children': [
                html.I(className=f"{action.get('icon', 'fas fa-cog')} me-2"),
                action['label']
            ],
            'color': action.get('color', 'primary'),
            'size': action.get('size', 'sm'),
            'className': 'me-2',
            'style': {
                'borderRadius': '8px',
                'fontWeight': '500'
            }
        }
        
        if action.get('id'):
            button_props['id'] = action['id']
        if action.get('href'):
            button_props['href'] = action['href']
        if action.get('disabled'):
            button_props['disabled'] = action['disabled']
        
        action_buttons.append(dbc.Button(**button_props))
    
    return html.Div([
        html.Div(
            action_buttons,
            className="d-flex flex-wrap align-items-center"
        )
    ],
    className="mb-3")

# Export all functions for easy import
__all__ = [
    'create_navbar',
    'create_mobile_navbar_toggle',
    'create_breadcrumb',
    'create_page_header',
    'create_notification_toast',
    'create_search_bar',
    'create_quick_stats_bar',
    'create_action_toolbar',
    'COLORS'
]