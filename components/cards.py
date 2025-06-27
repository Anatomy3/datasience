"""
Cards Component Library
Reusable card components for consistent UI design
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

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
    'gray': '#6b7280',
    'gradient': 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)'
}

def create_metric_card(title, value, subtitle=None, icon=None, color=COLORS['primary'], 
                      trend=None, href=None, size="md"):
    """
    Create metric/statistic card
    
    Args:
        title (str): Card title
        value (str): Main metric value
        subtitle (str): Optional subtitle
        icon (str): FontAwesome icon class
        color (str): Primary color for the card
        trend (dict): Trend info with 'value', 'direction', 'period'
        href (str): Optional link destination
        size (str): Card size ('sm', 'md', 'lg')
    
    Returns:
        html.Div: Metric card component
    """
    
    # Size configurations
    size_config = {
        'sm': {'padding': '1.5rem', 'value_size': '2rem', 'title_size': '0.875rem'},
        'md': {'padding': '2rem', 'value_size': '2.5rem', 'title_size': '1rem'},
        'lg': {'padding': '2.5rem', 'value_size': '3rem', 'title_size': '1.125rem'}
    }
    
    config = size_config.get(size, size_config['md'])
    
    # Icon section
    icon_section = None
    if icon:
        icon_section = html.Div([
            html.I(className=icon, style={
                'fontSize': '2.5rem' if size == 'lg' else '2rem',
                'color': color
            })
        ], className="text-center mb-3")
    
    # Trend indicator
    trend_section = None
    if trend:
        trend_color = COLORS['success'] if trend['direction'] == 'up' else COLORS['danger']
        trend_icon = 'fas fa-arrow-up' if trend['direction'] == 'up' else 'fas fa-arrow-down'
        
        trend_section = html.Div([
            html.I(className=trend_icon, style={'color': trend_color, 'fontSize': '0.8rem'}),
            html.Span(f" {trend['value']} ", style={'color': trend_color, 'fontWeight': '600'}),
            html.Span(f"vs {trend['period']}", style={'color': COLORS['gray'], 'fontSize': '0.8rem'})
        ], className="mb-2")
    
    # Card content
    content = [
        icon_section,
        html.H3(
            value, 
            className="fw-bold text-center mb-1",
            style={'color': COLORS['text'], 'fontSize': config['value_size']}
        ),
        html.H6(
            title, 
            className="text-center mb-2",
            style={'color': COLORS['gray'], 'fontSize': config['title_size']}
        ),
        trend_section,
        html.P(
            subtitle, 
            className="text-center small mb-0",
            style={'color': COLORS['gray']}
        ) if subtitle else None
    ]
    
    # Remove None elements
    content = [item for item in content if item is not None]
    
    card_style = {
        'backgroundColor': COLORS['white'],
        'padding': config['padding'],
        'borderRadius': '16px',
        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.08)',
        'border': f'1px solid {COLORS["light"]}',
        'height': '100%',
        'transition': 'all 0.3s ease',
        'position': 'relative',
        'overflow': 'hidden'
    }
    
    # Add top border for color accent
    card_style['borderTop'] = f'4px solid {color}'
    
    card = html.Div(content, style=card_style, className="hover-card")
    
    # Wrap in link if href provided
    if href:
        return html.A(
            card,
            href=href,
            style={'textDecoration': 'none'},
            className="d-block"
        )
    
    return card

def create_info_card(title, content, icon=None, color=COLORS['primary'], 
                    actions=None, collapsible=False):
    """
    Create information card with optional actions
    
    Args:
        title (str): Card title
        content (str|html): Card content
        icon (str): FontAwesome icon class
        color (str): Accent color
        actions (list): List of action buttons
        collapsible (bool): Whether card is collapsible
    
    Returns:
        html.Div: Information card component
    """
    
    # Header with optional icon
    header_content = []
    if icon:
        header_content.append(
            html.I(className=f"{icon} me-2", style={'color': color})
        )
    header_content.append(title)
    
    # Collapse button if collapsible
    if collapsible:
        header_content.append(
            dbc.Button([
                html.I(className="fas fa-chevron-down")
            ],
            color="light",
            size="sm",
            className="ms-auto border-0 shadow-none",
            id=f"collapse-toggle-{title.lower().replace(' ', '-')}")
        )
    
    header = html.H5(
        header_content,
        className="mb-3 d-flex align-items-center",
        style={'color': COLORS['text']}
    )
    
    # Card body
    body_content = [content]
    
    # Actions section
    if actions:
        action_buttons = []
        for action in actions:
            button_props = {
                'children': [
                    html.I(className=f"{action.get('icon', 'fas fa-cog')} me-2"),
                    action['label']
                ],
                'color': action.get('color', 'primary'),
                'size': action.get('size', 'sm'),
                'className': 'me-2 mb-2'
            }
            
            if action.get('id'):
                button_props['id'] = action['id']
            if action.get('href'):
                button_props['href'] = action['href']
            
            action_buttons.append(dbc.Button(**button_props))
        
        body_content.append(
            html.Div([
                html.Hr(),
                html.Div(action_buttons, className="d-flex flex-wrap")
            ])
        )
    
    # Collapsible wrapper
    if collapsible:
        body_content = [
            dbc.Collapse(
                body_content,
                id=f"collapse-{title.lower().replace(' ', '-')}",
                is_open=True
            )
        ]
    
    return html.Div([
        header,
        html.Div(body_content)
    ], style={
        'backgroundColor': COLORS['white'],
        'padding': '2rem',
        'borderRadius': '16px',
        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.08)',
        'border': f'1px solid {COLORS["light"]}',
        'borderLeft': f'4px solid {color}'
    })

def create_chart_card(title, figure, description=None, actions=None, height="400px"):
    """
    Create card for displaying charts
    
    Args:
        title (str): Chart title
        figure (plotly.graph_objects.Figure): Plotly figure
        description (str): Optional chart description
        actions (list): Optional action buttons
        height (str): Chart height
    
    Returns:
        html.Div: Chart card component
    """
    
    # Card header
    header_content = [
        html.H5(title, className="mb-0", style={'color': COLORS['text']})
    ]
    
    if actions:
        action_buttons = []
        for action in actions:
            action_buttons.append(
                dbc.Button([
                    html.I(className=f"{action.get('icon', 'fas fa-download')} me-1")
                ],
                color="outline-secondary",
                size="sm",
                title=action.get('title', ''),
                id=action.get('id'))
            )
        
        header_content.append(
            html.Div(
                action_buttons,
                className="ms-auto d-flex gap-1"
            )
        )
    
    header = html.Div(
        header_content,
        className="d-flex align-items-center justify-content-between mb-3"
    )
    
    # Chart section
    chart_section = dcc.Graph(
        figure=figure,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        },
        style={'height': height}
    )
    
    # Description section
    description_section = None
    if description:
        description_section = html.P(
            description,
            className="text-muted mt-3 mb-0",
            style={'fontSize': '0.9rem', 'lineHeight': '1.5'}
        )
    
    return html.Div([
        header,
        chart_section,
        description_section
    ], style={
        'backgroundColor': COLORS['white'],
        'padding': '1.5rem',
        'borderRadius': '16px',
        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.08)',
        'border': f'1px solid {COLORS["light"]}'
    })

def create_feature_card(title, description, icon, color=COLORS['primary'], 
                       status="available", link=None):
    """
    Create feature showcase card
    
    Args:
        title (str): Feature title
        description (str): Feature description
        icon (str): FontAwesome icon class
        color (str): Theme color
        status (str): Feature status ('available', 'coming_soon', 'beta')
        link (str): Optional link URL
    
    Returns:
        html.Div: Feature card component
    """
    
    # Status badge
    status_config = {
        'available': {'color': 'success', 'text': 'Available'},
        'coming_soon': {'color': 'warning', 'text': 'Coming Soon'},
        'beta': {'color': 'info', 'text': 'Beta'}
    }
    
    status_info = status_config.get(status, status_config['available'])
    
    # Card content
    content = [
        # Icon section
        html.Div([
            html.I(
                className=icon,
                style={
                    'fontSize': '3rem',
                    'color': color,
                    'marginBottom': '1rem'
                }
            )
        ], className="text-center"),
        
        # Title and status
        html.Div([
            html.H5(title, className="fw-bold mb-2", style={'color': COLORS['text']}),
            dbc.Badge(
                status_info['text'],
                color=status_info['color'],
                className="mb-3"
            )
        ], className="text-center"),
        
        # Description
        html.P(
            description,
            className="text-center mb-3",
            style={
                'color': COLORS['gray'],
                'lineHeight': '1.6',
                'fontSize': '0.9rem'
            }
        ),
        
        # Action button
        html.Div([
            dbc.Button(
                "Learn More" if status == 'available' else "Coming Soon",
                color="primary" if status == 'available' else "secondary",
                size="sm",
                href=link if link and status == 'available' else None,
                disabled=status != 'available',
                className="w-100"
            )
        ]) if status != 'beta' else html.Div([
            dbc.Button(
                "Try Beta",
                color="info",
                size="sm",
                href=link,
                className="w-100"
            )
        ])
    ]
    
    card_style = {
        'backgroundColor': COLORS['white'],
        'padding': '2rem 1.5rem',
        'borderRadius': '16px',
        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.08)',
        'border': f'2px solid {color}',
        'height': '100%',
        'transition': 'all 0.3s ease'
    }
    
    return html.Div(
        content,
        style=card_style,
        className="hover-lift"
    )

def create_progress_card(title, progress_items, color=COLORS['primary']):
    """
    Create progress tracking card
    
    Args:
        title (str): Card title
        progress_items (list): List of progress items with 'label', 'progress', 'status'
        color (str): Theme color
    
    Returns:
        html.Div: Progress card component
    """
    
    # Calculate overall progress
    total_progress = sum(item['progress'] for item in progress_items)
    avg_progress = total_progress / len(progress_items) if progress_items else 0
    
    # Progress items
    progress_elements = []
    for item in progress_items:
        # Status icon
        status_icons = {
            'completed': 'fas fa-check-circle',
            'in_progress': 'fas fa-clock',
            'pending': 'fas fa-circle'
        }
        
        status_colors = {
            'completed': COLORS['success'],
            'in_progress': COLORS['warning'],
            'pending': COLORS['gray']
        }
        
        status = item.get('status', 'pending')
        
        progress_elements.append(
            html.Div([
                # Item header
                html.Div([
                    html.Div([
                        html.I(
                            className=status_icons[status],
                            style={
                                'color': status_colors[status],
                                'fontSize': '1rem',
                                'marginRight': '0.5rem'
                            }
                        ),
                        html.Span(
                            item['label'],
                            className="fw-medium",
                            style={'color': COLORS['text']}
                        )
                    ]),
                    html.Small(
                        f"{item['progress']}%",
                        style={'color': COLORS['gray']}
                    )
                ], className="d-flex justify-content-between align-items-center mb-2"),
                
                # Progress bar
                dbc.Progress(
                    value=item['progress'],
                    color="success" if status == 'completed' else "primary",
                    size="sm",
                    style={'height': '6px', 'borderRadius': '3px'}
                )
            ], className="mb-3")
        )
    
    return html.Div([
        # Header with overall progress
        html.Div([
            html.H5(title, className="mb-0", style={'color': COLORS['text']}),
            html.Div([
                html.Span(f"{avg_progress:.0f}%", className="fw-bold me-2"),
                html.Small("Complete", className="text-muted")
            ])
        ], className="d-flex justify-content-between align-items-center mb-4"),
        
        # Overall progress bar
        dbc.Progress(
            value=avg_progress,
            color="primary",
            className="mb-4",
            style={'height': '8px', 'borderRadius': '4px'}
        ),
        
        # Individual progress items
        html.Div(progress_elements)
    ], style={
        'backgroundColor': COLORS['white'],
        'padding': '2rem',
        'borderRadius': '16px',
        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.08)',
        'border': f'1px solid {COLORS["light"]}',
        'borderTop': f'4px solid {color}'
    })

def create_notification_card(title, message, type="info", dismissible=True, 
                           actions=None, timestamp=None):
    """
    Create notification card
    
    Args:
        title (str): Notification title
        message (str): Notification message
        type (str): Notification type ('success', 'warning', 'danger', 'info')
        dismissible (bool): Whether notification can be dismissed
        actions (list): Optional action buttons
        timestamp (str): Optional timestamp
    
    Returns:
        html.Div: Notification card component
    """
    
    type_config = {
        'success': {
            'color': COLORS['success'],
            'bg_color': '#f0fdf4',
            'icon': 'fas fa-check-circle'
        },
        'warning': {
            'color': COLORS['warning'],
            'bg_color': '#fffbeb',
            'icon': 'fas fa-exclamation-triangle'
        },
        'danger': {
            'color': COLORS['danger'],
            'bg_color': '#fef2f2',
            'icon': 'fas fa-times-circle'
        },
        'info': {
            'color': COLORS['info'],
            'bg_color': '#f0f9ff',
            'icon': 'fas fa-info-circle'
        }
    }
    
    config = type_config.get(type, type_config['info'])
    
    # Header section
    header_content = [
        html.I(
            className=config['icon'],
            style={
                'color': config['color'],
                'fontSize': '1.2rem',
                'marginRight': '0.5rem'
            }
        ),
        html.Strong(title, style={'color': COLORS['text']})
    ]
    
    if dismissible:
        header_content.append(
            dbc.Button([
                html.I(className="fas fa-times")
            ],
            color="light",
            size="sm",
            className="ms-auto border-0 shadow-none p-0",
            style={'backgroundColor': 'transparent'})
        )
    
    header = html.Div(
        header_content,
        className="d-flex align-items-center mb-2"
    )
    
    # Message section
    message_section = html.P(
        message,
        className="mb-3",
        style={'color': COLORS['text'], 'lineHeight': '1.5'}
    )
    
    # Actions and timestamp section
    footer_content = []
    if actions:
        action_buttons = []
        for action in actions:
            action_buttons.append(
                dbc.Button(
                    action['label'],
                    color=action.get('color', 'primary'),
                    size="sm",
                    href=action.get('href'),
                    id=action.get('id'),
                    className="me-2"
                )
            )
        footer_content.append(
            html.Div(action_buttons, className="d-flex")
        )
    
    if timestamp:
        footer_content.append(
            html.Small(
                timestamp,
                className="text-muted ms-auto" if actions else "",
                style={'alignSelf': 'center'} if actions else {}
            )
        )
    
    footer = None
    if footer_content:
        footer = html.Div(
            footer_content,
            className="d-flex justify-content-between align-items-center"
        )
    
    return html.Div([
        header,
        message_section,
        footer
    ], style={
        'backgroundColor': config['bg_color'],
        'border': f'1px solid {config["color"]}',
        'borderLeft': f'4px solid {config["color"]}',
        'borderRadius': '12px',
        'padding': '1.5rem',
        'marginBottom': '1rem'
    })

# CSS for hover effects
CARD_CSS = f"""
.hover-card:hover {{
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 40px rgba(16, 185, 129, 0.15) !important;
}}

.hover-lift:hover {{
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15) !important;
}}

.feature-card {{
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

.feature-card:hover {{
    transform: translateY(-8px) rotate(1deg);
}}
"""

# Export all functions
__all__ = [
    'create_metric_card',
    'create_info_card',
    'create_chart_card',
    'create_feature_card',
    'create_progress_card',
    'create_notification_card',
    'COLORS',
    'CARD_CSS'
]