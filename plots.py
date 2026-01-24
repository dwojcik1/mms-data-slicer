"""
plots.py - Publication-Quality Visualization Module
=====================================================
Plotly-based visualization with LaTeX labels for scientific publications.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Responsive layout defaults
RESPONSIVE_LAYOUT = dict(
    template='plotly_white',
    hovermode='x unified',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        font=dict(size=11)
    ),
    margin=dict(l=50, r=20, t=50, b=40),
    autosize=True
)

# Publication-standard B-field colors (JGR/GRL style)
BFIELD_COLORS = {
    'Bx': '#1f77b4',   # Blue
    'By': '#2ca02c',   # Green
    'Bz': '#d62728',   # Red  
    'Bt': '#000000',   # Black
    'B0': '#1f77b4',
    'B1': '#2ca02c',
    'B2': '#d62728',
    'B3': '#000000',
}


def plot_magnetic_field(
    df,
    title: str = "Magnetic Field",
    height: int = 550,
    show_grid: bool = True
):
    """
    Create publication-quality unified magnetic field plot (JGR/GRL style).
    
    White background ensures visibility of black |B| trace and clean PNG export.
    
    Colors:
    - Bx: Blue (#1f77b4)
    - By: Green (#2ca02c)  
    - Bz: Red (#d62728)
    - |B|: Black (#000000)
    
    Args:
        df: DataFrame with DatetimeIndex and columns Bx, By, Bz, Bt
        title: Dynamic title string
        height: Plot height in pixels
        show_grid: Whether to show grid lines
    
    Returns:
        Tuple of (Figure, config_dict) for st.plotly_chart
    """
    fig = go.Figure()
    
    # LaTeX-style labels for legend
    label_map = {
        'Bx': r'$B_x$',
        'By': r'$B_y$',
        'Bz': r'$B_z$',
        'Bt': r'$|\mathbf{B}|$',
        'B0': r'$B_x$',
        'B1': r'$B_y$',
        'B2': r'$B_z$',
        'B3': r'$|\mathbf{B}|$',
    }
    
    # Plot each component with standardized colors
    for col in df.columns:
        color = BFIELD_COLORS.get(col, COLORS[0])
        label = label_map.get(col, col)
        
        # Use thick line for magnitude, thinner for components
        width = 2.5 if col in ['Bt', 'B3'] else 1.8
        
        fig.add_trace(go.Scattergl(
            x=df.index,
            y=df[col].values,
            mode='lines',
            name=label,
            line=dict(color=color, width=width),
            hovertemplate=f'{label}: %{{y:.3f}} nT<extra></extra>'
        ))
    
    # Publication-quality WHITE THEME layout
    fig.update_layout(
        # Force white background for visibility and clean export
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', family='Arial, sans-serif'),
        
        # Large, bold title with LaTeX
        title=dict(
            text=title,
            font=dict(size=20, color='black'),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        
        # Axis titles with LaTeX and proper sizing
        xaxis_title=dict(
            text=r"$\text{Epoch [UTC]}$",
            font=dict(size=16, color='black')
        ),
        yaxis_title=dict(
            text=r"$\mathbf{B}$ [nT]",
            font=dict(size=16, color='black')
        ),
        
        height=height,
        hovermode='x unified',
        
        # Legend inside plot, large font
        legend=dict(
            orientation='h',
            yanchor='top',
            y=0.98,
            xanchor='right',
            x=0.98,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1,
            font=dict(size=14, color='black')
        ),
        
        margin=dict(l=70, r=40, t=80, b=60),
    )
    
    # Grid styling - very light grey
    grid_color = '#E5E5E5'  # Light grey
    
    fig.update_xaxes(
        showgrid=show_grid,
        gridcolor=grid_color,
        gridwidth=1,
        tickfont=dict(size=13, color='black'),
        tickformat='%H:%M:%S',
        title_standoff=15,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=show_grid,
        gridcolor=grid_color,
        gridwidth=1,
        tickfont=dict(size=13, color='black'),
        title_standoff=15,
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.3)',
        zerolinewidth=1.5,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    # Range slider for time navigation
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    
    return fig


# Plotly config for high-resolution PNG export
PLOTLY_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'mms_magnetic_field',
        'height': 800,
        'width': 1400,
        'scale': 2  # 2x resolution for publication quality
    },
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}


def create_time_series_plot(
    time_data, 
    data, 
    title="Time Series", 
    ylabel="Value",
    component_labels: Optional[List[str]] = None,
    height=400
):

    """
    Create responsive time series plot with optional LaTeX component labels.
    
    Args:
        time_data: Time array
        data: Data array (1D or 2D)
        title: Plot title (can include LaTeX)
        ylabel: Y-axis label (can include LaTeX)
        component_labels: LaTeX labels for each component
        height: Plot height in pixels
    """
    fig = go.Figure()
    
    if len(data.shape) == 1:
        fig.add_trace(go.Scattergl(
            x=time_data, y=data, mode='lines', 
            name=ylabel, line=dict(color=COLORS[0], width=1.5)
        ))
    else:
        labels = component_labels or [f'Component {i}' for i in range(data.shape[1])]
        for i in range(min(data.shape[1], 4)):
            label = labels[i] if i < len(labels) else f'C{i}'
            fig.add_trace(go.Scattergl(
                x=time_data, y=data[:, i], mode='lines',
                name=label, line=dict(color=COLORS[i], width=1.5)
            ))
    
    fig.update_layout(
        **RESPONSIVE_LAYOUT,
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis_title="Time (UTC)",
        yaxis_title=ylabel,
        height=height
    )
    
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


def create_psd_plot(
    frequencies, 
    power, 
    title="Power Spectral Density",
    psd_units: str = r"PSD",
    height=450
):
    """
    Create log-log PSD plot with publication-quality units.
    
    Args:
        frequencies: Frequency array in Hz
        power: Power spectral density array
        title: Plot title (can include LaTeX)
        psd_units: Y-axis units (LaTeX formatted, e.g., r'$\mathrm{nT}^2/\mathrm{Hz}$')
        height: Plot height
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scattergl(
        x=frequencies, y=power, mode='lines', name='PSD',
        line=dict(color=COLORS[0], width=2),
        hovertemplate='f=%{x:.3g} Hz<br>PSD=%{y:.3g}<extra></extra>'
    ))
    
    # Add reference slopes
    if len(frequencies) > 2:
        f_pos = frequencies[frequencies > 0]
        if len(f_pos) > 0:
            f_min, f_max = f_pos.min(), f_pos.max()
            f_mid = np.sqrt(f_min * f_max)
            p_mid = np.interp(f_mid, frequencies, power)
            
            f_ref = np.array([f_mid / 10, f_mid * 10])
            
            # Kolmogorov -5/3 slope
            p_ref_53 = p_mid * (f_ref / f_mid) ** (-5/3)
            fig.add_trace(go.Scatter(
                x=f_ref, y=p_ref_53, mode='lines', 
                name=r'$f^{-5/3}$',
                line=dict(dash='dash', width=2, color=COLORS[1])
            ))
            
            # Kinetic -8/3 slope
            p_ref_83 = p_mid * (f_ref / f_mid) ** (-8/3)
            fig.add_trace(go.Scatter(
                x=f_ref, y=p_ref_83, mode='lines',
                name=r'$f^{-8/3}$',
                line=dict(dash='dot', width=2, color=COLORS[2])
            ))
    
    # Format Y-axis label with physics units
    ylabel = f"PSD ({psd_units})" if psd_units else "Power Spectral Density"
    
    fig.update_layout(
        **RESPONSIVE_LAYOUT,
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis_title="Frequency (Hz)",
        yaxis_title=ylabel,
        xaxis_type='log',
        yaxis_type='log',
        height=height
    )
    
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


def create_pdf_plot(
    bin_centers, 
    density, 
    title="PDF",
    xlabel="Value",
    log_y=False, 
    height=400
):
    """Create responsive PDF histogram plot with optional Gaussian overlay."""
    fig = go.Figure()
    
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1
    
    fig.add_trace(go.Bar(
        x=bin_centers, y=density, name='PDF', 
        marker_color=COLORS[0], opacity=0.75, 
        width=bin_width * 0.85,
        hovertemplate='Value=%{x:.3g}<br>Density=%{y:.3g}<extra></extra>'
    ))
    
    # Gaussian overlay
    if len(density) > 0:
        total = np.sum(density) * bin_width
        if total > 0:
            mean = np.sum(bin_centers * density * bin_width) / total
            var = np.sum((bin_centers - mean)**2 * density * bin_width) / total
            std = np.sqrt(var) if var > 0 else 1
            
            x_gauss = np.linspace(bin_centers.min(), bin_centers.max(), 100)
            y_gauss = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x_gauss - mean)**2 / (2 * std**2))
            
            fig.add_trace(go.Scatter(
                x=x_gauss, y=y_gauss, mode='lines', name='Gaussian',
                line=dict(color=COLORS[1], width=2, dash='dash')
            ))
    
    fig.update_layout(
        **RESPONSIVE_LAYOUT,
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis_title=xlabel,
        yaxis_title="Probability Density",
        height=height,
        bargap=0.05
    )
    
    if log_y:
        fig.update_yaxes(type='log')
    
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


def create_stats_display(stats) -> Dict[str, Any]:
    """Format statistics for display."""
    return {
        "Mean": f"{stats.mean:.6g}",
        "Median": f"{stats.median:.6g}",
        "Std Dev": f"{stats.std:.6g}",
        "Variance": f"{stats.variance:.6g}",
        "Skewness": f"{stats.skewness:.4f}",
        "Kurtosis": f"{stats.kurtosis:.4f}",
        "Min": f"{stats.min_val:.6g}",
        "Max": f"{stats.max_val:.6g}",
        "Samples": f"{stats.n_samples:,}",
        "NaN Count": f"{stats.n_nan:,}"
    }
