"""
plots.py - Publication-Quality Visualization Module
=====================================================
Plotly-based visualization with LaTeX labels for scientific publications.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Grid color - very light grey (matches TS plots)
GRID_COLOR = '#E5E5E5'

# Publication-quality layout defaults (matches time series plots)
PUBLICATION_LAYOUT = dict(
    # White background for visibility and clean export
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black', family='Arial, sans-serif'),
    hovermode='x unified',
    # Legend inside plot, large font with border
    legend=dict(
        orientation='h',
        yanchor='top',
        y=0.98,
        xanchor='right',
        x=0.98,
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='rgba(0,0,0,0.3)',
        borderwidth=1,
        font=dict(size=12, color='black')
    ),
    margin=dict(l=70, r=40, t=80, b=60),
)

# Responsive layout defaults (legacy, for backwards compat)
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

# Velocity field colors (same scheme)
VELOCITY_COLORS = {
    'Vx': '#1f77b4',   # Blue
    'Vy': '#2ca02c',   # Green
    'Vz': '#d62728',   # Red  
    'Vt': '#000000',   # Black
    'V0': '#1f77b4',
    'V1': '#2ca02c',
    'V2': '#d62728',
    'V3': '#000000',
}

# Electric field colors (same scheme)
EFIELD_COLORS = {
    'Ex': '#1f77b4',   # Blue
    'Ey': '#2ca02c',   # Green
    'Ez': '#d62728',   # Red  
    'Et': '#000000',   # Black
}

# Position/ephemeris colors
POSITION_COLORS = {
    'X': '#1f77b4',    # Blue
    'Y': '#2ca02c',    # Green
    'Z': '#d62728',    # Red  
    'R': '#000000',    # Black
}

# Scalar data color (for density, flux, current)
SCALAR_COLOR = '#9467bd'  # Purple

# Instrument unit mapping
INSTRUMENT_UNITS = {
    'fgm': 'nT',
    'scm': 'nT',
    'fsm': 'nT',
    'edp': 'mV/m',
    'edi': 'mV/m',
    'fpi_velocity': 'km/s',
    'fpi_density': 'cm⁻³',
    'hpca': 'cm⁻³',
    'feeps': '1/(cm² s sr keV)',
    'eis': '1/(cm² s sr keV)',
    'aspoc': 'μA',
    'mec': 'km',
    'state': 'deg',
    'tqf': '',
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
    
    # HTML subscript labels for legend (LaTeX unreliable in Streamlit)
    label_map = {
        'Bx': 'B<sub>x</sub>',
        'By': 'B<sub>y</sub>',
        'Bz': 'B<sub>z</sub>',
        'Bt': '|<b>B</b>|',
        'B0': 'B<sub>x</sub>',
        'B1': 'B<sub>y</sub>',
        'B2': 'B<sub>z</sub>',
        'B3': '|<b>B</b>|',
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
        
        # Axis titles - plain text X-axis, LaTeX Y-axis
        xaxis_title=dict(
            text="Epoch [UTC]",
            font=dict(size=16, color='black')
        ),
        yaxis_title=dict(
            text=r"<b>B</b> [nT]",
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

# PSD plot config with editable shapes for interactive frequency selection
PSD_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'mms_psd',
        'height': 800,
        'width': 800,
        'scale': 2
    },
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'editable': True,  # Enable shape dragging
    'edits': {
        'shapePosition': True,  # Allow dragging shapes
    }
}


def plot_velocity_field(
    df,
    title: str = "Bulk Velocity",
    species: str = "ion",
    height: int = 500,
    show_grid: bool = True
):
    """
    Create publication-quality velocity field plot for FPI data.
    
    Same color scheme as magnetic field:
    - Vx: Blue (#1f77b4)
    - Vy: Green (#2ca02c)  
    - Vz: Red (#d62728)
    - |V|: Black (#000000)
    
    Args:
        df: DataFrame with DatetimeIndex and columns Vx, Vy, Vz, Vt
        title: Dynamic title string
        species: 'ion' or 'electron' for appropriate labeling
        height: Plot height in pixels
        show_grid: Whether to show grid lines
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # HTML subscript labels for legend
    label_map = {
        'Vx': 'V<sub>x</sub>',
        'Vy': 'V<sub>y</sub>',
        'Vz': 'V<sub>z</sub>',
        'Vt': '|<b>V</b>|',
        'V0': 'V<sub>x</sub>',
        'V1': 'V<sub>y</sub>',
        'V2': 'V<sub>z</sub>',
        'V3': '|<b>V</b>|',
    }
    
    # Plot each component
    for col in df.columns:
        color = VELOCITY_COLORS.get(col, COLORS[0])
        label = label_map.get(col, col)
        width = 2.5 if col in ['Vt', 'V3'] else 1.8
        
        fig.add_trace(go.Scattergl(
            x=df.index,
            y=df[col].values,
            mode='lines',
            name=label,
            line=dict(color=color, width=width),
            hovertemplate=f'{label}: %{{y:.1f}} km/s<extra></extra>'
        ))
    
    # Y-axis label based on species
    if species == 'electron':
        ylabel = '<b>V</b><sub>e</sub> [km/s]'
    else:
        ylabel = '<b>V</b><sub>i</sub> [km/s]'
    
    # Publication-quality WHITE THEME layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', family='Arial, sans-serif'),
        title=dict(
            text=title,
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        xaxis_title=dict(
            text="Epoch [UTC]",
            font=dict(size=14, color='black')
        ),
        yaxis_title=dict(
            text=ylabel,
            font=dict(size=14, color='black')
        ),
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=0.98,
            xanchor='right',
            x=0.98,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1,
            font=dict(size=12, color='black')
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    
    # Grid styling
    grid_color = '#E5E5E5'
    
    fig.update_xaxes(
        showgrid=show_grid,
        gridcolor=grid_color,
        gridwidth=1,
        tickfont=dict(size=12, color='black'),
        tickformat='%H:%M:%S',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=show_grid,
        gridcolor=grid_color,
        gridwidth=1,
        tickfont=dict(size=12, color='black'),
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.3)',
        zerolinewidth=1.5,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    
    return fig



def plot_time_series(df, meta, title=None):
    """
    Unified "Publication-Standard" Time Series Plot.
    
    Args:
        df: DataFrame (index=Epoch)
        meta: Dict with keys 'label', 'unit', 'type'
              example: {'label': r"$\mathbf{B}$", 'unit': "[nT]", 'type': 'vector'}
    """
    fig = go.Figure()

    # --- 1. Trace Generation ---
    if meta['type'] == 'vector':
        # Mapping rules for generic vector components
        # We look for x, y, z, and total magnitude columns
        cols = df.columns
        x_col = next((c for c in cols if c.lower().endswith('x') or c.lower() == 'x'), None)
        y_col = next((c for c in cols if c.lower().endswith('y') or c.lower() == 'y'), None)
        z_col = next((c for c in cols if c.lower().endswith('z') or c.lower() == 'z'), None)
        tot_col = next((c for c in cols if c.lower() in ['tot', 'bt', 'vt', 'et', 'mag', 't']), None)

        if x_col:
            fig.add_trace(go.Scattergl(
                x=df.index, y=df[x_col], mode='lines',
                name=f"{meta['label']}_x",
                line=dict(color="#1f77b4", width=1.5)
            ))
        if y_col:
            fig.add_trace(go.Scattergl(
                x=df.index, y=df[y_col], mode='lines',
                name=f"{meta['label']}_y",
                line=dict(color="#2ca02c", width=1.5)
            ))
        if z_col:
            fig.add_trace(go.Scattergl(
                x=df.index, y=df[z_col], mode='lines',
                name=f"{meta['label']}_z",
                line=dict(color="#d62728", width=1.5)
            ))
        if tot_col:
            fig.add_trace(go.Scattergl(
                x=df.index, y=df[tot_col], mode='lines',
                name=f"|{meta['label']}|",
                line=dict(color="#000000", width=2, dash='solid')
            ))

    elif meta['type'] == 'scalar':
        # Just take the first column
        col = df.columns[0]
        fig.add_trace(go.Scattergl(
            x=df.index, y=df[col], mode='lines',
            name=meta['label'],
            line=dict(color="#9467bd", width=1.5)
        ))

    # --- 2. Styling & Layout (Strict "FGM Style") ---
    
    # Dynamic Y-Axis Label
    y_title = f"{meta['label']} {meta['unit']}"
    
    fig.update_layout(
        # Background
        plot_bgcolor='white',
        paper_bgcolor='white',
        
        # Font settings
        font=dict(family='Space Grotesk, sans-serif'),
        
        # Title
        title=dict(
            text=title or f"MMS {meta['label']} Time Series",
            font=dict(size=22, color='black'),
            x=0.05,
            y=0.95
        ),
        
        # Axes Labels
        xaxis=dict(
            title_text="Epoch [UTC]",
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=14, color='black'),
            showgrid=True,
            gridcolor='#E5E5E5',
            # griddash='dash',  # Note: 'griddash' supported in newer Plotly versions
            zeroline=False
        ),
        yaxis=dict(
            title_text=y_title,
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=14, color='black'),
            showgrid=True,
            gridcolor='#E5E5E5',
            # griddash='dash',
            zeroline=False
        ),
        
        # Legend
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=14, color='black')
        ),
        
        # Margins
        margin=dict(l=70, r=40, t=80, b=60),
        hovermode='x unified',
    )

    # Axis styling (standardized across all plots)
    fig.update_xaxes(
        griddash='dash',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        title_standoff=15
    )
    fig.update_yaxes(
        griddash='dash',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        title_standoff=15
    )

    return fig


def create_psd_plot(
    frequencies, 
    power, 
    title="Power Spectral Density",
    psd_units: str = r"PSD",
    height=500,
    show_fit_range: bool = True,
    fit_range: tuple = None
):
    """
    Create log-log PSD plot with publication-quality units and interactive fit range.
    
    Args:
        frequencies: Frequency array in Hz
        power: Power spectral density array
        title: Plot title (can include LaTeX)
        psd_units: Y-axis units (LaTeX formatted, e.g., r'$\\mathrm{nT}^2/\\mathrm{Hz}$')
        height: Plot height (also used for width to create square aspect)
        show_fit_range: Whether to show draggable frequency range selectors
        fit_range: Optional tuple (f_min, f_max) for initial fit range bounds
    
    Returns:
        Plotly Figure object with interactive elements
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
                name='f⁻⁵ᐟ³ (Kolmogorov)',
                line=dict(dash='dash', width=2, color=COLORS[1])
            ))
            
            # Kinetic -8/3 slope
            p_ref_83 = p_mid * (f_ref / f_mid) ** (-8/3)
            fig.add_trace(go.Scatter(
                x=f_ref, y=p_ref_83, mode='lines',
                name='f⁻⁸ᐟ³ (Kinetic)',
                line=dict(dash='dot', width=2, color=COLORS[2])
            ))
    
    # Format Y-axis label with physics units
    ylabel = f"PSD ({psd_units})" if psd_units else "Power Spectral Density"
    
    # Calculate square dimensions
    square_size = height
    
    # Publication-quality layout (matches time series plots)
    fig.update_layout(
        **PUBLICATION_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        xaxis_title=dict(
            text="Frequency [Hz]",
            font=dict(size=14, color='black')
        ),
        yaxis_title=dict(
            text=ylabel,
            font=dict(size=14, color='black')
        ),
        xaxis_type='log',
        yaxis_type='log',
        height=square_size,
        width=square_size,  # Square aspect ratio
    )
    
    # Axis styling (matches time series plots)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickfont=dict(size=12, color='black'),
        title_standoff=15,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickfont=dict(size=12, color='black'),
        title_standoff=15,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    # Add interactive frequency band selection (draggable vertical lines)
    if show_fit_range and len(frequencies) > 2:
        f_pos = frequencies[frequencies > 0]
        if len(f_pos) > 0:
            # Default fit range: middle decade of the spectrum
            if fit_range is None:
                log_f_min = np.log10(f_pos.min())
                log_f_max = np.log10(f_pos.max())
                log_range = log_f_max - log_f_min
                fit_f_min = 10 ** (log_f_min + log_range * 0.25)
                fit_f_max = 10 ** (log_f_min + log_range * 0.75)
            else:
                fit_f_min, fit_f_max = fit_range
            
            # Get y-range for the lines
            p_pos = power[power > 0]
            if len(p_pos) > 0:
                y_min = p_pos.min() * 0.1
                y_max = p_pos.max() * 10
            else:
                y_min, y_max = 1e-10, 1e10
            
            # Add shaded region between fit bounds
            fig.add_vrect(
                x0=fit_f_min, x1=fit_f_max,
                fillcolor="rgba(129, 140, 248, 0.15)",
                layer="below",
                line_width=0,
                annotation_text="Fit Range",
                annotation_position="top left",
                annotation=dict(font_size=10, font_color="rgba(129, 140, 248, 0.8)")
            )
            
            # Left bound line (draggable)
            fig.add_shape(
                type="line",
                x0=fit_f_min, x1=fit_f_min,
                y0=y_min, y1=y_max,
                line=dict(color="rgba(129, 140, 248, 0.8)", width=2, dash="solid"),
                name="f_min",
                editable=True,
            )
            
            # Right bound line (draggable)
            fig.add_shape(
                type="line",
                x0=fit_f_max, x1=fit_f_max,
                y0=y_min, y1=y_max,
                line=dict(color="rgba(129, 140, 248, 0.8)", width=2, dash="solid"),
                name="f_max",
                editable=True,
            )
            
            # Add instruction annotation
            fig.add_annotation(
                text="💡 Drag purple lines to adjust fit range",
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(size=10, color="rgba(100, 100, 100, 0.7)"),
                align="center"
            )
    
    return fig


def create_pdf_plot(
    bin_centers, 
    density, 
    title="PDF",
    xlabel="Value",
    log_y=False, 
    height=400
):
    """Create publication-quality PDF histogram plot with optional Gaussian overlay."""
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
    
    # Publication-quality layout (matches time series plots)
    fig.update_layout(
        **PUBLICATION_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        xaxis_title=dict(
            text=xlabel,
            font=dict(size=14, color='black')
        ),
        yaxis_title=dict(
            text="Probability Density",
            font=dict(size=14, color='black')
        ),
        height=height,
        bargap=0.05
    )
    
    # Axis styling (matches time series plots)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickfont=dict(size=12, color='black'),
        title_standoff=15,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickfont=dict(size=12, color='black'),
        title_standoff=15,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    if log_y:
        fig.update_yaxes(type='log')
    
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
