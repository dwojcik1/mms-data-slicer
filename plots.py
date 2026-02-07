"""
plots.py - Publication-Quality Visualization Module
=====================================================
Plotly-based visualization with LaTeX labels for scientific publications.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
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
    'BL': '#1f77b4',   # Max Variance (Blue)
    'BM': '#2ca02c',   # Intermediate (Green)
    'BN': '#d62728',   # Min Variance (Red)
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
    'VL': '#1f77b4',
    'VM': '#2ca02c',
    'VN': '#d62728',
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

    'edp': 'mV/m',
    'edi': 'mV/m',
    'fpi_velocity': 'km/s',
    'fpi_density': 'cm⁻³',
    'hpca': 'cm⁻³',
    'feeps': '1/(cm² s sr keV)',

    'mec': 'km',

}


@st.cache_data(show_spinner=False)
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
        'BL': 'B<sub>L</sub>',
        'BM': 'B<sub>M</sub>',
        'BN': 'B<sub>N</sub>',
    }

    
    # Plot each component with standardized colors
    for col in df.columns:
        color = BFIELD_COLORS.get(col, COLORS[0])
        label = label_map.get(col, col)
        
        # Use thick line for magnitude, thinner for components
        width = 2.5 if col in ['Bt', 'B3'] else 1.8
        
        fig.add_trace(go.Scatter(
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
    'editable': False,  # Disable editable to prevent "Click to enter title" artifacts
    'edits': {
        'shapePosition': False, 
    }
}


@st.cache_data(show_spinner=False)
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
        'VL': 'V<sub>L</sub>',
        'VM': 'V<sub>M</sub>',
        'VN': 'V<sub>N</sub>',
    }
    
    # Plot each component
    for col in df.columns:
        color = VELOCITY_COLORS.get(col, COLORS[0])
        label = label_map.get(col, col)
        width = 2.5 if col in ['Vt', 'V3'] else 1.8
        
        fig.add_trace(go.Scatter(
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



@st.cache_data(show_spinner=False)
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
    # --- 1. Trace Generation ---
    if meta['type'] == 'vector':
        # Mapping rules for generic vector components
        # We look for x, y, z OR l, m, n and total magnitude columns
        cols = df.columns
        
        # Check for LMN first (since it's a specific mode)
        l_col = next((c for c in cols if c.endswith('L')), None)
        m_col = next((c for c in cols if c.endswith('M')), None)
        n_col = next((c for c in cols if c.endswith('N')), None)
        
        # Check for Standard XYZ
        x_col = next((c for c in cols if c.lower().endswith('x') or c.lower() == 'x'), None)
        y_col = next((c for c in cols if c.lower().endswith('y') or c.lower() == 'y'), None)
        z_col = next((c for c in cols if c.lower().endswith('z') or c.lower() == 'z'), None)
        
        tot_col = next((c for c in cols if c.lower() in ['tot', 'bt', 'vt', 'et', 'mag', 't']), None)

        # Plot LMN if present
        if l_col and m_col and n_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[l_col], mode='lines',
                name=f"{meta['label']}_L",
                line=dict(color="#1f77b4", width=1.5) # Blue
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df[m_col], mode='lines',
                name=f"{meta['label']}_M",
                line=dict(color="#2ca02c", width=1.5) # Green
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df[n_col], mode='lines',
                name=f"{meta['label']}_N",
                line=dict(color="#d62728", width=1.5) # Red
            ))
            
        # Fallback to XYZ
        elif x_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[x_col], mode='lines',
                name=f"{meta['label']}_x",
                line=dict(color="#1f77b4", width=1.5)
            ))
            if y_col:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[y_col], mode='lines',
                    name=f"{meta['label']}_y",
                    line=dict(color="#2ca02c", width=1.5)
                ))
            if z_col:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[z_col], mode='lines',
                    name=f"{meta['label']}_z",
                    line=dict(color="#d62728", width=1.5)
                ))
        
        if tot_col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[tot_col], mode='lines',
                name=f"|{meta['label']}|",
                line=dict(color="#000000", width=2, dash='solid')
            ))

    elif meta['type'] == 'scalar':
        # Just take the first column
        col = df.columns[0]
        fig.add_trace(go.Scatter(
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


@st.cache_data(show_spinner=False)
def create_psd_plot(
    frequencies, 
    power, 
    title="Power Spectral Density",
    psd_units: str = r"nT²/Hz",
    fit1_range: tuple = None,
    fit2_range: tuple = None,
    show_reference_slopes: bool = True
):
    """
    Create publication-quality log-log PSD plot with dual fit support.
    
    Args:
        frequencies: Frequency array in Hz
        power: Power spectral density array
        title: Plot title
        psd_units: Y-axis units string
        fit1_range: Optional tuple (f_min, f_max) for first fit (displayed in red)
        fit2_range: Optional tuple (f_min, f_max) for second fit (displayed in green)
        show_reference_slopes: Whether to show Kolmogorov and Kinetic reference lines
    
    Returns:
        Tuple of (fig, alpha1, alpha2) where alphas are fitted slopes or None
    """
    fig = go.Figure()
    alpha1, alpha2 = None, None
    fit1_midpoint = None
    fit2_midpoint = None
    
    # Store PSD data - will be added LAST so fit lines render on top
    psd_trace = go.Scatter(
        x=frequencies, y=power, mode='lines', name='PSD',
        line=dict(color='#000000', width=2.2),
        hovertemplate='f=%{x:.3g} Hz<br>PSD=%{y:.3g}<extra></extra>'
    )
    
    # Reference slopes (optional) - add AFTER PSD for correct z-order
    ref_traces = []
    if show_reference_slopes and len(frequencies) > 2:
        f_pos = frequencies[frequencies > 0]
        if len(f_pos) > 0:
            f_min, f_max = f_pos.min(), f_pos.max()
            log_min = np.log10(f_min)
            log_max = np.log10(f_max)
            log_span = log_max - log_min
            # Regions: inertial (left), kinetic (mid-right), dissipation (far right)
            f_inertial_min = 10 ** (log_min + 0.05 * log_span)
            f_inertial_max = 10 ** (log_min + 0.55 * log_span)
            f_kinetic_min = 10 ** (log_min + 0.55 * log_span)
            f_kinetic_max = 10 ** (log_min + 0.90 * log_span)
            f_diss_min = 10 ** (log_min + 0.90 * log_span)
            f_diss_max = 10 ** (log_min + 0.99 * log_span)

            def _ref_segment(f_start, f_end, slope, label, dash):
                if f_start <= 0 or f_end <= 0 or f_end <= f_start:
                    return None
                f_ref = np.array([f_start, f_end])
                f_anchor = np.sqrt(f_start * f_end)
                p_anchor = np.interp(f_anchor, frequencies, power)
                p_ref = p_anchor * (f_ref / f_anchor) ** slope
                return go.Scatter(
                    x=f_ref, y=p_ref, mode='lines',
                    name=label,
                    line=dict(dash=dash, width=2, color='#888888'),
                    visible='legendonly'
                )

            # Kolmogorov -5/3 slope (inertial range)
            tr = _ref_segment(
                f_inertial_min, f_inertial_max, -5/3,
                'f⁻⁵ᐟ³ (Kolmogorov)', 'dash'
            )
            if tr is not None:
                ref_traces.append(tr)

            # Kinetic -8/3 slope (kinetic range)
            tr = _ref_segment(
                f_kinetic_min, f_kinetic_max, -8/3,
                'f⁻⁸ᐟ³ (Kinetic)', 'dot'
            )
            if tr is not None:
                ref_traces.append(tr)

            # Dissipation ~ -2.8 slope (dissipation range)
            tr = _ref_segment(
                f_diss_min, f_diss_max, -2.8,
                'f⁻²·⁸ (Dissipation)', 'dashdot'
            )
            if tr is not None:
                ref_traces.append(tr)
    
    # Helper function to compute fit and add to plot
    def add_fit_trace(fit_range, color, fit_name, fit_idx):
        if fit_range is None or len(frequencies) < 3:
            return None, None, None
            
        fit_f_min, fit_f_max = fit_range
        
        # Get data in the fit range
        mask = (frequencies >= fit_f_min) & (frequencies <= fit_f_max) & (frequencies > 0) & (power > 0)
        f_fit = frequencies[mask]
        p_fit = power[mask]
        
        if len(f_fit) < 3:
            return None, None, None
            
        # Linear fit in log-log space
        log_f = np.log10(f_fit)
        log_p = np.log10(p_fit)
        slope, intercept = np.polyfit(log_f, log_p, 1)
        
        # Generate fit line
        f_line = np.array([fit_f_min * 0.9, fit_f_max * 1.1])
        p_line = 10 ** (intercept + slope * np.log10(f_line))
        
        # Add shaded fit region (but NOT the trace - that's added later for z-order)
        rgba_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.08)"
        fig.add_vrect(
            x0=fit_f_min, x1=fit_f_max,
            fillcolor=rgba_color,
            layer="below",
            line_width=0,
        )
        
        # Calculate midpoint for annotation
        mid_f = np.sqrt(fit_f_min * fit_f_max)
        mid_p = 10 ** (intercept + slope * np.log10(mid_f))
        
        return slope, (mid_f, mid_p), (f_line, p_line, color)
    
    # Compute fits first (but don't add traces yet)
    fit1_result = add_fit_trace(fit1_range, '#d62728', 'Fit 1', 1)
    fit2_result = add_fit_trace(fit2_range, '#2ca02c', 'Fit 2', 2)
    
    alpha1 = fit1_result[0] if fit1_result[0] else None
    fit1_midpoint = fit1_result[1] if fit1_result[0] else None
    alpha2 = fit2_result[0] if fit2_result[0] else None
    fit2_midpoint = fit2_result[1] if fit2_result[0] else None
    
    # Add PSD trace FIRST
    fig.add_trace(psd_trace)
    
    # Add reference slopes AFTER PSD so they render on top when enabled
    for tr in ref_traces:
        fig.add_trace(tr)

    
    # Add fit lines AFTER PSD so they render ON TOP
    if fit1_result[0] is not None:
        f_line, p_line, color = fit1_result[2]
        fig.add_trace(go.Scatter(
            x=f_line, y=p_line, mode='lines',
            name='Fit 1',
            line=dict(dash='solid', width=3, color=color),
            showlegend=False
        ))
    
    if fit2_result[0] is not None:
        f_line, p_line, color = fit2_result[2]
        fig.add_trace(go.Scatter(
            x=f_line, y=p_line, mode='lines',
            name='Fit 2',
            line=dict(dash='solid', width=3, color=color),
            showlegend=False
        ))
    
    # Add on-plot annotations for slopes
    if alpha1 is not None and fit1_midpoint is not None:
        fig.add_annotation(
            x=np.log10(fit1_midpoint[0]),
            y=np.log10(fit1_midpoint[1]) + 0.3,
            text=f"<b>α₁ = {alpha1:.2f}</b>",
            showarrow=False,
            font=dict(size=16, color='#d62728', family='Space Grotesk'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#d62728',
            borderwidth=1,
            borderpad=4,
            xref='x', yref='y'
        )
    
    if alpha2 is not None and fit2_midpoint is not None:
        fig.add_annotation(
            x=np.log10(fit2_midpoint[0]),
            y=np.log10(fit2_midpoint[1]) + 0.3,
            text=f"<b>α₂ = {alpha2:.2f}</b>",
            showarrow=False,
            font=dict(size=16, color='#2ca02c', family='Space Grotesk'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#2ca02c',
            borderwidth=1,
            borderpad=4,
            xref='x', yref='y'
        )
    
    # Y-axis label
    ylabel = f"PSD [{psd_units}]"
    
    # Publication-quality layout - SQUARE ASPECT RATIO
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', family='Space Grotesk, sans-serif'),
        hovermode='x unified',
        
        # Square dimensions
        width=700,
        height=700,
        
        # No title in the plot (use st.subheader instead)
        title=None,
        
        xaxis=dict(
            title_text="Frequency [Hz]",
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=14, color='black'),
            type='log',
            exponentformat='power',
            dtick=1,
            showgrid=True,
            gridcolor=GRID_COLOR,
            showline=True,
            linewidth=0.8,
            linecolor='#666666',
            mirror=True,
            minor=dict(ticks="outside", ticklen=4, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        ),
        
        yaxis=dict(
            title_text=ylabel,
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=14, color='black'),
            type='log',
            exponentformat='power',
            dtick=1,
            showgrid=True,
            gridcolor=GRID_COLOR,
            showline=True,
            linewidth=0.8,
            linecolor='#666666',
            mirror=True,
            minor=dict(ticks="outside", ticklen=4, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        ),
        
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=13, color='black')
        ),
        
        margin=dict(l=80, r=40, t=60, b=80),
    )

    # Lock x-axis range to data (prevents legend/annotations from expanding range)
    f_pos = frequencies[frequencies > 0]
    if len(f_pos) > 0:
        f_min_plot = float(np.min(f_pos))
        f_max_plot = float(np.max(f_pos))
        if f_min_plot > 0 and f_max_plot > f_min_plot:
            fig.update_xaxes(range=[np.log10(f_min_plot), np.log10(f_max_plot)])
    
    return fig, alpha1, alpha2


@st.cache_data(show_spinner=False)
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

# ============================================================================
# Orbit Visualization
# ============================================================================

def plot_mms_orbit_wrapper(
    trange: List[str],
    probes: List[str],
    plane: str = 'xy',
    coord: str = 'gse'
):
    """
    Generate MMS Orbit Plots manually using standard data loaders.
    
    Loads MEC ephemeris data via downloader module, converts to Re,
    and plots using matplotlib with strict styling.
    
    Args:
        trange: Time range ['start', 'end']
        probes: List of probes using strings (e.g. ['1', '2'])
        plane: Projection plane ('xy', 'xz', 'yz')
        coord: Coordinate system ('gse', 'gsm')
    
    Returns:
        matplotlib.figure.Figure: The generated orbit plot figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from downloader import load_mms_universal
    
    # Constants
    RE_KM = 6371.2
    
    # Close existing figures
    plt.close('all')
    
    # Create Figure and Axes
    # Use figsize 10x10 to ensure base canvas is square
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color Map
    color_map = {
        '1': 'red',
        '2': 'green',
        '3': 'blue',
        '4': 'black'
    }
    
    has_data = False
    
    for p in probes:
        probe_id = str(p)
        try:
            # Load MEC data (Ephemeris)
            # data_rate='srvy' maps to epht89q in most cases
            data_dict = load_mms_universal(
                instrument='mec',
                trange=trange,
                probe=probe_id,
                data_rate='srvy',
                level='l2',
                coord=coord,
                datatype='epht89q'
            )
            
            if not data_dict or 'MEC' not in data_dict:
                continue
                
            df = data_dict['MEC']
            if df.empty:
                continue
                
            has_data = True
            
            # Convert to Re
            # Unit in downloader is 'km', assuming standard MEC
            x_re = df['X'] / RE_KM
            y_re = df['Y'] / RE_KM
            z_re = df['Z'] / RE_KM
            
            # Select Plane Data
            if plane == 'xy':
                x_plot, y_plot = x_re, y_re
                xlabel, ylabel = 'X Position, Re', 'Y Position, Re'
            elif plane == 'xz':
                x_plot, y_plot = x_re, z_re
                xlabel, ylabel = 'X Position, Re', 'Z Position, Re'
            elif plane == 'yz':
                x_plot, y_plot = y_re, z_re
                xlabel, ylabel = 'Y Position, Re', 'Z Position, Re'
            else:
                st.error(f"Unknown plane: {plane}")
                return None
            
            # Plot Orbit
            color = color_map.get(probe_id, 'black')
            # Styling based on user feedback:
            # - Markers along the line (marker='x', markevery)
            # - Thinner lines but clear markers
            # Z-order 12 to be ON TOP of Earth image (Z=10)
            ax.plot(
                x_plot, y_plot,
                label=f'MMS{probe_id}',
                color=color,
                linewidth=1.0,      # Slightly thicker than 0.5 for visibility
                marker='x',         # Markers as seen in target image
                markevery=20,       # Regular intervals
                markersize=4,
                zorder=12
            )
            
            # Add Start/End markers (Optional, keeping for clarity but making subtle)
            if len(x_plot) > 0:
                 ax.scatter(x_plot.iloc[0], y_plot.iloc[0], marker='o', color=color, s=30, zorder=13) # Start
                 ax.scatter(x_plot.iloc[-1], y_plot.iloc[-1], marker='s', color=color, s=30, zorder=13) # End

        except Exception as e:
            # Log warning but continue with other probes
            print(f"Failed to plot MMS{probe_id}: {e}")
            continue

    if not has_data:
        st.error("No orbit data found for selected configuration.")
        return fig
    
    # --- STYLING ---
    
    # 1. Earth Representation
    try:
        import matplotlib.image as mpimg
        import os
        # Load earth image
        current_dir = os.path.dirname(os.path.abspath(__file__))
        earth_path = os.path.join(current_dir, 'assets', 'earth.png')
        if os.path.exists(earth_path):
            img = mpimg.imread(earth_path)
            # Display centered at 0,0 with radius 1 (Extent includes -1 to 1 in both axis)
            ax.imshow(img, extent=[-1, 1, -1, 1], zorder=10)
        else:
            raise FileNotFoundError("Asset not found")
    except Exception as e:
        # Fallback to simple circle
        print(f"Warning: Could not load earth image: {e}")
        earth = patches.Circle((0, 0), radius=1.0, facecolor='#1f77b4', edgecolor='white', linewidth=1, alpha=0.8, zorder=10)
        ax.add_patch(earth)
    
    # Text annotation for Earth?
    # ax.text(0, 0, 'Earth', color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=11)
    
    # 2. Aspect Ratio & Limits
    ax.set_box_aspect(1)
    ax.set_aspect('equal', adjustable='datalim')
    
    # 3. Grid
    ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 4. Labels and Title
    # Large fonts as requested (approximated from visual)
    font_size_labels = 14
    font_size_title = 16
    font_size_ticks = 12
    
    ax.set_xlabel(xlabel, fontsize=font_size_labels)
    ax.set_ylabel(ylabel, fontsize=font_size_labels)
    ax.set_title(f"MMS Orbit - {plane.upper()} Plane ({coord.upper()})", fontsize=font_size_title, pad=15)
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=font_size_ticks)
    
    # 5. Legend
    # Only if data exists
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', frameon=True, fontsize=12, markerscale=1.5)
        
    # Final Autoscale to ensure lines are seen despite Image
    ax.autoscale(True)
    
    return fig
