"""
plots.py - Responsive Visualization Module
============================================
Plotly-based visualization with mobile-first responsive design.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LABELS = ['X', 'Y', 'Z', 'W']

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


def create_time_series_plot(time_data, data, title="Time Series", ylabel="Value", height=400):
    """Create responsive interactive time series plot."""
    fig = go.Figure()
    
    if len(data.shape) == 1:
        fig.add_trace(go.Scattergl(
            x=time_data, y=data, mode='lines', 
            name=ylabel, line=dict(color=COLORS[0], width=1.5)
        ))
    else:
        for i in range(min(data.shape[1], 4)):
            fig.add_trace(go.Scattergl(
                x=time_data, y=data[:, i], mode='lines',
                name=LABELS[i], line=dict(color=COLORS[i], width=1.5)
            ))
    
    fig.update_layout(
        **RESPONSIVE_LAYOUT,
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis_title="Time (UTC)",
        yaxis_title=ylabel,
        height=height
    )
    
    # Range slider for navigation
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04),
        tickfont=dict(size=10)
    )
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


def create_psd_plot(frequencies, power, title="Power Spectral Density", height=450):
    """Create responsive log-log PSD plot with reference slopes."""
    fig = go.Figure()
    
    fig.add_trace(go.Scattergl(
        x=frequencies, y=power, mode='lines', name='PSD',
        line=dict(color=COLORS[0], width=2),
        hovertemplate='f=%{x:.3g} Hz<br>PSD=%{y:.3g}<extra></extra>'
    ))
    
    # Add Kolmogorov -5/3 slope reference
    if len(frequencies) > 2:
        f_pos = frequencies[frequencies > 0]
        if len(f_pos) > 0:
            f_min, f_max = f_pos.min(), f_pos.max()
            f_mid = np.sqrt(f_min * f_max)
            p_mid = np.interp(f_mid, frequencies, power)
            
            f_ref = np.array([f_mid / 10, f_mid * 10])
            p_ref_53 = p_mid * (f_ref / f_mid) ** (-5/3)
            p_ref_83 = p_mid * (f_ref / f_mid) ** (-8/3)
            
            fig.add_trace(go.Scatter(
                x=f_ref, y=p_ref_53, mode='lines', name='f⁻⁵/³',
                line=dict(dash='dash', width=2, color=COLORS[1])
            ))
            fig.add_trace(go.Scatter(
                x=f_ref, y=p_ref_83, mode='lines', name='f⁻⁸/³',
                line=dict(dash='dot', width=2, color=COLORS[2])
            ))
    
    fig.update_layout(
        **RESPONSIVE_LAYOUT,
        title=dict(text=title, font=dict(size=16), x=0.5),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        xaxis_type='log',
        yaxis_type='log',
        height=height
    )
    
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


def create_pdf_plot(bin_centers, density, title="PDF", xlabel="Value", log_y=False, height=400):
    """Create responsive PDF histogram plot."""
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
