"""
plots.py - Visualization Module
================================
Plotly-based visualization functions for MMS data analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LABELS = ['X', 'Y', 'Z', 'W']


def create_time_series_plot(time_data, data, title="Time Series", ylabel="Value", height=400):
    """Create interactive time series plot."""
    fig = go.Figure()
    
    if len(data.shape) == 1:
        fig.add_trace(go.Scattergl(x=time_data, y=data, mode='lines', 
                                    name=ylabel, line=dict(color=COLORS[0], width=1)))
    else:
        for i in range(min(data.shape[1], 4)):
            fig.add_trace(go.Scattergl(x=time_data, y=data[:, i], mode='lines',
                                        name=LABELS[i], line=dict(color=COLORS[i], width=1)))
    
    fig.update_layout(title=title, xaxis_title="Time (UTC)", yaxis_title=ylabel,
                      height=height, template='plotly_white', hovermode='x unified')
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05))
    return fig


def create_psd_plot(frequencies, power, title="Power Spectral Density", height=500):
    """Create log-log PSD plot with reference slopes."""
    fig = go.Figure()
    
    fig.add_trace(go.Scattergl(x=frequencies, y=power, mode='lines', name='PSD',
                                line=dict(color=COLORS[0], width=1.5)))
    
    # Add Kolmogorov -5/3 slope reference
    if len(frequencies) > 2:
        f_min, f_max = frequencies[frequencies > 0].min(), frequencies.max()
        f_mid = np.sqrt(f_min * f_max)
        p_mid = np.interp(f_mid, frequencies, power)
        
        f_ref = np.array([f_mid / 10, f_mid * 10])
        p_ref = p_mid * (f_ref / f_mid) ** (-5/3)
        fig.add_trace(go.Scatter(x=f_ref, y=p_ref, mode='lines', name='f^{-5/3}',
                                  line=dict(dash='dash', width=1.5, color=COLORS[1])))
    
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="PSD",
                      height=height, template='plotly_white', 
                      xaxis_type='log', yaxis_type='log')
    return fig


def create_pdf_plot(bin_centers, density, title="PDF", xlabel="Value", log_y=False, height=450):
    """Create PDF histogram plot."""
    fig = go.Figure()
    
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1
    fig.add_trace(go.Bar(x=bin_centers, y=density, name='PDF', 
                         marker_color=COLORS[0], opacity=0.7, width=bin_width * 0.9))
    
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Probability Density",
                      height=height, template='plotly_white', bargap=0.1)
    
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
        "N samples": f"{stats.n_samples:,}",
        "N NaN": f"{stats.n_nan:,}"
    }
