"""
Turbulence Analysis Suite
==========================
Time series processing for space plasma physics.
"""

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Turbulence Analysis Suite",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto"
)

import numpy as np

# Local imports
from styles import apply_custom_css
from utils import (
    CDFLoader, extract_component, get_variable_metadata, 
    get_component_label, VariableMetadata
)
from physics import compute_psd_welch, compute_pdf, compute_statistics
from plots import create_time_series_plot, create_psd_plot, create_pdf_plot, create_stats_display


# ============================================================================
# Apple-Style Glassmorphism CSS
# ============================================================================

GLASS_CSS = """
<style>
/* Aurora Background */
.stApp {
    background: linear-gradient(135deg, 
        #0a0a1a 0%, 
        #1a1a3e 25%, 
        #0d1f3c 50%, 
        #1a0a2e 75%, 
        #0a0a1a 100%);
    background-size: 400% 400%;
    animation: aurora 20s ease infinite;
}

@keyframes aurora {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Hero Section */
.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #a8b5ff 50%, #7dd3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem;
    padding-top: 1rem;
}

.hero-subtitle {
    font-size: clamp(1.1rem, 2vw, 1.6rem);
    font-weight: 400;
    color: rgba(255, 255, 255, 0.65);
    text-align: center;
    margin-bottom: 3rem;
    letter-spacing: 0.01em;
}

/* Bento Grid Container */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    padding: 0 1rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 32px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    display: flex;
    flex-direction: column;
    min-height: 200px;
}

.glass-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 
        0 16px 48px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.15);
}

.card-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.95);
    margin-bottom: 12px;
    letter-spacing: -0.01em;
}

.card-description {
    font-size: 0.95rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.6);
    flex-grow: 1;
}

.card-description strong {
    color: rgba(168, 181, 255, 0.9);
    font-weight: 500;
}

/* Footer */
.glass-footer {
    text-align: center;
    padding: 3rem 1rem;
    color: rgba(255, 255, 255, 0.4);
    font-size: 1rem;
    letter-spacing: 0.02em;
}

/* Divider */
.glass-divider {
    height: 1px;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255, 255, 255, 0.1) 20%, 
        rgba(255, 255, 255, 0.1) 80%, 
        transparent);
    margin: 2rem auto;
    max-width: 800px;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(10, 10, 26, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] .stMarkdown {
    color: rgba(255, 255, 255, 0.8);
}

/* Override Streamlit defaults for dark theme */
.stApp [data-testid="stHeader"] {
    background: transparent;
}

.stApp .block-container {
    max-width: 100%;
    padding-top: 2rem;
}

/* Metric cards in analysis mode */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1rem;
}

[data-testid="stMetric"] label {
    color: rgba(255, 255, 255, 0.5) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: rgba(255, 255, 255, 0.9) !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    color: rgba(255, 255, 255, 0.8) !important;
}

/* Radio buttons */
.stRadio > div {
    background: transparent;
}

.stRadio label {
    color: rgba(255, 255, 255, 0.7) !important;
}
</style>
"""

BENTO_GRID_HTML = """
<div class="hero-title">Turbulence Analysis Suite</div>
<div class="hero-subtitle">Time series processing for space plasma physics</div>

<div class="glass-divider"></div>

<div class="bento-grid">
    <div class="glass-card">
        <div class="card-title">Spectral Analysis</div>
        <div class="card-description">
            <strong>Welch PSD</strong> estimation with configurable windowing. 
            Analysis of spectral indices <strong>α</strong> across inertial and kinetic ranges.
            Reference slopes for Kolmogorov and kinetic turbulence.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Stochastic Dynamics</div>
        <div class="card-description">
            <strong>PDFs & Moments</strong> computation. Quantification of non-Gaussianity 
            via Kurtosis <strong>κ</strong> and Skewness <strong>S</strong>. 
            Structure functions for intermittency analysis.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Dissipation Proxies</div>
        <div class="card-description">
            <strong>J·E'</strong> analysis for energy conversion. Detection of 
            EDR/IDR signatures, Hall fields, and magnetic reconnection events 
            in kinetic-scale plasmas.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Coherent Structures</div>
        <div class="card-description">
            <strong>PVI Method</strong> (Partial Variance of Increments) for 
            discontinuity detection. Identification of current sheets, 
            flux ropes, and dipolarization fronts.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Wave Modes</div>
        <div class="card-description">
            <strong>Compressibility</strong> analysis and magnetic helicity σ<sub>m</sub>. 
            Ratio δB<sub>⊥</sub>/δB<sub>∥</sub> for mode identification. 
            Kinetic Alfvén Wave (KAW) signatures.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Signal Integrity</div>
        <div class="card-description">
            <strong>Stationarity tests</strong> (ADF) for time series validation. 
            Outlier removal via despiking algorithms. 
            Linear interpolation for data gap handling.
        </div>
    </div>
</div>

<div class="glass-divider"></div>

<div class="glass-footer">
    Select a dataset from the sidebar to begin analysis
</div>
"""


# ============================================================================
# Cached computation functions
# ============================================================================

@st.cache_data
def cached_psd(data_tuple, time_tuple):
    data = np.array(data_tuple)
    time_data = np.array(time_tuple, dtype='datetime64[ns]')
    return compute_psd_welch(data, time_data)


@st.cache_data  
def cached_pdf(data_tuple, n_bins):
    return compute_pdf(np.array(data_tuple), n_bins=n_bins)


@st.cache_data
def cached_stats(data_tuple):
    return compute_statistics(np.array(data_tuple))


@st.cache_data
def cached_metadata(raw_name: str, units: str = '') -> dict:
    meta = get_variable_metadata(raw_name, units)
    return {
        'raw_name': meta.raw_name,
        'label': meta.label,
        'short_label': meta.short_label,
        'category': meta.category,
        'components': meta.components,
        'units': meta.units,
        'psd_units': meta.psd_units
    }


# ============================================================================
# Landing Page
# ============================================================================

def show_landing_page():
    """Render the glassmorphism landing page."""
    st.markdown(GLASS_CSS, unsafe_allow_html=True)
    st.markdown(BENTO_GRID_HTML, unsafe_allow_html=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    # Apply glass CSS globally
    st.markdown(GLASS_CSS, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### Configuration")
    st.sidebar.markdown("##### Data Input")
    uploaded_file = st.sidebar.file_uploader("CDF File", type=['cdf'], label_visibility="collapsed")
    
    if uploaded_file is None:
        show_landing_page()
        return
    
    # Load CDF
    try:
        loader = CDFLoader.from_uploaded_file(uploaded_file)
        st.sidebar.caption(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return
    
    time_data = loader.get_time_data()
    if time_data is None:
        st.error("No time variable detected in CDF file.")
        return
    
    st.sidebar.markdown("##### Analysis Mode")
    mode = st.sidebar.radio("Mode", ["Time Series", "Spectral Analysis"], label_visibility="collapsed")
    
    plottable_vars = loader.get_plottable_variables()
    if not plottable_vars:
        st.warning("No plottable variables found in file.")
        return
    
    var_metadata = {}
    for var in plottable_vars:
        attrs = loader.get_variable_attributes(var)
        units = attrs.get('UNITS', '') if isinstance(attrs.get('UNITS'), str) else ''
        var_metadata[var] = cached_metadata(var, units)
    
    def format_var_label(raw_name):
        meta = var_metadata.get(raw_name, {})
        return meta.get('label', raw_name)
    
    # ========================================================================
    # TIME SERIES MODE
    # ========================================================================
    if mode == "Time Series":
        st.markdown("### Time Series Inspector")
        
        with st.sidebar.expander("Settings", expanded=True):
            selected_vars = st.multiselect(
                "Variables", 
                plottable_vars,
                default=plottable_vars[:min(3, len(plottable_vars))],
                format_func=format_var_label
            )
            enable_subsample = st.checkbox("Subsample data", value=True)
            max_points = st.slider("Max points", 1000, 50000, 10000) if enable_subsample else len(time_data)
        
        if not selected_vars:
            st.caption("Select variables from the sidebar.")
            return
        
        for var in selected_vars:
            data = loader.get_variable_data(var)
            if data is None:
                continue
            
            meta = var_metadata[var]
            ylabel = f"{meta['short_label']} ({meta['units']})" if meta['units'] else meta['short_label']
            
            t_plot, d_plot = time_data, data
            if enable_subsample and len(time_data) > max_points:
                step = len(time_data) // max_points
                t_plot = time_data[::step]
                d_plot = data[::step] if len(data.shape) == 1 else data[::step, :]
            
            comp_labels = meta.get('components', None)
            fig = create_time_series_plot(t_plot, d_plot, title=meta['label'], 
                                          ylabel=ylabel, component_labels=comp_labels)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("File Metadata"):
            attrs = loader.get_global_attributes()
            for key, val in list(attrs.items())[:10]:
                st.text(f"{key}: {str(val)[:80]}")
    
    # ========================================================================
    # SPECTRAL ANALYSIS MODE
    # ========================================================================
    else:
        st.markdown("### Spectral Analysis")
        
        with st.sidebar.expander("Variable Selection", expanded=True):
            categories = loader.get_physics_variables()
            non_empty = [k for k, v in categories.items() if v]
            
            if not non_empty:
                st.caption("No physics variables detected.")
                return
            
            category_labels = {
                'magnetic_field': 'Magnetic Field',
                'electric_field': 'Electric Field',
                'velocity': 'Velocity',
                'density': 'Density',
                'temperature': 'Temperature',
                'pressure': 'Pressure',
                'other': 'Other'
            }
            category = st.selectbox("Category", non_empty,
                                    format_func=lambda x: category_labels.get(x, x.title()))
            
            category_vars = categories.get(category, [])
            selected_var = st.selectbox("Variable", category_vars, format_func=format_var_label)
        
        meta = var_metadata[selected_var]
        var_data = loader.get_variable_data(selected_var)
        var_info = loader.classify_variable(selected_var)
        
        if var_data is None:
            st.error(f"Failed to load: {selected_var}")
            return
        
        component = None
        component_label = meta['short_label']
        
        if var_info['type'] == 'vector':
            with st.sidebar.expander("Component", expanded=True):
                n_comp = var_info['n_components']
                options = ['X', 'Y', 'Z'][:n_comp] + ['Magnitude']
                component = st.radio("Select", options, horizontal=True, label_visibility="collapsed")
            
            analysis_data = extract_component(var_data, component)
            comp_idx = {'X': 0, 'Y': 1, 'Z': 2, 'Magnitude': 3}.get(component, 0)
            if comp_idx < len(meta['components']):
                component_label = meta['components'][comp_idx]
        else:
            analysis_data = var_data
        
        with st.sidebar.expander("Method", expanded=True):
            method = st.radio("Analysis", ["PSD", "PDF", "Summary"], horizontal=True, label_visibility="collapsed")
        
        cols = st.columns(3)
        cols[0].metric("Variable", meta['short_label'])
        cols[1].metric("Samples", f"{len(analysis_data):,}")
        cols[2].metric("Component", component if component else "Scalar")
        
        st.divider()
        
        if method == "PSD":
            st.markdown(f"#### Power Spectral Density: {component_label}")
            
            try:
                data_tuple = tuple(analysis_data.flatten())
                time_tuple = tuple(time_data.astype('datetime64[ns]').astype(np.int64))
                
                with st.spinner("Computing..."):
                    psd_result = cached_psd(data_tuple, time_tuple)
                
                title = f"PSD: {meta['label']}" + (f" ({component})" if component else "")
                fig = create_psd_plot(psd_result.frequencies, psd_result.power,
                                      title=title, psd_units=meta['psd_units'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"Sampling: {psd_result.sampling_frequency:.2f} Hz | Segments: {psd_result.nperseg} pts")
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        elif method == "PDF":
            st.markdown(f"#### Probability Distribution: {component_label}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                n_bins = st.slider("Bins", 20, 200, 50)
            with col2:
                log_y = st.checkbox("Log scale")
            
            col_pdf, col_stats = st.columns([2, 1])
            
            with col_pdf:
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    pdf_result = cached_pdf(data_tuple, n_bins)
                    
                    xlabel = f"{component_label} ({meta['units']})" if meta['units'] else component_label
                    title = f"PDF: {meta['short_label']}" + (f" ({component})" if component else "")
                    
                    fig = create_pdf_plot(pdf_result.bin_centers, pdf_result.density,
                                          title=title, xlabel=xlabel, log_y=log_y)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            with col_stats:
                st.markdown("##### Statistical Moments")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    stats = cached_stats(data_tuple)
                    for name, value in create_stats_display(stats).items():
                        st.metric(name, value)
                except Exception as e:
                    st.error(str(e))
        
        else:
            st.markdown(f"#### Summary: {meta['label']}")
            
            step = max(1, len(time_data) // 8000)
            t_plot = time_data[::step]
            d_plot = analysis_data[::step]
            
            ylabel = f"{component_label} ({meta['units']})" if meta['units'] else component_label
            fig = create_time_series_plot(t_plot, d_plot, title=meta['label'], ylabel=ylabel, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Statistics")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    stats = cached_stats(data_tuple)
                    for name, value in list(create_stats_display(stats).items())[:6]:
                        st.text(f"{name}: {value}")
                except Exception as e:
                    st.error(str(e))
            
            with col2:
                st.markdown("##### PSD")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    time_tuple = tuple(time_data.astype('datetime64[ns]').astype(np.int64))
                    psd = cached_psd(data_tuple, time_tuple)
                    fig = create_psd_plot(psd.frequencies, psd.power, 
                                          psd_units=meta['psd_units'], height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


if __name__ == "__main__":
    main()
