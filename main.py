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

# Apply responsive CSS
apply_custom_css()

# Additional minimal styling
st.markdown("""
<style>
    .scientific-header {
        font-size: clamp(1.8rem, 3.5vw, 2.8rem);
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .scientific-subtitle {
        font-size: clamp(1rem, 1.8vw, 1.2rem);
        color: #4a4a68;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #4a4a68;
        line-height: 1.5;
    }
    .footer-text {
        font-size: 0.9rem;
        color: #6a6a8a;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(.stContainer) {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Cached computation functions
# ============================================================================

@st.cache_data
def cached_psd(data_tuple, time_tuple):
    """Cached PSD computation."""
    data = np.array(data_tuple)
    time_data = np.array(time_tuple, dtype='datetime64[ns]')
    return compute_psd_welch(data, time_data)


@st.cache_data  
def cached_pdf(data_tuple, n_bins):
    """Cached PDF computation."""
    return compute_pdf(np.array(data_tuple), n_bins=n_bins)


@st.cache_data
def cached_stats(data_tuple):
    """Cached statistics computation."""
    return compute_statistics(np.array(data_tuple))


@st.cache_data
def cached_metadata(raw_name: str, units: str = '') -> dict:
    """Cache variable metadata lookup."""
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
    """Render the scientific landing page."""
    
    # Header
    st.markdown('<p class="scientific-header">Turbulence Analysis Suite</p>', unsafe_allow_html=True)
    st.markdown('<p class="scientific-subtitle">Time series processing for space plasma physics.</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Feature Grid - Row 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.latex(r"\mathcal{P}(f): \text{Spectral Density \& Scaling}")
            st.markdown(
                '<p class="card-desc">Welch method estimation of PSD. '
                'Analysis of spectral indices α in inertial and kinetic ranges.</p>',
                unsafe_allow_html=True
            )
    
    with col2:
        with st.container(border=True):
            st.latex(r"P(\delta B_\tau): \text{PDFs \& Moments}")
            st.markdown(
                '<p class="card-desc">Quantification of non-Gaussianity via Kurtosis κ '
                'and Skewness S. Intermittency analysis via structure functions.</p>',
                unsafe_allow_html=True
            )
    
    with col3:
        with st.container(border=True):
            st.latex(r"\mathbf{J} \cdot \mathbf{E}': \text{Dissipation Proxies}")
            st.markdown(
                '<p class="card-desc">Detection of dissipation regions (EDR/IDR signatures), '
                'Hall fields, and energy conversion events.</p>',
                unsafe_allow_html=True
            )
    
    # Feature Grid - Row 2
    col4, col5, col6 = st.columns(3)
    
    with col4:
        with st.container(border=True):
            st.latex(r"\text{PVI}: \text{Discontinuity Detection}")
            st.markdown(
                '<p class="card-desc">Partial Variance of Increments (PVI) method for identifying '
                'current sheets, flux ropes, and dipolarization fronts.</p>',
                unsafe_allow_html=True
            )
    
    with col5:
        with st.container(border=True):
            st.latex(r"\delta \mathbf{B}_{\perp} / \delta B_{\parallel}: \text{Wave Modes}")
            st.markdown(
                '<p class="card-desc">Compressibility analysis, magnetic helicity σ_m, '
                'and identification of Kinetic Alfvén Waves (KAW).</p>',
                unsafe_allow_html=True
            )
    
    with col6:
        with st.container(border=True):
            st.latex(r"\mathbf{X}(t): \text{Signal Integrity}")
            st.markdown(
                '<p class="card-desc">Stationarity tests (ADF), outlier removal (despiking), '
                'and linear interpolation of data gaps.</p>',
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # Footer
    st.markdown(
        '<p class="footer-text">Load L2/Burst data via the sidebar to initialize the analysis pipeline.</p>',
        unsafe_allow_html=True
    )


# ============================================================================
# Main Application
# ============================================================================

def main():
    # Sidebar
    st.sidebar.markdown("### Configuration")
    
    # File Upload
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
    
    # Get time data
    time_data = loader.get_time_data()
    if time_data is None:
        st.error("No time variable detected in CDF file.")
        return
    
    # Analysis Mode
    st.sidebar.markdown("##### Analysis Mode")
    mode = st.sidebar.radio("Mode", ["Time Series", "Spectral Analysis"], label_visibility="collapsed")
    
    # Get available variables
    plottable_vars = loader.get_plottable_variables()
    if not plottable_vars:
        st.warning("No plottable variables found in file.")
        return
    
    # Build metadata
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
            category = st.selectbox(
                "Category", 
                non_empty,
                format_func=lambda x: category_labels.get(x, x.title())
            )
            
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
        
        # Info bar
        cols = st.columns(3)
        cols[0].metric("Variable", meta['short_label'])
        cols[1].metric("Samples", f"{len(analysis_data):,}")
        cols[2].metric("Component", component if component else "Scalar")
        
        st.divider()
        
        # PSD
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
                
                st.caption(f"Sampling frequency: {psd_result.sampling_frequency:.2f} Hz | Segment length: {psd_result.nperseg} pts")
                
            except Exception as e:
                st.error(f"Computation error: {e}")
        
        # PDF
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
        
        # Summary
        else:
            st.markdown(f"#### Analysis Summary: {meta['label']}")
            
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
