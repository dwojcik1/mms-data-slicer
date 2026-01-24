"""
MMS Data Slicer - Kinetic Scale Explorer
=========================================
Main Streamlit application with LaTeX variable labels.
"""

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="MMS Data Slicer",
    page_icon="🛰️",
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
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🛰️ MMS Data Slicer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Kinetic Scale Explorer | Turbulence Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Step 1: File Upload
    st.sidebar.subheader("📁 Upload CDF")
    uploaded_file = st.sidebar.file_uploader("Choose a CDF file", type=['cdf'], label_visibility="collapsed")
    
    if uploaded_file is None:
        st.info("👆 Upload a CDF file using the sidebar to begin analysis.")
        show_welcome()
        return
    
    # Load CDF
    try:
        loader = CDFLoader.from_uploaded_file(uploaded_file)
        st.sidebar.success(f"✅ {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return
    
    # Get time data
    time_data = loader.get_time_data()
    if time_data is None:
        st.error("❌ No time variable found.")
        return
    
    # Step 2: Navigation Mode
    st.sidebar.subheader("🧭 Analysis Mode")
    mode = st.sidebar.radio("Mode:", ["📊 Raw Data", "🌀 Turbulence"], label_visibility="collapsed")
    
    # Get available variables and build metadata cache
    plottable_vars = loader.get_plottable_variables()
    if not plottable_vars:
        st.warning("⚠️ No plottable variables found.")
        return
    
    # Build metadata for all variables
    var_metadata = {}
    for var in plottable_vars:
        attrs = loader.get_variable_attributes(var)
        units = attrs.get('UNITS', '') if isinstance(attrs.get('UNITS'), str) else ''
        var_metadata[var] = cached_metadata(var, units)
    
    # Format function for selectbox - shows LaTeX label
    def format_var_label(raw_name):
        meta = var_metadata.get(raw_name, {})
        return meta.get('label', raw_name)
    
    # ========================================================================
    # RAW DATA INSPECTOR
    # ========================================================================
    if mode == "📊 Raw Data":
        st.subheader("📊 Raw Data Inspector")
        
        with st.sidebar.expander("⚙️ Settings", expanded=True):
            selected_vars = st.multiselect(
                "Variables:", 
                plottable_vars,
                default=plottable_vars[:min(3, len(plottable_vars))],
                format_func=format_var_label
            )
            enable_subsample = st.checkbox("Subsample", value=True)
            max_points = st.slider("Max points", 1000, 50000, 10000) if enable_subsample else len(time_data)
        
        if not selected_vars:
            st.warning("Select variables from Settings.")
            return
        
        for var in selected_vars:
            data = loader.get_variable_data(var)
            if data is None:
                continue
            
            meta = var_metadata[var]
            ylabel = f"{meta['short_label']} ({meta['units']})" if meta['units'] else meta['short_label']
            
            # Subsample
            t_plot, d_plot = time_data, data
            if enable_subsample and len(time_data) > max_points:
                step = len(time_data) // max_points
                t_plot = time_data[::step]
                d_plot = data[::step] if len(data.shape) == 1 else data[::step, :]
            
            # Use LaTeX component labels
            comp_labels = meta.get('components', None)
            fig = create_time_series_plot(t_plot, d_plot, title=meta['label'], 
                                          ylabel=ylabel, component_labels=comp_labels)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Metadata"):
            attrs = loader.get_global_attributes()
            for key, val in list(attrs.items())[:10]:
                st.text(f"{key}: {str(val)[:80]}")
    
    # ========================================================================
    # TURBULENCE ANALYSIS
    # ========================================================================
    else:
        st.subheader("🌀 Turbulence Analysis")
        
        with st.sidebar.expander("⚙️ Variable Selection", expanded=True):
            # Group by category for cleaner selection
            categories = loader.get_physics_variables()
            non_empty = [k for k, v in categories.items() if v]
            
            if not non_empty:
                st.warning("No physics variables found.")
                return
            
            # Category selector with nice labels
            category_labels = {
                'magnetic_field': '🧲 Magnetic Field',
                'electric_field': '⚡ Electric Field',
                'velocity': '💨 Velocity',
                'density': '🔵 Density',
                'temperature': '🌡️ Temperature',
                'pressure': '📊 Pressure',
                'other': '📁 Other'
            }
            category = st.selectbox(
                "Category:", 
                non_empty,
                format_func=lambda x: category_labels.get(x, x.title())
            )
            
            category_vars = categories.get(category, [])
            
            # Variable selector with LaTeX labels
            selected_var = st.selectbox(
                "Variable:", 
                category_vars,
                format_func=format_var_label
            )
        
        # Get metadata for selected variable
        meta = var_metadata[selected_var]
        var_data = loader.get_variable_data(selected_var)
        var_info = loader.classify_variable(selected_var)
        
        if var_data is None:
            st.error(f"Could not load: {selected_var}")
            return
        
        # Component selector for vectors
        component = None
        component_label = meta['short_label']
        
        if var_info['type'] == 'vector':
            with st.sidebar.expander("📐 Component", expanded=True):
                n_comp = var_info['n_components']
                options = ['X', 'Y', 'Z'][:n_comp] + ['Magnitude']
                component = st.radio("Component:", options, horizontal=True)
            
            analysis_data = extract_component(var_data, component)
            
            # Get LaTeX label for component
            comp_idx = {'X': 0, 'Y': 1, 'Z': 2, 'Magnitude': 3}.get(component, 0)
            if comp_idx < len(meta['components']):
                component_label = meta['components'][comp_idx]
        else:
            analysis_data = var_data
        
        # Method selector
        with st.sidebar.expander("🔬 Analysis Method", expanded=True):
            method = st.radio("Method:", ["PSD", "PDF", "Summary"], horizontal=True)
        
        # Info metrics with LaTeX labels
        cols = st.columns(3)
        cols[0].metric("Variable", meta['short_label'])
        cols[1].metric("Points", f"{len(analysis_data):,}")
        cols[2].metric("Component", component if component else "Scalar")
        
        st.divider()
        
        # ====================================================================
        # PSD Analysis
        # ====================================================================
        if method == "PSD":
            st.markdown(f"### 📉 Power Spectral Density: {component_label}")
            
            try:
                data_tuple = tuple(analysis_data.flatten())
                time_tuple = tuple(time_data.astype('datetime64[ns]').astype(np.int64))
                
                with st.spinner("Computing PSD..."):
                    psd_result = cached_psd(data_tuple, time_tuple)
                
                # Use physics-aware units from metadata
                title = f"PSD: {meta['label']}" + (f" ({component})" if component else "")
                fig = create_psd_plot(
                    psd_result.frequencies, 
                    psd_result.power,
                    title=title,
                    psd_units=meta['psd_units']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"Sampling: **{psd_result.sampling_frequency:.2f} Hz** | Segments: **{psd_result.nperseg}** pts")
                
            except Exception as e:
                st.error(f"PSD error: {e}")
        
        # ====================================================================
        # PDF & Moments
        # ====================================================================
        elif method == "PDF":
            st.markdown(f"### 📊 Probability Distribution: {component_label}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                n_bins = st.slider("Bins:", 20, 200, 50)
            with col2:
                log_y = st.checkbox("Log Y")
            
            col_pdf, col_stats = st.columns([2, 1])
            
            with col_pdf:
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    pdf_result = cached_pdf(data_tuple, n_bins)
                    
                    xlabel = f"{component_label} ({meta['units']})" if meta['units'] else component_label
                    title = f"PDF: {meta['short_label']}" + (f" ({component})" if component else "")
                    
                    fig = create_pdf_plot(
                        pdf_result.bin_centers, 
                        pdf_result.density,
                        title=title, 
                        xlabel=xlabel, 
                        log_y=log_y
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"PDF error: {e}")
            
            with col_stats:
                st.markdown("#### Statistics")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    stats = cached_stats(data_tuple)
                    for name, value in create_stats_display(stats).items():
                        st.metric(name, value)
                except Exception as e:
                    st.error(str(e))
        
        # ====================================================================
        # Summary
        # ====================================================================
        else:
            st.markdown(f"### 📋 Summary: {meta['label']}")
            
            # Time series
            step = max(1, len(time_data) // 8000)
            t_plot = time_data[::step]
            d_plot = analysis_data[::step]
            
            ylabel = f"{component_label} ({meta['units']})" if meta['units'] else component_label
            fig = create_time_series_plot(t_plot, d_plot, title=meta['label'], ylabel=ylabel, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Statistics")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    stats = cached_stats(data_tuple)
                    for name, value in list(create_stats_display(stats).items())[:6]:
                        st.text(f"{name}: {value}")
                except Exception as e:
                    st.error(str(e))
            
            with col2:
                st.markdown("#### PSD Preview")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    time_tuple = tuple(time_data.astype('datetime64[ns]').astype(np.int64))
                    psd = cached_psd(data_tuple, time_tuple)
                    fig = create_psd_plot(psd.frequencies, psd.power, 
                                          psd_units=meta['psd_units'], height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


def show_welcome():
    """Display welcome info."""
    with st.expander("📖 About", expanded=True):
        st.markdown("""
        **MMS Data Slicer** visualizes NASA MMS mission CDF files with publication-quality labels.
        
        **Features:**
        - 📊 Raw Data Inspector - View time series with LaTeX notation
        - 🌀 Turbulence Analysis - PSD with physics units ($\\mathrm{nT}^2/\\mathrm{Hz}$)
        - 📱 Responsive design - Phone to 4K displays
        """)


if __name__ == "__main__":
    main()
