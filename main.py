"""
MMS Data Slicer - Kinetic Scale Explorer
=========================================
Main Streamlit application for NASA MMS mission data analysis.
"""

import streamlit as st
import numpy as np

# Local imports
from utils import CDFLoader, extract_component
from physics import compute_psd_welch, compute_pdf, compute_statistics
from plots import create_time_series_plot, create_psd_plot, create_pdf_plot, create_stats_display

# Page config
st.set_page_config(page_title="MMS Data Slicer", page_icon="🛰️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; 
        background: linear-gradient(90deg, #1e3a5f, #3d7ab0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .sub-header {font-size: 1.1rem; color: #6c757d; margin-bottom: 1.5rem;}
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
    data = np.array(data_tuple)
    return compute_pdf(data, n_bins=n_bins)


@st.cache_data
def cached_stats(data_tuple):
    """Cached statistics computation."""
    data = np.array(data_tuple)
    return compute_statistics(data)


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
    st.sidebar.subheader("📁 Step 1: Upload CDF")
    uploaded_file = st.sidebar.file_uploader("Choose a CDF file", type=['cdf'])
    
    if uploaded_file is None:
        st.info("👆 Upload a CDF file to begin analysis.")
        show_welcome()
        return
    
    # Load CDF
    try:
        loader = CDFLoader.from_uploaded_file(uploaded_file)
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return
    
    # Get time data
    time_data = loader.get_time_data()
    if time_data is None:
        st.error("❌ No time variable found in CDF file.")
        return
    
    # Step 2: Navigation Mode
    st.sidebar.subheader("🧭 Step 2: Analysis Mode")
    mode = st.sidebar.radio("Select mode:", ["Raw Data Inspector", "Turbulence Analysis"])
    
    # Get available variables
    plottable_vars = loader.get_plottable_variables()
    
    if not plottable_vars:
        st.warning("⚠️ No plottable variables found.")
        return
    
    # ========================================================================
    # RAW DATA INSPECTOR
    # ========================================================================
    if mode == "Raw Data Inspector":
        st.subheader("📊 Raw Data Inspector")
        
        # Variable selection
        selected_vars = st.sidebar.multiselect(
            "Select variables to plot:",
            options=plottable_vars,
            default=plottable_vars[:3] if len(plottable_vars) >= 3 else plottable_vars
        )
        
        # Subsampling
        enable_subsample = st.sidebar.checkbox("Enable subsampling", value=True)
        max_points = st.sidebar.slider("Max points", 1000, 50000, 10000) if enable_subsample else len(time_data)
        
        if not selected_vars:
            st.warning("Select at least one variable from the sidebar.")
            return
        
        # Plot each selected variable
        for var in selected_vars:
            data = loader.get_variable_data(var)
            if data is None:
                continue
            
            # Get units
            attrs = loader.get_variable_attributes(var)
            units = attrs.get('UNITS', '')
            ylabel = f"{var} [{units}]" if units else var
            
            # Subsample if needed
            t_plot = time_data
            d_plot = data
            if enable_subsample and len(time_data) > max_points:
                step = len(time_data) // max_points
                t_plot = time_data[::step]
                d_plot = data[::step] if len(data.shape) == 1 else data[::step, :]
            
            fig = create_time_series_plot(t_plot, d_plot, title=var, ylabel=ylabel)
            st.plotly_chart(fig, use_container_width=True)
        
        # Metadata expander
        with st.expander("📋 Global Attributes"):
            attrs = loader.get_global_attributes()
            for key, val in list(attrs.items())[:15]:
                st.text(f"{key}: {str(val)[:100]}")
    
    # ========================================================================
    # TURBULENCE ANALYSIS
    # ========================================================================
    else:
        st.subheader("🌀 Turbulence Analysis")
        
        # Variable selector
        st.sidebar.subheader("📈 Step 3: Select Variable")
        
        # Categorize variables
        categories = loader.get_physics_variables()
        
        # Show categories
        category = st.sidebar.selectbox("Variable category:", 
            [k for k, v in categories.items() if v])
        
        category_vars = categories.get(category, [])
        if not category_vars:
            st.warning(f"No variables in category: {category}")
            return
        
        selected_var = st.sidebar.selectbox("Variable:", category_vars)
        
        # Get variable data and info
        var_data = loader.get_variable_data(selected_var)
        var_info = loader.classify_variable(selected_var)
        
        if var_data is None:
            st.error(f"Could not load variable: {selected_var}")
            return
        
        # Component selector for vectors
        component = None
        if var_info['type'] == 'vector':
            n_comp = var_info['n_components']
            options = ['X', 'Y', 'Z'][:n_comp] + ['Magnitude']
            component = st.sidebar.selectbox("Component:", options)
            analysis_data = extract_component(var_data, component)
        else:
            analysis_data = var_data
        
        # Method selector
        st.sidebar.subheader("🔬 Step 4: Analysis Method")
        method = st.sidebar.radio("Method:", 
            ["Power Spectral Density", "PDF & Moments", "Summary"])
        
        # Display info
        col1, col2, col3 = st.columns(3)
        col1.metric("Variable", selected_var)
        col2.metric("Data Points", f"{len(analysis_data):,}")
        col3.metric("Component", component if component else "Scalar")
        
        st.divider()
        
        # ====================================================================
        # PSD Analysis
        # ====================================================================
        if method == "Power Spectral Density":
            st.markdown("### 📉 Power Spectral Density (Welch Method)")
            
            try:
                # Convert to tuple for caching
                data_tuple = tuple(analysis_data.flatten())
                time_tuple = tuple(time_data.astype('datetime64[ns]').astype(np.int64))
                
                with st.spinner("Computing PSD..."):
                    psd_result = cached_psd(data_tuple, time_tuple)
                
                # Plot
                fig = create_psd_plot(psd_result.frequencies, psd_result.power,
                    title=f"PSD: {selected_var}" + (f" ({component})" if component else ""))
                st.plotly_chart(fig, use_container_width=True)
                
                # Info
                st.info(f"Sampling frequency: **{psd_result.sampling_frequency:.2f} Hz** | "
                       f"Segment length: **{psd_result.nperseg}** points")
                
            except Exception as e:
                st.error(f"PSD computation error: {e}")
        
        # ====================================================================
        # PDF & Moments
        # ====================================================================
        elif method == "PDF & Moments":
            st.markdown("### 📊 Probability Distribution & Statistical Moments")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                n_bins = st.slider("Number of bins:", 20, 200, 50)
                log_y = st.checkbox("Log Y-axis", value=False)
                
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    
                    with st.spinner("Computing PDF..."):
                        pdf_result = cached_pdf(data_tuple, n_bins)
                    
                    # Get units for label
                    attrs = loader.get_variable_attributes(selected_var)
                    units = attrs.get('UNITS', '')
                    xlabel = f"{selected_var} [{units}]" if units else selected_var
                    
                    fig = create_pdf_plot(pdf_result.bin_centers, pdf_result.density,
                        title=f"PDF: {selected_var}" + (f" ({component})" if component else ""),
                        xlabel=xlabel, log_y=log_y)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"PDF computation error: {e}")
            
            with col2:
                st.markdown("#### Statistical Moments")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    stats = cached_stats(data_tuple)
                    
                    for name, value in create_stats_display(stats).items():
                        st.metric(name, value)
                        
                except Exception as e:
                    st.error(f"Statistics error: {e}")
        
        # ====================================================================
        # Summary
        # ====================================================================
        else:
            st.markdown("### 📋 Analysis Summary")
            
            # Time series
            st.markdown("#### Time Series")
            attrs = loader.get_variable_attributes(selected_var)
            units = attrs.get('UNITS', '')
            
            # Subsample for display
            step = max(1, len(time_data) // 10000)
            t_plot = time_data[::step]
            d_plot = analysis_data[::step]
            
            fig = create_time_series_plot(t_plot, d_plot, 
                title=f"{selected_var}" + (f" ({component})" if component else ""),
                ylabel=f"{selected_var} [{units}]" if units else selected_var)
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats and PSD side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Statistics")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    stats = cached_stats(data_tuple)
                    for name, value in create_stats_display(stats).items():
                        st.text(f"{name}: {value}")
                except Exception as e:
                    st.error(str(e))
            
            with col2:
                st.markdown("#### Quick PSD")
                try:
                    data_tuple = tuple(analysis_data.flatten())
                    time_tuple = tuple(time_data.astype('datetime64[ns]').astype(np.int64))
                    psd = cached_psd(data_tuple, time_tuple)
                    fig = create_psd_plot(psd.frequencies, psd.power, height=350)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


def show_welcome():
    """Display welcome information."""
    with st.expander("📖 About MMS Data Slicer", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Features
            - **Raw Data Inspector**: View time series of any variable
            - **Turbulence Analysis**: PSD, PDF, statistical moments
            - **Interactive Plots**: Zoom, pan, and explore
            """)
        with col2:
            st.markdown("""
            ### Supported Data
            - Magnetic field (FGM)
            - Electric field (EDP)
            - Ion/electron velocity, density
            - Temperature, pressure
            """)


if __name__ == "__main__":
    main()
