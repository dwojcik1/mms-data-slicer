"""
MMS Data Slicer - Kinetic Scale Explorer
=========================================
A lightweight, interactive viewer for CDF (Common Data Format) files,
designed for NASA MMS mission data analysis.

Author: Senior Python Developer & Space Physicist
Stack: Streamlit, cdflib, Plotly, Pandas, NumPy
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cdflib
from cdflib import cdfepoch
from datetime import datetime
import io
import tempfile
import os

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="MMS Data Slicer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS for Professional Look
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .upload-section {
        border: 2px dashed #3d7ab0;
        border-radius: 10px;
        padding: 1rem;
        background: rgba(61, 122, 176, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================

def detect_time_variable(cdf_file):
    """
    Automatically detect the time/epoch variable in the CDF file.
    Common names: Epoch, Epoch_TT2000, time, Time, unix_time
    """
    var_info = cdf_file.cdf_info()
    z_vars = getattr(var_info, 'zVariables', []) or []
    r_vars = getattr(var_info, 'rVariables', []) or []
    all_vars = list(z_vars) + list(r_vars)
    
    # Priority list for time variable names
    time_candidates = ['Epoch', 'Epoch_TT2000', 'epoch', 'EPOCH', 
                       'time', 'Time', 'TIME', 'unix_time', 'Unix_Time',
                       'tt2000', 'TT2000']
    
    for candidate in time_candidates:
        if candidate in all_vars:
            return candidate
    
    # Fallback: look for variables containing 'epoch' or 'time'
    for var in all_vars:
        if 'epoch' in var.lower() or 'time' in var.lower():
            return var
    
    return None


def convert_epoch_to_datetime(epoch_data, cdf_file, var_name):
    """
    Convert CDF epoch data to Python datetime objects.
    Handles TT2000, Epoch, and Epoch16 formats.
    """
    try:
        # Get variable info to determine epoch type
        var_info = cdf_file.varattsget(var_name)
        
        # Try to detect epoch type from data characteristics
        if epoch_data.dtype == np.int64:
            # TT2000 format (nanoseconds since J2000)
            datetimes = cdfepoch.to_datetime(epoch_data, to_np=True)
        elif epoch_data.dtype == np.float64:
            # Standard Epoch (milliseconds since 0 AD)
            datetimes = cdfepoch.to_datetime(epoch_data, to_np=True)
        else:
            # Generic conversion attempt
            datetimes = cdfepoch.to_datetime(epoch_data, to_np=True)
        
        return datetimes
    except Exception as e:
        st.warning(f"Epoch conversion warning: {e}. Using raw values.")
        return epoch_data


def get_1d_data_variables(cdf_file):
    """
    Extract 1D data variables from the CDF file.
    Filters out support data and focuses on actual measurement data.
    """
    var_info = cdf_file.cdf_info()
    z_vars = getattr(var_info, 'zVariables', []) or []
    
    data_vars = []
    time_var = detect_time_variable(cdf_file)
    
    for var in z_vars:
        try:
            # Skip the time variable
            if var == time_var:
                continue
            
            # Get variable attributes
            var_atts = cdf_file.varattsget(var)
            
            # Check if it's marked as support data
            var_type = var_atts.get('VAR_TYPE', ['Data'])[0] if isinstance(var_atts.get('VAR_TYPE', ['Data']), list) else var_atts.get('VAR_TYPE', 'Data')
            
            # Get data to check dimensions
            data = cdf_file.varget(var)
            if data is None:
                continue
            
            # Only include 1D or 2D variables (time series)
            if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] <= 4):
                # Prefer 'data' type but include others if they look like measurements
                if var_type == 'data' or 'Data' in str(var_type):
                    data_vars.append(var)
                elif any(key in var.lower() for key in ['b_', 'bx', 'by', 'bz', 'fgm', 'density', 'velocity', 'temp', 'pressure']):
                    data_vars.append(var)
        except Exception:
            continue
    
    return data_vars


def get_default_selected_vars(available_vars):
    """
    Pre-select typical magnetic field and plasma variables.
    """
    priority_patterns = [
        'b_gse', 'b_gsm', 'fgm', 'bx', 'by', 'bz',
        'b_', 'mag', 'mms_fgm',
        'density', 'ni', 'ne', 'n_i', 'n_e',
        'velocity', 'vi', 've', 'v_'
    ]
    
    selected = []
    for var in available_vars:
        var_lower = var.lower()
        for pattern in priority_patterns:
            if pattern in var_lower:
                selected.append(var)
                break
    
    # Return first 3-5 matches or empty if none found
    return selected[:5] if selected else []


def subsample_data(data, max_points=10000):
    """
    Subsample data if it exceeds max_points to keep visualization fast.
    """
    if len(data) <= max_points:
        return data, 1
    
    step = len(data) // max_points
    return data[::step], step


def create_interactive_plot(time_data, selected_vars, cdf_file, enable_subsampling, max_points):
    """
    Create interactive Plotly figure with multiple subplots.
    """
    n_vars = len(selected_vars)
    
    if n_vars == 0:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=n_vars, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=selected_vars
    )
    
    # Color palette for multi-component variables
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, var in enumerate(selected_vars):
        try:
            data = cdf_file.varget(var)
            
            if data is None:
                continue
            
            # Get time data matching this variable's dimension
            time_subset = time_data
            
            # Apply subsampling if enabled
            if enable_subsampling and len(time_subset) > max_points:
                time_subset, step = subsample_data(time_subset, max_points)
                if len(data.shape) == 1:
                    data = data[::step]
                else:
                    data = data[::step, :]
            
            # Handle 1D and 2D data
            if len(data.shape) == 1:
                # Single component
                fig.add_trace(
                    go.Scattergl(
                        x=time_subset,
                        y=data,
                        mode='lines',
                        name=var,
                        line=dict(color=colors[0], width=1),
                        hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
                    ),
                    row=idx + 1, col=1
                )
            else:
                # Multiple components (e.g., Bx, By, Bz)
                n_components = data.shape[1]
                component_labels = ['X', 'Y', 'Z', 'Total'][:n_components]
                
                for comp_idx in range(min(n_components, 4)):
                    fig.add_trace(
                        go.Scattergl(
                            x=time_subset,
                            y=data[:, comp_idx],
                            mode='lines',
                            name=f"{var}_{component_labels[comp_idx]}",
                            line=dict(color=colors[comp_idx], width=1),
                            legendgroup=var,
                            showlegend=True,
                            hovertemplate=f'{component_labels[comp_idx]}: ' + '%{y:.4f}<extra></extra>'
                        ),
                        row=idx + 1, col=1
                    )
            
            # Get units if available
            try:
                var_atts = cdf_file.varattsget(var)
                units = var_atts.get('UNITS', [''])[0] if isinstance(var_atts.get('UNITS', ['']), list) else var_atts.get('UNITS', '')
                fig.update_yaxes(title_text=f"{var} [{units}]" if units else var, row=idx + 1, col=1)
            except:
                fig.update_yaxes(title_text=var, row=idx + 1, col=1)
                
        except Exception as e:
            st.warning(f"Could not plot variable '{var}': {e}")
    
    # Update layout
    fig.update_layout(
        height=250 * n_vars,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=80, r=40, t=60, b=60)
    )
    
    # Update x-axis on the bottom subplot
    fig.update_xaxes(title_text="Time (UTC)", row=n_vars, col=1)
    
    # Enable range slider on x-axis for time navigation
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        row=n_vars, col=1
    )
    
    return fig


def display_global_attributes(cdf_file):
    """
    Display global attributes from the CDF file.
    """
    try:
        global_atts = cdf_file.globalattsget()
        
        # Key attributes to display prominently
        key_attrs = {
            'Project': global_atts.get('Project', 'N/A'),
            'Source_name': global_atts.get('Source_name', 'N/A'),
            'Discipline': global_atts.get('Discipline', 'N/A'),
            'Data_type': global_atts.get('Data_type', 'N/A'),
            'Descriptor': global_atts.get('Descriptor', 'N/A'),
            'Instrument_type': global_atts.get('Instrument_type', 'N/A'),
            'Mission_group': global_atts.get('Mission_group', 'N/A'),
            'Logical_source': global_atts.get('Logical_source', 'N/A'),
            'Logical_source_description': global_atts.get('Logical_source_description', 'N/A'),
        }
        
        return key_attrs, global_atts
    except Exception as e:
        st.error(f"Error reading global attributes: {e}")
        return {}, {}


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🛰️ MMS Data Slicer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Kinetic Scale Explorer | NASA MMS Mission Data Visualization</p>', unsafe_allow_html=True)
    
    # Brief description
    st.markdown("""
    **Welcome to MMS Data Slicer** — A lightweight, interactive viewer for CDF (Common Data Format) files 
    from the NASA Magnetospheric Multiscale (MMS) mission. Quickly visualize magnetic field fluctuations 
    ($B_x, B_y, B_z$), plasma density, and other kinetic-scale turbulence data.
    """)
    
    st.divider()
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File Upload
    st.sidebar.subheader("📁 Upload CDF File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CDF file",
        type=['cdf'],
        help="Upload a .cdf file from MMS, CDAWeb, or other compatible sources."
    )
    
    # Processing options
    st.sidebar.subheader("📊 Processing Options")
    enable_subsampling = st.sidebar.toggle(
        "Enable Sub-sampling",
        value=True,
        help="Reduces plot points for large files (>10k points) to improve performance."
    )
    
    max_points = st.sidebar.slider(
        "Max Points (if subsampling)",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        disabled=not enable_subsampling,
        help="Maximum number of points to display when subsampling is enabled."
    )
    
    # Main content area
    if uploaded_file is None:
        # Show placeholder content when no file is uploaded
        st.info("👆 Upload a CDF file using the sidebar to get started.")
        
        with st.expander("📖 About CDF Files & MMS Mission", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### What is a CDF file?
                The Common Data Format (CDF) is a self-describing data format 
                developed by NASA for storing and manipulating scalar and 
                multidimensional data. It's widely used in space physics for:
                
                - Magnetic field measurements
                - Plasma particle distributions
                - Electric field data
                - Ephemeris information
                """)
            
            with col2:
                st.markdown("""
                ### About MMS Mission
                The Magnetospheric Multiscale (MMS) mission is a NASA 
                robotic space mission studying Earth's magnetosphere. 
                Key instruments include:
                
                - **FGM**: Fluxgate Magnetometer
                - **FPI**: Fast Plasma Investigation
                - **EDP**: Electric Double Probes
                - **SCM**: Search Coil Magnetometer
                """)
        
        st.markdown("""
        ### Quick Start
        1. Download MMS data from [LASP MMS SDC](https://lasp.colorado.edu/mms/sdc/) or [CDAWeb](https://cdaweb.gsfc.nasa.gov/)
        2. Upload the `.cdf` file using the sidebar
        3. Select variables to visualize
        4. Explore your data with interactive plots!
        """)
        
        return
    
    # Process uploaded file
    try:
        # Save uploaded file to temp location (cdflib needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Open CDF file
        try:
            cdf_file = cdflib.CDF(tmp_path)
        except Exception as e:
            st.error(f"❌ Error opening CDF file: {e}")
            st.info("Please ensure the file is a valid CDF format. Common issues include corrupted downloads or incompatible CDF versions.")
            os.unlink(tmp_path)
            return
        
        # Display file info
        st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
        
        # Get file metadata
        cdf_info = cdf_file.cdf_info()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            z_vars_list = getattr(cdf_info, 'zVariables', []) or []
            st.metric("📊 Z-Variables", len(z_vars_list))
        with col2:
            r_vars_list = getattr(cdf_info, 'rVariables', []) or []
            st.metric("📈 R-Variables", len(r_vars_list))
        with col3:
            time_var = detect_time_variable(cdf_file)
            if time_var:
                time_data = cdf_file.varget(time_var)
                st.metric("⏱️ Time Points", f"{len(time_data):,}")
            else:
                st.metric("⏱️ Time Points", "N/A")
        with col4:
            file_size = uploaded_file.size / 1024 / 1024
            st.metric("💾 File Size", f"{file_size:.2f} MB")
        
        st.divider()
        
        # Metadata Expander
        with st.expander("📋 Global Attributes (Metadata)", expanded=False):
            key_attrs, all_attrs = display_global_attributes(cdf_file)
            
            if key_attrs:
                # Display key attributes in a nice format
                cols = st.columns(3)
                for idx, (key, value) in enumerate(key_attrs.items()):
                    with cols[idx % 3]:
                        if isinstance(value, (list, np.ndarray)):
                            value = ', '.join(str(v) for v in value)
                        st.markdown(f"**{key}:**")
                        st.text(str(value)[:100] + "..." if len(str(value)) > 100 else str(value))
                
                # Show all attributes in a table
                st.markdown("---")
                st.markdown("**All Global Attributes:**")
                attr_df = pd.DataFrame([
                    {"Attribute": k, "Value": str(v)[:200]} 
                    for k, v in all_attrs.items()
                ])
                st.dataframe(attr_df, use_container_width=True, hide_index=True)
        
        # Check for time variable
        time_var = detect_time_variable(cdf_file)
        
        if time_var is None:
            st.error("❌ Could not detect a time/epoch variable in this CDF file.")
            st.info("Expected variable names: 'Epoch', 'Epoch_TT2000', 'time', etc.")
            os.unlink(tmp_path)
            return
        
        # Load and convert time data
        try:
            time_data_raw = cdf_file.varget(time_var)
            time_data = convert_epoch_to_datetime(time_data_raw, cdf_file, time_var)
        except Exception as e:
            st.error(f"❌ Error reading time variable '{time_var}': {e}")
            os.unlink(tmp_path)
            return
        
        # Get available data variables
        data_vars = get_1d_data_variables(cdf_file)
        
        if not data_vars:
            # Fallback: show all z-variables
            all_z_vars = getattr(cdf_info, 'zVariables', []) or []
            data_vars = [v for v in all_z_vars if v != time_var]
        
        if not data_vars:
            st.warning("⚠️ No plottable data variables found in this CDF file.")
            os.unlink(tmp_path)
            return
        
        # Variable Selection in Sidebar
        st.sidebar.subheader("📈 Variable Selection")
        
        default_selected = get_default_selected_vars(data_vars)
        
        selected_vars = st.sidebar.multiselect(
            "Select variables to plot",
            options=data_vars,
            default=default_selected if default_selected else data_vars[:3],
            help="Choose which variables to visualize. Multi-component variables (e.g., B_GSE) will show all components."
        )
        
        # Quick select buttons
        st.sidebar.markdown("**Quick Select:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🧲 Magnetic", use_container_width=True):
                mag_vars = [v for v in data_vars if any(p in v.lower() for p in ['b_', 'fgm', 'mag', 'bx', 'by', 'bz'])]
                if mag_vars:
                    st.session_state['selected_vars'] = mag_vars[:5]
                    st.rerun()
        with col2:
            if st.button("🔬 Plasma", use_container_width=True):
                plasma_vars = [v for v in data_vars if any(p in v.lower() for p in ['density', 'ni', 'ne', 'velocity', 'temp', 'pressure'])]
                if plasma_vars:
                    st.session_state['selected_vars'] = plasma_vars[:5]
                    st.rerun()
        
        # Display time range info
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📅 Data Time Range:**")
        try:
            start_time = time_data[0]
            end_time = time_data[-1]
            st.sidebar.text(f"Start: {start_time}")
            st.sidebar.text(f"End: {end_time}")
        except:
            st.sidebar.text("Could not determine time range")
        
        # Visualization Section
        st.subheader("📊 Interactive Visualization")
        
        if not selected_vars:
            st.warning("⚠️ Please select at least one variable to plot from the sidebar.")
        else:
            with st.spinner("Generating interactive plots..."):
                fig = create_interactive_plot(
                    time_data, 
                    selected_vars, 
                    cdf_file, 
                    enable_subsampling, 
                    max_points
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Subsampling info
                    if enable_subsampling and len(time_data) > max_points:
                        step = len(time_data) // max_points
                        st.info(f"📉 Data subsampled: showing every {step}th point ({max_points:,} of {len(time_data):,} total points)")
                else:
                    st.error("❌ Could not generate plot. Please check your variable selections.")
        
        # Variable Details Expander
        with st.expander("🔍 Variable Details", expanded=False):
            for var in selected_vars:
                try:
                    var_atts = cdf_file.varattsget(var)
                    data = cdf_file.varget(var)
                    
                    st.markdown(f"### {var}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Shape:** `{data.shape}`")
                    with col2:
                        units = var_atts.get('UNITS', 'N/A')
                        st.markdown(f"**Units:** {units}")
                    with col3:
                        var_type = var_atts.get('VAR_TYPE', 'N/A')
                        st.markdown(f"**Type:** {var_type}")
                    
                    # Show description if available
                    catdesc = var_atts.get('CATDESC', var_atts.get('FIELDNAM', 'No description available'))
                    st.markdown(f"**Description:** {catdesc}")
                    
                    st.markdown("---")
                except Exception as e:
                    st.warning(f"Could not load details for '{var}': {e}")
        
        # Data Export Section
        with st.expander("💾 Export Data", expanded=False):
            st.markdown("Export selected variables to CSV format for further analysis.")
            
            if selected_vars and st.button("📥 Generate CSV"):
                with st.spinner("Preparing data for export..."):
                    try:
                        export_data = {'Time': time_data}
                        
                        for var in selected_vars:
                            data = cdf_file.varget(var)
                            if len(data.shape) == 1:
                                export_data[var] = data
                            else:
                                for i in range(min(data.shape[1], 4)):
                                    component = ['X', 'Y', 'Z', 'T'][i]
                                    export_data[f"{var}_{component}"] = data[:, i]
                        
                        df = pd.DataFrame(export_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv,
                            file_name=f"{uploaded_file.name.replace('.cdf', '')}_export.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Export error: {e}")
        
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.info("Please try uploading a different file or check the file format.")
        import traceback
        with st.expander("🐛 Debug Info"):
            st.code(traceback.format_exc())


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
