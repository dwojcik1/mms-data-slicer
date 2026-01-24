"""
MMS Turbulence Analysis Suite
==============================
Kinetic scale time series processing for space plasma physics.
"""

import streamlit as st
import streamlit.components.v1 as components

# MUST be first Streamlit command
st.set_page_config(
    page_title="MMS Turbulence Analysis Suite",
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
# Liquid Glass Landing Page (Full HTML/CSS)
# ============================================================================

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 35%, #0d1f3c 65%, #0a0a1a 100%);
    background-size: 400% 400%;
    animation: aurora 15s ease infinite;
    min-height: 100vh;
    color: #f8fafc;
    padding: 40px 30px;
}

@keyframes aurora {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero {
    text-align: center;
    padding: 20px 0 50px 0;
}

.hero-title {
    font-size: clamp(2.5rem, 6vw, 4rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 40%, #a5b4fc 70%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 40px rgba(129, 140, 248, 0.3));
    margin-bottom: 16px;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: clamp(1rem, 2vw, 1.4rem);
    font-weight: 400;
    color: rgba(248, 250, 252, 0.5);
    letter-spacing: 0.01em;
    max-width: 600px;
    margin: 0 auto;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1) 20%, rgba(255,255,255,0.1) 80%, transparent);
    margin: 30px auto;
    max-width: 800px;
}

.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 24px;
    max-width: 1300px;
    margin: 0 auto;
    padding: 0 10px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 32px;
    box-shadow: 
        0 0 0 1px rgba(255, 255, 255, 0.03) inset,
        0 20px 50px -15px rgba(0, 0, 0, 0.4);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}

.glass-card:hover {
    transform: translateY(-6px) scale(1.015);
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 
        0 0 0 1px rgba(255, 255, 255, 0.08) inset,
        0 30px 60px -15px rgba(0, 0, 0, 0.5),
        0 0 80px -20px rgba(99, 102, 241, 0.2);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: rgba(248, 250, 252, 0.95);
    margin-bottom: 14px;
    letter-spacing: -0.01em;
}

.card-body {
    font-size: 0.95rem;
    line-height: 1.7;
    color: rgba(248, 250, 252, 0.5);
}

.card-body strong {
    color: rgba(165, 180, 252, 0.9);
    font-weight: 500;
}

.footer {
    text-align: center;
    padding: 50px 20px 20px 20px;
    color: rgba(248, 250, 252, 0.3);
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
</style>
</head>
<body>

<div class="hero">
    <div class="hero-title">MMS Turbulence Analysis Suite</div>
    <div class="hero-subtitle">Kinetic scale time series processing for space plasma physics</div>
</div>

<div class="divider"></div>

<div class="grid-container">
    <div class="glass-card">
        <div class="card-title">Spectral Analysis</div>
        <div class="card-body">
            <strong>Welch PSD</strong> estimation with configurable windowing. 
            Spectral indices <strong>α</strong> analysis across inertial and kinetic ranges.
            Reference slopes for Kolmogorov and kinetic turbulence.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Stochastic Dynamics</div>
        <div class="card-body">
            <strong>PDFs & Moments</strong> computation for statistical characterization.
            Quantification of non-Gaussianity via Kurtosis <strong>κ</strong> and Skewness <strong>S</strong>.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Dissipation Proxies</div>
        <div class="card-body">
            <strong>J·E'</strong> analysis for energy conversion rates.
            Detection of EDR/IDR signatures, Hall fields, and reconnection events.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Coherent Structures</div>
        <div class="card-body">
            <strong>PVI Method</strong> (Partial Variance of Increments) for discontinuity detection.
            Identification of current sheets, flux ropes, and dipolarization fronts.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Wave Modes</div>
        <div class="card-body">
            <strong>Compressibility</strong> and magnetic helicity analysis.
            Ratio δB⊥/δB∥ for mode classification. Kinetic Alfvén Waves (KAW) signatures.
        </div>
    </div>
    
    <div class="glass-card">
        <div class="card-title">Signal Integrity</div>
        <div class="card-body">
            <strong>Stationarity tests</strong> (ADF) for time series validation.
            Outlier despiking algorithms and linear interpolation for data gaps.
        </div>
    </div>
</div>

<div class="divider"></div>

<div class="footer">Select a dataset from the sidebar to begin</div>

</body>
</html>
"""


# ============================================================================
# Cached Functions
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
# DataFrame Analysis (for PySPEDAS downloaded data)
# ============================================================================

def render_dataframe_analysis(df, time_data):
    """Render analysis UI for DataFrame-based data (from PySPEDAS)."""
    
    st.sidebar.markdown("##### Analysis Mode")
    mode = st.sidebar.radio("", ["Time Series", "Spectral"], label_visibility="collapsed", key="df_mode")
    
    columns = list(df.columns)
    
    if mode == "Time Series":
        # Get metadata from session state for dynamic title
        info = st.session_state.get('download_info', {})
        probe = info.get('probe', '?')
        coord = info.get('coord', 'GSE').upper()
        
        # Generate dynamic publication-style title
        if len(df) > 0:
            date_str = df.index[0].strftime('%d %B %Y')
            start_time_str = df.index[0].strftime('%H:%M:%S')
            end_time_str = df.index[-1].strftime('%H:%M:%S')
            title = f"MMS {probe} | B | {coord} | {date_str} | {start_time_str} – {end_time_str}"
        else:
            title = "Magnetic Field Time Series"
        
        with st.sidebar.expander("Settings", expanded=True):
            sub = st.checkbox("Subsample", value=True, key="df_sub")
            pts = st.slider("Points", 1000, 50000, 15000, key="df_pts") if sub else len(df)
        
        # Subsample if needed
        if sub and len(df) > pts:
            step = len(df) // pts
            plot_df = df.iloc[::step]
        else:
            plot_df = df
        
        # Import the new publication-quality plotter
        from plots import plot_magnetic_field
        
        # Create unified magnetic field plot
        fig = plot_magnetic_field(plot_df, title=title, height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data summary
        st.caption(f"Displaying {len(plot_df):,} of {len(df):,} points | Sampling: {1/(df.index[1] - df.index[0]).total_seconds():.1f} Hz" if len(df) > 1 else "")

    
    else:
        st.markdown("### Spectral Analysis")
        
        with st.sidebar.expander("Variable", expanded=True):
            selected_col = st.selectbox("Column", columns, key="df_col")
        
        data = df[selected_col].values
        
        with st.sidebar.expander("Method", expanded=True):
            method = st.radio("", ["PSD", "PDF", "Summary"], horizontal=True, label_visibility="collapsed", key="df_method")
        
        cols = st.columns(3)
        cols[0].metric("Variable", selected_col)
        cols[1].metric("Samples", f"{len(data):,}")
        cols[2].metric("NaN", f"{np.isnan(data).sum():,}")
        st.divider()
        
        # Clean data
        clean_data = data[~np.isnan(data)]
        
        if method == "PSD":
            st.markdown(f"#### PSD: {selected_col}")
            try:
                psd = cached_psd(tuple(clean_data), tuple(time_data[:len(clean_data)].astype(np.int64)))
                fig = create_psd_plot(psd.frequencies, psd.power, title=f"PSD: {selected_col}",
                                      psd_units=r"$\mathrm{nT}^2/\mathrm{Hz}$")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Sampling: {psd.sampling_frequency:.2f} Hz | Segments: {psd.nperseg}")
            except Exception as e:
                st.error(str(e))
        
        elif method == "PDF":
            st.markdown(f"#### PDF: {selected_col}")
            c1, c2 = st.columns([3, 1])
            bins = c1.slider("Bins", 20, 200, 50, key="df_bins")
            logy = c2.checkbox("Log Y", key="df_logy")
            
            cp, cs = st.columns([2, 1])
            with cp:
                try:
                    pdf = cached_pdf(tuple(clean_data), bins)
                    fig = create_pdf_plot(pdf.bin_centers, pdf.density, xlabel=f"{selected_col} (nT)", log_y=logy)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))
            with cs:
                st.markdown("##### Statistics")
                try:
                    stats = cached_stats(tuple(clean_data))
                    for n, v in create_stats_display(stats).items():
                        st.metric(n, v)
                except Exception as e:
                    st.error(str(e))
        
        else:
            st.markdown(f"#### Summary: {selected_col}")
            step = max(1, len(time_data) // 8000)
            fig = create_time_series_plot(time_data[::step], data[::step], 
                                          title=selected_col, ylabel=f"{selected_col} (nT)", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Statistics")
                try:
                    stats = cached_stats(tuple(clean_data))
                    for n, v in list(create_stats_display(stats).items())[:6]:
                        st.text(f"{n}: {v}")
                except Exception as e:
                    st.error(str(e))
            with c2:
                st.markdown("##### PSD")
                try:
                    psd = cached_psd(tuple(clean_data), tuple(time_data[:len(clean_data)].astype(np.int64)))
                    fig = create_psd_plot(psd.frequencies, psd.power, 
                                          psd_units=r"$\mathrm{nT}^2/\mathrm{Hz}$", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


# ============================================================================
# Main Application
# ============================================================================


def main():
    # Apply dark theme CSS FIRST (before any content)
    apply_custom_css()
    
    # Initialize session state for downloaded data
    if 'downloaded_df' not in st.session_state:
        st.session_state.downloaded_df = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    
    # Sidebar
    st.sidebar.markdown("### Data Source")
    
    # Data source tabs
    source_mode = st.sidebar.radio(
        "Source", 
        ["Upload CDF", "Download from NASA"],
        label_visibility="collapsed",
        horizontal=True
    )
    
    uploaded_file = None
    loader = None
    time_data = None
    
    # ========================================================================
    # UPLOAD CDF MODE
    # ========================================================================
    if source_mode == "Upload CDF":
        st.sidebar.markdown("##### Upload CDF File")
        uploaded_file = st.sidebar.file_uploader("CDF", type=['cdf'], label_visibility="collapsed")
        
        if uploaded_file is None and st.session_state.downloaded_df is None:
            components.html(LANDING_PAGE_HTML, height=850, scrolling=True)
            return
        
        if uploaded_file is not None:
            st.session_state.data_source = 'cdf'
            try:
                loader = CDFLoader.from_uploaded_file(uploaded_file)
                st.sidebar.caption(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {e}")
                return
            
            time_data = loader.get_time_data()
            if time_data is None:
                st.error("No time variable found.")
                return
    
    # ========================================================================
    # DOWNLOAD FROM NASA MODE
    # ========================================================================
    else:
        st.sidebar.markdown("##### NASA CDAWeb Download")
        st.sidebar.caption("Powered by CDAWeb (NASA/GSFC)")
        
        # Check if cdasws is available
        try:
            from downloader import check_cdasws_available, load_fgm_cdasws, format_trange
            cdasws_ok = check_cdasws_available()
        except ImportError:
            cdasws_ok = False
        
        if not cdasws_ok:
            st.sidebar.warning("cdasws not installed. Run: pip install cdasws")
            if st.session_state.downloaded_df is None:
                components.html(LANDING_PAGE_HTML, height=850, scrolling=True)
            return
        
        # Time range selection
        st.sidebar.markdown("###### Time Range")
        
        from datetime import date, time, timedelta
        
        # Default to yesterday for 1 hour
        default_start = date.today() - timedelta(days=1)
        default_end = date.today() - timedelta(days=1)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=default_start, label_visibility="visible")
            start_time = st.time_input("Start Time", value=time(12, 0), label_visibility="visible")
        with col2:
            end_date = st.date_input("End Date", value=default_end, label_visibility="visible")
            end_time = st.time_input("End Time", value=time(12, 30), label_visibility="visible")
        
        # Configuration
        st.sidebar.markdown("###### Configuration")
        
        cfg_col1, cfg_col2 = st.sidebar.columns(2)
        with cfg_col1:
            probe = st.selectbox("Probe", ['1', '2', '3', '4'], index=0)
            data_rate = st.selectbox("Rate", ['srvy', 'brst', 'fast', 'slow'], index=0)
        with cfg_col2:
            level = st.selectbox("Level", ['l2', 'l1b', 'ql'], index=0)
            coord = st.selectbox("Coord", ['gse', 'gsm', 'dmpa', 'bcs'], index=0)
        
        # Download button
        if st.sidebar.button("Download FGM Data", type="primary", use_container_width=True):
            trange = format_trange(start_date, start_time, end_date, end_time)
            
            with st.spinner(f"Downloading MMS{probe} FGM data from NASA..."):
                try:
                    df = load_fgm_cdasws(
                        trange=trange,
                        probe=probe,
                        data_rate=data_rate,
                        level=level,
                        coord=coord
                    )
                    st.session_state.downloaded_df = df

                    st.session_state.data_source = 'pyspedas'
                    st.session_state.download_info = {
                        'probe': probe,
                        'data_rate': data_rate,
                        'level': level,
                        'coord': coord.upper(),
                        'trange': trange
                    }
                    st.sidebar.success(f"Downloaded {len(df):,} points")
                except Exception as e:
                    st.sidebar.error(f"Download failed: {e}")
                    return
        
        # Show current data info
        if st.session_state.downloaded_df is not None and st.session_state.data_source == 'pyspedas':
            info = st.session_state.get('download_info', {})
            st.sidebar.caption(
                f"MMS{info.get('probe', '?')} FGM {info.get('coord', '')} "
                f"({info.get('data_rate', '')}/{info.get('level', '')})"
            )
        
        # If no data yet, show landing page
        if st.session_state.downloaded_df is None:
            components.html(LANDING_PAGE_HTML, height=850, scrolling=True)
            return
    
    # ========================================================================
    # DATA ANALYSIS (Common to both modes)
    # ========================================================================
    
    # Determine data source
    if st.session_state.data_source == 'pyspedas' and st.session_state.downloaded_df is not None:
        # Use downloaded DataFrame
        df = st.session_state.downloaded_df
        time_data = df.index.values.astype('datetime64[ns]')
        
        # Create a simple wrapper for DataFrame-based analysis
        render_dataframe_analysis(df, time_data)
        return
    
    # CDF-based analysis continues below
    if loader is None:
        return
    
    if time_data is None:
        time_data = loader.get_time_data()
        if time_data is None:
            st.error("No time variable found.")
            return

    
    st.sidebar.markdown("##### Mode")
    mode = st.sidebar.radio("", ["Time Series", "Spectral"], label_visibility="collapsed")
    
    plottable_vars = loader.get_plottable_variables()
    if not plottable_vars:
        st.warning("No plottable variables.")
        return
    
    var_metadata = {}
    for var in plottable_vars:
        attrs = loader.get_variable_attributes(var)
        units = attrs.get('UNITS', '') if isinstance(attrs.get('UNITS'), str) else ''
        var_metadata[var] = cached_metadata(var, units)
    
    fmt = lambda x: var_metadata.get(x, {}).get('label', x)
    
    # ========================================================================
    # TIME SERIES MODE
    # ========================================================================
    if mode == "Time Series":
        st.markdown("### Time Series Inspector")
        
        with st.sidebar.expander("Settings", expanded=True):
            sel = st.multiselect("Variables", plottable_vars, 
                                 default=plottable_vars[:min(3, len(plottable_vars))], format_func=fmt)
            sub = st.checkbox("Subsample", value=True)
            pts = st.slider("Points", 1000, 50000, 10000) if sub else len(time_data)
        
        if not sel:
            st.caption("Select variables from the sidebar.")
            return
        
        for var in sel:
            data = loader.get_variable_data(var)
            if data is None: 
                continue
            meta = var_metadata[var]
            ylabel = f"{meta['short_label']} ({meta['units']})" if meta['units'] else meta['short_label']
            t, d = time_data, data
            if sub and len(time_data) > pts:
                step = len(time_data) // pts
                t = time_data[::step]
                d = data[::step] if len(data.shape) == 1 else data[::step, :]
            fig = create_time_series_plot(t, d, title=meta['label'], ylabel=ylabel, 
                                          component_labels=meta.get('components'))
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Metadata"):
            attrs = loader.get_global_attributes()
            for key, val in list(attrs.items())[:10]:
                st.text(f"{key}: {str(val)[:80]}")
    
    # ========================================================================
    # SPECTRAL ANALYSIS MODE
    # ========================================================================
    else:
        st.markdown("### Spectral Analysis")
        
        with st.sidebar.expander("Variable", expanded=True):
            cats = loader.get_physics_variables()
            non_empty = [k for k, v in cats.items() if v]
            if not non_empty:
                st.caption("No variables detected.")
                return
            cat = st.selectbox("Category", non_empty, 
                               format_func=lambda x: x.replace('_', ' ').title())
            var = st.selectbox("Variable", cats.get(cat, []), format_func=fmt)
        
        meta = var_metadata[var]
        vdata = loader.get_variable_data(var)
        vinfo = loader.classify_variable(var)
        
        if vdata is None:
            st.error("Failed to load variable.")
            return
        
        comp, comp_label = None, meta['short_label']
        if vinfo['type'] == 'vector':
            with st.sidebar.expander("Component", expanded=True):
                opts = ['X', 'Y', 'Z'][:vinfo['n_components']] + ['Magnitude']
                comp = st.radio("", opts, horizontal=True, label_visibility="collapsed")
            adata = extract_component(vdata, comp)
            cidx = {'X': 0, 'Y': 1, 'Z': 2, 'Magnitude': 3}.get(comp, 0)
            comp_label = meta['components'][cidx] if cidx < len(meta['components']) else comp
        else:
            adata = vdata
        
        with st.sidebar.expander("Method", expanded=True):
            method = st.radio("", ["PSD", "PDF", "Summary"], horizontal=True, label_visibility="collapsed")
        
        cols = st.columns(3)
        cols[0].metric("Variable", meta['short_label'])
        cols[1].metric("Samples", f"{len(adata):,}")
        cols[2].metric("Component", comp or "Scalar")
        st.divider()
        
        # PSD
        if method == "PSD":
            st.markdown(f"#### Power Spectral Density: {comp_label}")
            try:
                psd = cached_psd(tuple(adata.flatten()), 
                                 tuple(time_data.astype('datetime64[ns]').astype(np.int64)))
                fig = create_psd_plot(psd.frequencies, psd.power, 
                                      title=f"PSD: {meta['label']}", psd_units=meta['psd_units'])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Sampling: {psd.sampling_frequency:.2f} Hz | Segments: {psd.nperseg} pts")
            except Exception as e:
                st.error(str(e))
        
        # PDF
        elif method == "PDF":
            st.markdown(f"#### PDF: {comp_label}")
            c1, c2 = st.columns([3, 1])
            bins = c1.slider("Bins", 20, 200, 50)
            logy = c2.checkbox("Log Y")
            
            cp, cs = st.columns([2, 1])
            with cp:
                try:
                    pdf = cached_pdf(tuple(adata.flatten()), bins)
                    xlabel = f"{comp_label} ({meta['units']})" if meta['units'] else comp_label
                    fig = create_pdf_plot(pdf.bin_centers, pdf.density, xlabel=xlabel, log_y=logy)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))
            with cs:
                st.markdown("##### Statistics")
                try:
                    stats = cached_stats(tuple(adata.flatten()))
                    for n, v in create_stats_display(stats).items():
                        st.metric(n, v)
                except Exception as e:
                    st.error(str(e))
        
        # Summary
        else:
            st.markdown(f"#### Summary: {meta['label']}")
            step = max(1, len(time_data) // 8000)
            ylabel = f"{comp_label} ({meta['units']})" if meta['units'] else comp_label
            fig = create_time_series_plot(time_data[::step], adata[::step], 
                                          title=meta['label'], ylabel=ylabel, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Statistics")
                try:
                    stats = cached_stats(tuple(adata.flatten()))
                    for n, v in list(create_stats_display(stats).items())[:6]:
                        st.text(f"{n}: {v}")
                except Exception as e:
                    st.error(str(e))
            with c2:
                st.markdown("##### PSD Preview")
                try:
                    psd = cached_psd(tuple(adata.flatten()), 
                                     tuple(time_data.astype('datetime64[ns]').astype(np.int64)))
                    fig = create_psd_plot(psd.frequencies, psd.power, 
                                          psd_units=meta['psd_units'], height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


if __name__ == "__main__":
    main()
