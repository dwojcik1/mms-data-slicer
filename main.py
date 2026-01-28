"""
MMS Turbulence Analysis Suite
==============================
Kinetic scale time series processing for space plasma physics.
"""

import streamlit as st
import streamlit.components.v1 as components

# MUST be first Streamlit command
st.set_page_config(
    page_title="MMS Turbulence Laboratory",
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
# Multi-Dataset Analysis (FGM or FPI with DIS/DES)
# ============================================================================

def render_multi_dataset_analysis(datasets: dict, info: dict):
    """
    Render analysis for multiple datasets (FGM single plot or FPI dual plots).
    
    Args:
        datasets: Dict with keys like 'FGM', 'DIS', 'DES' mapping to DataFrames
        info: Download info dict with probe, coord, instrument, etc.
    """
    from plots import plot_magnetic_field, plot_velocity_field, PLOTLY_CONFIG
    
    st.sidebar.markdown("##### Analysis Mode")
    mode = st.sidebar.radio("", ["Time Series", "Spectral"], label_visibility="collapsed", key="multi_mode")
    
    probe = info.get('probe', '?')
    coord = info.get('coord', 'GSE').upper()
    instrument = info.get('instrument', 'FGM')
    
    if mode == "Time Series":
        with st.sidebar.expander("Settings", expanded=True):
            sub = st.checkbox("Subsample", value=True, key="multi_sub")
            pts = st.slider("Points", 1000, 50000, 15000, key="multi_pts") if sub else 999999
        
        # Iterate through each dataset
        for key, df in datasets.items():
            if df is None or len(df) == 0:
                continue
            
            # Generate dynamic title
            start_dt = df.index[0]
            end_dt = df.index[-1]
            
            if start_dt.date() == end_dt.date():
                date_str = start_dt.strftime('%d %B %Y')
                time_range = f"{start_dt.strftime('%H:%M:%S')} – {end_dt.strftime('%H:%M:%S')}"
            else:
                date_str = f"{start_dt.strftime('%d %b %Y %H:%M')} – {end_dt.strftime('%d %b %Y %H:%M')}"
                time_range = ""
            
            # Subsample
            if sub and len(df) > pts:
                step = len(df) // pts
                plot_df = df.iloc[::step]
            else:
                plot_df = df
            
            # Determine plot type based on dataset key
            if key == 'FGM':
                # Magnetic field
                if time_range:
                    title = f"MMS {probe} | B | {coord} | {date_str} | {time_range}"
                else:
                    title = f"MMS {probe} | B | {coord} | {date_str}"
                fig = plot_magnetic_field(plot_df, title=title, height=550)
            elif key == 'DES':
                # Electron velocity
                if time_range:
                    title = f"MMS {probe} | Electron Bulk Velocity | {coord} | {date_str} | {time_range}"
                else:
                    title = f"MMS {probe} | V<sub>e</sub> | {coord} | {date_str}"
                fig = plot_velocity_field(plot_df, title=title, species='electron', height=450)
            elif key == 'DIS':
                # Ion velocity
                if time_range:
                    title = f"MMS {probe} | Ion Bulk Velocity | {coord} | {date_str} | {time_range}"
                else:
                    title = f"MMS {probe} | V<sub>i</sub> | {coord} | {date_str}"
                fig = plot_velocity_field(plot_df, title=title, species='ion', height=450)
            else:
                # Unknown - skip or use generic
                continue
            
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Data summary caption
            if len(df) > 1:
                try:
                    sampling_hz = 1 / (df.index[1] - df.index[0]).total_seconds()
                    st.caption(f"{key}: {len(plot_df):,} of {len(df):,} points | {sampling_hz:.1f} Hz")
                except:
                    st.caption(f"{key}: {len(plot_df):,} of {len(df):,} points")
    
    else:
        # Spectral analysis - let user pick which dataset
        st.markdown("### Spectral Analysis")
        
        dataset_keys = list(datasets.keys())
        selected_key = st.sidebar.selectbox("Dataset", dataset_keys, key="spectral_dataset")
        
        df = datasets[selected_key]
        columns = list(df.columns)
        
        with st.sidebar.expander("Variable", expanded=True):
            selected_col = st.selectbox("Column", columns, key="spectral_col")
        
        data = df[selected_col].values
        
        with st.sidebar.expander("Method", expanded=True):
            method = st.radio("", ["PSD", "PDF", "Summary"], horizontal=True, label_visibility="collapsed", key="spectral_method")
        
        # Show metrics
        cols = st.columns(3)
        cols[0].metric("Dataset", selected_key)
        cols[1].metric("Samples", f"{len(data):,}")
        cols[2].metric("NaN", f"{np.isnan(data).sum():,}")
        st.divider()
        
        clean_data = data[~np.isnan(data)]
        time_data = df.index.values.astype('datetime64[ns]')
        
        if method == "PSD":
            st.markdown(f"#### PSD: {selected_key} {selected_col}")
            try:
                psd = cached_psd(tuple(clean_data), tuple(time_data[:len(clean_data)].astype(np.int64)))
                fig = create_psd_plot(psd.frequencies, psd.power, title=f"PSD: {selected_key} {selected_col}",
                                      psd_units=r"nT²/Hz" if 'B' in selected_col else "km²/s²/Hz")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Sampling: {psd.sampling_frequency:.2f} Hz | Segments: {psd.nperseg}")
            except Exception as e:
                st.error(str(e))
        
        elif method == "PDF":
            st.markdown(f"#### PDF: {selected_key} {selected_col}")
            c1, c2 = st.columns([3, 1])
            bins = c1.slider("Bins", 20, 200, 50, key="spectral_bins")
            logy = c2.checkbox("Log Y", key="spectral_logy")
            
            try:
                pdf = cached_pdf(tuple(clean_data), bins)
                units = "nT" if 'B' in selected_col else "km/s"
                fig = create_pdf_plot(pdf.bin_centers, pdf.density, xlabel=f"{selected_col} ({units})", log_y=logy)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))
        
        else:
            st.markdown(f"#### Summary: {selected_key} {selected_col}")
            try:
                stats = cached_stats(tuple(clean_data))
                for n, v in create_stats_display(stats).items():
                    st.text(f"{n}: {v}")
            except Exception as e:
                st.error(str(e))


# ============================================================================
# DataFrame Analysis (legacy single DataFrame support)
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
        
        # Generate dynamic publication-style title (handles multi-day intervals)
        if len(df) > 0:
            start_dt = df.index[0]
            end_dt = df.index[-1]
            
            # Check if same day or multi-day interval
            if start_dt.date() == end_dt.date():
                # Same day: "MMS 1 | B | GSE | 11 January 2024 | 12:00:00 – 12:30:00"
                date_str = start_dt.strftime('%d %B %Y')
                time_range = f"{start_dt.strftime('%H:%M:%S')} – {end_dt.strftime('%H:%M:%S')}"
                title = f"MMS {probe} | B | {coord} | {date_str} | {time_range}"
            else:
                # Multi-day: "MMS 1 | B | GSE | 13 Mar 2021 23:00 – 14 Mar 2021 05:00"
                start_str = start_dt.strftime('%d %b %Y %H:%M')
                end_str = end_dt.strftime('%d %b %Y %H:%M')
                title = f"MMS {probe} | B | {coord} | {start_str} – {end_str}"
        else:
            title = "Magnetic Field Time Series"
        
        with st.sidebar.expander("Settings", expanded=True):
            sub = st.checkbox("Subsample", value=True, key="df_sub")
            pts = st.slider("Points", 1000, 50000, 15000, key="df_pts") if sub else len(df)
        
        # Subsample if needed
        if sub and len(df) > pts:
            step = len(df) // pts
            plot_df = df.iloc[::step]
            is_subsampled = True
        else:
            plot_df = df
            is_subsampled = False
        
        # Import the publication-quality plotter and config
        from plots import plot_magnetic_field, PLOTLY_CONFIG
        
        # Create unified magnetic field plot with white background
        fig = plot_magnetic_field(plot_df, title=title, height=580)
        
        # Render plot (using HTML labels in plots.py for reliable rendering)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Show data summary
        if len(df) > 1:
            sampling_hz = 1 / (df.index[1] - df.index[0]).total_seconds()
            st.caption(f"Displaying {len(plot_df):,} of {len(df):,} points | Original sampling: {sampling_hz:.1f} Hz")
        
        # Subsample info notice (icon only in parameter, not in text)
        if is_subsampled:
            st.info(
                "**Note:** To maintain performance, large datasets are subsampled. "
                "Adjust the **Points** slider in Settings (sidebar) to show more detail. "
                "Use the camera icon in the plot toolbar to export a high-resolution PNG.",
                icon=""
            )



    
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

# MMS Instrument catalog
MMS_INSTRUMENTS = {
    "Fluxgate Magnetometer (FGM)": {"key": "fgm", "active": True},
    "Fast Plasma Investigation (FPI)": {"key": "fpi", "active": True},
    "Search Coil Magnetometer (SCM)": {"key": "scm", "active": False},
    "FGM+SCM Data (FSM)": {"key": "fsm", "active": False},
    "Electric Field Double Probe (EDP)": {"key": "edp", "active": False},
    "Electron Drift Instrument (EDI)": {"key": "edi", "active": False},
    "Fly's Eye Energetic Particle Sensor (FEEPS)": {"key": "feeps", "active": False},
    "Energetic Ion Spectrometer (EIS)": {"key": "eis", "active": False},
    "Active Spacecraft Potential Control (ASPOC)": {"key": "aspoc", "active": False},
    "Hot Plasma Composition Analyzer (HPCA)": {"key": "hpca", "active": False},
    "Mission Ephemeris Coordinates (MEC)": {"key": "mec", "active": False},
    "Attitude and Ephemeris (STATE)": {"key": "state", "active": False},
    "Tetrahedron Quality Factor (TQF)": {"key": "tqf", "active": False},
}


def render_data_loader():
    """Render the main page data configuration wizard."""
    

    # Main title
    st.markdown("## Magnetospheric Multiscale (MMS) Turbulence Lab 🛰️")
    st.markdown("### Data Configuration 📡")
    st.caption("Select your data source and configure the download parameters.")
    
    st.markdown("")  # Spacer
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Download from NASA CDAWeb", "Upload own .CDF files"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    if data_source == "Download from NASA CDAWeb":
        render_nasa_download_form()
    else:
        render_upload_form()
    
    # Acknowledgments footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; opacity: 0.7; font-size: 0.85em; padding: 20px 0;">
            <strong>Acknowledgments:</strong> This work relies on efforts of the entire MMS mission team, 
            including development, science operations, and the Science Data Center at the University of Colorado.<br>
            <a href="http://doi.org/10.1007/s11214-015-0164-9" target="_blank" style="color: #818cf8;">
                J.L. Burch et al., Space Sci Rev (2016) — Mission Overview
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )




def render_nasa_download_form():
    """Render the NASA CDAWeb download configuration form."""
    
    # Check if cdasws is available
    try:
        from downloader import check_cdasws_available, load_fgm_cdasws, load_fpi_cdasws, format_trange
        cdasws_ok = check_cdasws_available()
    except ImportError:
        cdasws_ok = False
    
    if not cdasws_ok:
        st.error("**Required module not installed.** Run: `pip install cdasws cdflib`")
        return
    
    # Instrument selection
    st.markdown("### Instrument Selection")
    
    instrument_name = st.selectbox(
        "Select Instrument",
        list(MMS_INSTRUMENTS.keys()),
        index=0
    )
    
    instrument_info = MMS_INSTRUMENTS[instrument_name]
    instrument_key = instrument_info["key"]
    is_active = instrument_info["active"]
    
    if not is_active:
        st.info(f"**Support for {instrument_name} is coming soon.**\n\nCurrently available: FGM, FPI")
        return
    
    # Time Range section
    st.markdown("### Time Range")
    
    # Show latest data availability
    from utils import get_latest_data_time, get_mms_dataset_id
    dataset_id = get_mms_dataset_id('1', instrument_key, 'srvy' if instrument_key == 'fgm' else 'fast', 'l2')
    latest_time = get_latest_data_time(dataset_id)
    if latest_time:
        st.caption(f"📅 Latest data available up to: **{latest_time}**")
    
    from datetime import date, time, timedelta
    
    default_start = date.today() - timedelta(days=1)
    default_end = date.today() - timedelta(days=1)
    
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        start_date = st.date_input("Start Date", value=default_start)
    with t2:
        start_time = st.time_input("Start Time", value=time(12, 0))
    with t3:
        end_date = st.date_input("End Date", value=default_end)
    with t4:
        end_time = st.time_input("End Time", value=time(12, 30))

    
    # Configuration section
    st.markdown("### Configuration")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        probe = st.selectbox("Probe", ['1', '2', '3', '4'], index=0)
    with c2:
        if instrument_key == 'fgm':
            data_rate = st.selectbox("Data Rate", ['srvy', 'brst', 'fast', 'slow'], index=0)
        else:
            data_rate = st.selectbox("Data Rate", ['fast', 'brst'], index=0)
    with c3:
        level = st.selectbox("Level", ['l2'], index=0)
    with c4:
        coord = st.selectbox("Coordinates", ['gse', 'gsm'], index=0)
    
    st.markdown("")  # Spacer

    
    # Download button
    if st.button(f"Download {instrument_key.upper()} Data", type="primary", use_container_width=True):
        trange = format_trange(start_date, start_time, end_date, end_time)
        
        with st.spinner(f"Downloading MMS{probe} {instrument_key.upper()} data from NASA CDAWeb..."):
            try:
                if instrument_key == 'fgm':
                    df = load_fgm_cdasws(
                        trange=trange,
                        probe=probe,
                        data_rate=data_rate,
                        level=level,
                        coord=coord
                    )
                    st.session_state.data = {'FGM': df}
                else:
                    datasets = load_fpi_cdasws(
                        trange=trange,
                        probe=probe,
                        data_rate=data_rate,
                        level=level,
                        coord=coord
                    )
                    st.session_state.data = datasets
                
                st.session_state.data_loaded = True
                st.session_state.download_info = {
                    'probe': probe,
                    'data_rate': data_rate,
                    'level': level,
                    'coord': coord.upper(),
                    'trange': trange,
                    'instrument': instrument_key.upper()
                }
                
                total_pts = sum(len(v) for v in st.session_state.data.values())
                st.success(f"Downloaded {total_pts:,} data points ({len(st.session_state.data)} dataset(s))")
                st.rerun()
                
            except Exception as e:
                st.error(f"Download failed: {e}")
    
    # Footer disclaimer (centered)
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; opacity: 0.7; font-size: 0.85em;'>"
        "📝 <em>Note: Data will not be saved to your local device unless explicitly exported.</em>"
        "</p>",
        unsafe_allow_html=True
    )


def render_upload_form():
    """Render the CDF file upload form."""
    
    st.markdown("### 📁 Upload CDF File")
    
    uploaded_file = st.file_uploader(
        "Drop your .CDF file here",
        type=['cdf'],
        help="Upload a CDF file from the MMS mission or other space physics data sources."
    )
    
    if uploaded_file is not None:
        try:
            loader = CDFLoader.from_uploaded_file(uploaded_file)
            time_data = loader.get_time_data()
            
            if time_data is None:
                st.error("No time variable found in the CDF file.")
                return
            
            # Store in session state
            st.session_state.data = {'CDF': loader}
            st.session_state.data_loaded = True
            st.session_state.upload_info = {
                'filename': uploaded_file.name,
                'loader': loader
            }
            
            st.success(f"Loaded: {uploaded_file.name}")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error loading file: {e}")


def render_sidebar():
    """Render the sidebar with global controls and analysis navigation."""
    
    with st.sidebar:
        st.markdown("### Global Controls ⚙️")
        
        # Dynamic subsample control based on loaded data
        data = st.session_state.get('data', None)
        data_loaded = st.session_state.get('data_loaded', False)
        
        if data_loaded and data:
            total_len = sum(len(df) for df in data.values() if hasattr(df, '__len__'))
            max_pts = max(1000, total_len)
            default_pts = min(30000, total_len)
        else:
            max_pts = 100000
            default_pts = 30000
            total_len = 0
        
        # Single number input for subsample control
        subsample_pts = st.number_input(
            "Subsample Points",
            min_value=1000,
            max_value=max_pts,
            value=default_pts,
            step=1000,
            key="subsample_pts"
        )
        
        # Explanation text
        st.info(
            "**Note:** Large datasets are subsampled to maintain performance. "
            "Increasing this value improves resolution but increases memory usage.",
            icon="ℹ️"
        )
        
        st.divider()
        
        # Time Context Section (show duration from download info)
        info = st.session_state.get('download_info', {})
        trange = info.get('trange', None)
        if trange and len(trange) == 2:
            try:
                from datetime import datetime
                start_dt = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
                end_dt = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')
                duration = end_dt - start_dt
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                if duration.days > 0:
                    duration_str = f"{duration.days}d {hours}h {minutes}m"
                elif hours > 0:
                    duration_str = f"{hours}h {minutes}m {seconds}s"
                else:
                    duration_str = f"{minutes}m {seconds}s"
                st.info(f"**Duration:** {duration_str}")
            except:
                pass
        
        # Data Rate Warnings
        data_rate = info.get('data_rate', '')
        if data_rate:
            st.markdown("**Data Rate Info**")
            if data_rate == 'srvy':
                st.caption("SRVY cadence ≈ **4.5 s** (FGM) / **4.5 s** (FPI)")
            elif data_rate == 'brst':
                st.warning("BURST mode — not available for long durations")
            elif data_rate == 'fast':
                st.caption("FAST cadence ≈ **4.5 s**")
            st.divider()
        
        # Analysis mode navigation
        st.markdown("### Analysis Mode 📊")
        
        if data_loaded:
            analysis_mode = st.radio(
                "Select Analysis",
                ["Time Series Inspector", "Power Spectral Density", "PDF & Moments", "Summary Statistics"],
                label_visibility="collapsed",
                key="analysis_mode"
            )
        else:
            st.info("Load data to enable analysis")
            analysis_mode = None
        
        st.divider()

        
        # Data Export section (only when data is loaded)
        if data_loaded and data:
            st.markdown("### Export Data")
            
            dataset_keys = list(data.keys())
            export_dataset = st.selectbox("Dataset", dataset_keys, key="export_dataset")
            export_format = st.selectbox(
                "Format", 
                ["CSV (.csv)", "Text File (.txt)"],
                key="export_format"
            )
            
            df = data.get(export_dataset)
            if df is not None and hasattr(df, 'to_csv'):
                export_df = df.copy()
                export_df.columns = [col.replace('_', '') for col in export_df.columns]
                
                if "CSV" in export_format:
                    csv_data = export_df.to_csv(index=True)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"mms_{export_dataset.lower()}_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    txt_data = export_df.to_csv(sep='\t', index=True)
                    st.download_button(
                        label="Download TXT",
                        data=txt_data,
                        file_name=f"mms_{export_dataset.lower()}_data.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            st.divider()
        
        # Load new data button
        if data_loaded:
            if st.button("Load New Data", use_container_width=True):
                st.session_state.data = None
                st.session_state.data_loaded = False
                st.session_state.download_info = {}
                st.session_state.upload_info = {}
                st.rerun()
            
            # Show current data info
            if info:
                st.caption(
                    f"MMS{info.get('probe', '?')} {info.get('instrument', '')} "
                    f"{info.get('coord', '')} ({info.get('data_rate', '')}/{info.get('level', '')})"
                )
    
    return analysis_mode, subsample_pts




def render_analysis(analysis_mode: str, subsample_pts: int):
    """Render the selected analysis view."""
    
    data = st.session_state.get('data', {})
    info = st.session_state.get('download_info', {})
    
    if not data:
        st.warning("No data loaded.")
        return
    
    if analysis_mode == "Time Series Inspector":
        render_time_series_analysis(data, info, subsample_pts)
    elif analysis_mode == "Power Spectral Density":
        render_psd_analysis(data, info)
    elif analysis_mode == "PDF & Moments":
        render_pdf_analysis(data, info)
    elif analysis_mode == "Summary Statistics":
        render_summary_analysis(data, info, subsample_pts)


def render_time_series_analysis(datasets: dict, info: dict, subsample_pts: int):
    """Render Time Series Inspector view."""
    from plots import plot_magnetic_field, plot_velocity_field, PLOTLY_CONFIG
    
    probe = info.get('probe', '?')
    coord = info.get('coord', 'GSE').upper()
    
    for key, df in datasets.items():
        if df is None or len(df) == 0:
            continue
        
        # Generate dynamic title
        start_dt = df.index[0]
        end_dt = df.index[-1]
        
        if start_dt.date() == end_dt.date():
            date_str = start_dt.strftime('%d %B %Y')
            time_range = f"{start_dt.strftime('%H:%M:%S')} – {end_dt.strftime('%H:%M:%S')}"
        else:
            date_str = f"{start_dt.strftime('%d %b %Y %H:%M')} – {end_dt.strftime('%d %b %Y %H:%M')}"
            time_range = ""
        
        # Subsample
        if len(df) > subsample_pts:
            step = len(df) // subsample_pts
            plot_df = df.iloc[::step]
        else:
            plot_df = df
        
        # Determine plot type
        if key == 'FGM':
            if time_range:
                title = f"MMS {probe} | B | {coord} | {date_str} | {time_range}"
            else:
                title = f"MMS {probe} | B | {coord} | {date_str}"
            fig = plot_magnetic_field(plot_df, title=title, height=550)
        elif key == 'DES':
            if time_range:
                title = f"MMS {probe} | Electron Velocity | {coord} | {date_str} | {time_range}"
            else:
                title = f"MMS {probe} | V<sub>e</sub> | {coord} | {date_str}"
            fig = plot_velocity_field(plot_df, title=title, species='electron', height=450)
        elif key == 'DIS':
            if time_range:
                title = f"MMS {probe} | Ion Velocity | {coord} | {date_str} | {time_range}"
            else:
                title = f"MMS {probe} | V<sub>i</sub> | {coord} | {date_str}"
            fig = plot_velocity_field(plot_df, title=title, species='ion', height=450)
        else:
            continue
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        if len(df) > 1:
            try:
                sampling_hz = 1 / (df.index[1] - df.index[0]).total_seconds()
                st.caption(f"{key}: {len(plot_df):,} of {len(df):,} points | {sampling_hz:.1f} Hz")
            except:
                st.caption(f"{key}: {len(plot_df):,} of {len(df):,} points")


def render_psd_analysis(datasets: dict, info: dict):
    """Render Power Spectral Density analysis."""
    st.markdown("### Power Spectral Density")
    
    dataset_keys = list(datasets.keys())
    selected_key = st.selectbox("Dataset", dataset_keys, key="psd_dataset")
    
    df = datasets[selected_key]
    columns = list(df.columns)
    selected_col = st.selectbox("Variable", columns, key="psd_col")
    
    data = df[selected_col].values
    clean_data = data[~np.isnan(data)]
    time_data = df.index.values.astype('datetime64[ns]')
    
    st.markdown(f"#### PSD: {selected_key} {selected_col}")
    
    try:
        psd = cached_psd(tuple(clean_data), tuple(time_data[:len(clean_data)].astype(np.int64)))
        units = "nT²/Hz" if 'B' in selected_col else "km²/s²/Hz"
        fig = create_psd_plot(psd.frequencies, psd.power, title=f"PSD: {selected_key} {selected_col}", psd_units=units)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Sampling: {psd.sampling_frequency:.2f} Hz | Segments: {psd.nperseg}")
    except Exception as e:
        st.error(str(e))


def render_pdf_analysis(datasets: dict, info: dict):
    """Render PDF & Moments analysis."""
    st.markdown("### PDF & Moments")
    
    dataset_keys = list(datasets.keys())
    selected_key = st.selectbox("Dataset", dataset_keys, key="pdf_dataset")
    
    df = datasets[selected_key]
    columns = list(df.columns)
    selected_col = st.selectbox("Variable", columns, key="pdf_col")
    
    data = df[selected_col].values
    clean_data = data[~np.isnan(data)]
    
    c1, c2 = st.columns([3, 1])
    bins = c1.slider("Bins", 20, 200, 50, key="pdf_bins")
    logy = c2.checkbox("Log Y", key="pdf_logy")
    
    cp, cs = st.columns([2, 1])
    
    with cp:
        try:
            pdf = cached_pdf(tuple(clean_data), bins)
            units = "nT" if 'B' in selected_col else "km/s"
            fig = create_pdf_plot(pdf.bin_centers, pdf.density, xlabel=f"{selected_col} ({units})", log_y=logy)
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


def render_summary_analysis(datasets: dict, info: dict, subsample_pts: int):
    """Render Summary Statistics analysis."""
    st.markdown("### Summary Statistics")
    
    dataset_keys = list(datasets.keys())
    selected_key = st.selectbox("Dataset", dataset_keys, key="sum_dataset")
    
    df = datasets[selected_key]
    columns = list(df.columns)
    selected_col = st.selectbox("Variable", columns, key="sum_col")
    
    data = df[selected_col].values
    clean_data = data[~np.isnan(data)]
    time_data = df.index.values.astype('datetime64[ns]')
    
    # Quick time series preview
    step = max(1, len(time_data) // min(subsample_pts, 8000))
    fig = create_time_series_plot(time_data[::step], data[::step], 
                                  title=f"{selected_key}: {selected_col}", 
                                  ylabel=f"{selected_col}", height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### Statistics")
        try:
            stats = cached_stats(tuple(clean_data))
            for n, v in create_stats_display(stats).items():
                st.text(f"{n}: {v}")
        except Exception as e:
            st.error(str(e))
    
    with c2:
        st.markdown("##### Quick PSD")
        try:
            psd = cached_psd(tuple(clean_data), tuple(time_data[:len(clean_data)].astype(np.int64)))
            units = "nT²/Hz" if 'B' in selected_col else "km²/s²/Hz"
            fig = create_psd_plot(psd.frequencies, psd.power, psd_units=units, height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))


def main():
    """Main application entry point."""
    
    # Apply dark theme CSS
    apply_custom_css()
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'download_info' not in st.session_state:
        st.session_state.download_info = {}
    if 'upload_info' not in st.session_state:
        st.session_state.upload_info = {}
    
    # Render sidebar (always visible)
    analysis_mode, subsample_pts = render_sidebar()
    
    # Main content area
    if not st.session_state.get('data_loaded', False):
        # Show data loader wizard
        render_data_loader()
    else:
        # Show analysis view
        if analysis_mode:
            render_analysis(analysis_mode, subsample_pts)
        else:
            st.info("Select an analysis mode from the sidebar.")


if __name__ == "__main__":
    main()

