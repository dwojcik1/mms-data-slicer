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
    page_icon="◇",
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

from plots import plot_time_series, create_psd_plot, create_pdf_plot, create_stats_display



# ============================================================================
# Liquid Glass Landing Page (Full HTML/CSS)
# ============================================================================

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
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

.info-box {
    max-width: 800px;
    margin: 0 auto 20px auto;
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px 32px;
}

.info-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(248, 250, 252, 0.9);
    margin-bottom: 14px;
    letter-spacing: -0.01em;
}

.info-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.info-list li {
    position: relative;
    padding-left: 20px;
    margin-bottom: 10px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: rgba(248, 250, 252, 0.6);
}

.info-list li::before {
    content: '▸';
    position: absolute;
    left: 0;
    color: rgba(165, 180, 252, 0.7);
}

.info-list li:last-child {
    margin-bottom: 0;
}

.info-list a {
    color: #a5b4fc;
    text-decoration: none;
    transition: color 0.2s ease;
}

.info-list a:hover {
    color: #c7d2fe;
    text-decoration: underline;
}

.info-list strong {
    color: rgba(200, 210, 255, 0.85);
    font-weight: 500;
}

.glass-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.12) 0%,
        rgba(255, 255, 255, 0.05) 50%,
        rgba(200, 220, 255, 0.08) 100%
    );
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.12),
        inset 0 1px 1px rgba(255, 255, 255, 0.25),
        inset 0 -1px 1px rgba(255, 255, 255, 0.1);
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

.glass-icon::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        135deg,
        rgba(255, 120, 200, 0.05) 0%,
        rgba(120, 200, 255, 0.05) 50%,
        rgba(200, 255, 150, 0.03) 100%
    );
    opacity: 0;
    transition: opacity 0.4s ease;
    border-radius: inherit;
}

.glass-icon:hover {
    transform: translateY(-3px) scale(1.08);
    border-color: rgba(255, 255, 255, 0.28);
    box-shadow: 
        0 12px 40px rgba(0, 0, 0, 0.15),
        0 0 0 1px rgba(255, 255, 255, 0.1),
        inset 0 1px 2px rgba(255, 255, 255, 0.35),
        inset 0 -1px 2px rgba(255, 255, 255, 0.15);
}

.glass-icon:hover::before {
    opacity: 1;
}

.glass-icon .material-icons {
    font-size: 26px;
    background: linear-gradient(
        135deg, 
        rgba(255, 255, 255, 0.95) 0%, 
        rgba(200, 210, 255, 0.9) 40%,
        rgba(180, 200, 255, 0.85) 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.15));
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

<div class="info-box">
    <div class="info-title">What this app does</div>
    <ul class="info-list">
        <li>Load magnetic field and plasma time series from NASA's <a href="https://mms.gsfc.nasa.gov" target="_blank">Magnetospheric Multiscale (MMS)</a> mission via CDAWeb</li>
        <li>Compute <strong>Welch Power Spectral Densities</strong> with configurable windowing, segment overlap, and detrending options</li>
        <li>Fit and compare <strong>inertial- and kinetic-range spectral indices</strong> against reference slopes (Kolmogorov −5/3, kinetic-range −2.8, etc.)</li>
    </ul>
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
                
                # Frequency range for custom fit
                f_pos = psd.frequencies[psd.frequencies > 0]
                if len(f_pos) > 0:
                    f_min_data, f_max_data = float(f_pos.min()), float(f_pos.max())
                    
                    st.markdown("##### Custom Spectral Fit")
                    st.caption("Select frequency range to fit your own spectral index (red line)")
                    
                    # Calculate safe default values - use 25% and 75% of log range
                    log_f_min = np.log10(f_min_data)
                    log_f_max = np.log10(f_max_data)
                    log_range = log_f_max - log_f_min
                    default_f_min = float(10 ** (log_f_min + log_range * 0.25))
                    default_f_max = float(10 ** (log_f_min + log_range * 0.75))
                    # Final clamp to ensure within bounds
                    default_f_min = max(f_min_data, min(default_f_min, f_max_data))
                    default_f_max = max(f_min_data, min(default_f_max, f_max_data))
                    if default_f_min >= default_f_max:
                        default_f_min = f_min_data
                        default_f_max = f_max_data
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fit_f_min = st.slider(
                            "f_min [Hz]", 
                            min_value=0.0, 
                            max_value=f_max_data,
                            value=default_f_min,
                            format="%.2e",
                            key="psd_fit_fmin_1"
                        )
                    with col2:
                        fit_f_max = st.slider(
                            "f_max [Hz]", 
                            min_value=0.0, 
                            max_value=f_max_data,
                            value=default_f_max,
                            format="%.2e",
                            key="psd_fit_fmax_1"
                        )
                    
                    # Ensure f_min < f_max
                    if fit_f_min >= fit_f_max:
                        st.warning("f_min must be less than f_max")
                        user_fit_range = None
                    else:
                        user_fit_range = (fit_f_min, fit_f_max)
                else:
                    user_fit_range = None
                
                fig, fitted_slope = create_psd_plot(
                    psd.frequencies, psd.power, 
                    title=f"PSD: {selected_key} {selected_col}",
                    psd_units=r"nT²/Hz" if 'B' in selected_col else "km²/s²/Hz",
                    user_fit_range=user_fit_range
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display fitted slope
                if fitted_slope is not None:
                    st.success(f"**Fitted Spectral Index:** α = {fitted_slope:.3f}")
                
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
                fig, _ = create_psd_plot(psd.frequencies, psd.power, title=f"PSD: {selected_col}",
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
            
            # Construct temp DF for the unified plotter
            plot_df = df[[selected_col]].iloc[::step]
            meta = {'label': selected_col, 'unit': '(nT)', 'type': 'scalar'}
            
            fig = plot_time_series(plot_df, meta, title=selected_col)
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
                    fig, _ = create_psd_plot(psd.frequencies, psd.power, 
                                          psd_units=r"$\mathrm{nT}^2/\mathrm{Hz}$", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


# ============================================================================
# Main Application
# ============================================================================

# MMS Instrument catalog with descriptions
MMS_INSTRUMENTS = {
    "Fluxgate Magnetometer (FGM)": {
        "key": "fgm", "active": True,
        "desc": "Load data from the MMS fluxgate magnetometer"
    },
    "Fast Plasma Investigation (FPI)": {
        "key": "fpi", "active": True,
        "desc": "Load data from the MMS Fast Plasma Investigation (FPI)"
    },
    "Search Coil Magnetometer (SCM)": {
        "key": "scm", "active": True,
        "desc": "Load data from the MMS Search Coil Magnetometer (SCM)"
    },
    "FGM+SCM Data (FSM)": {
        "key": "fsm", "active": True,
        "desc": "Load data from the MMS FSM (FGM + SCM) data"
    },
    "Electric Field Double Probe (EDP)": {
        "key": "edp", "active": True,
        "desc": "Load data from the MMS Electric field Double Probes (EDP) instrument"
    },
    "Electron Drift Instrument (EDI)": {
        "key": "edi", "active": True,
        "desc": "Load data from the MMS Electron Drift Instrument (EDI)"
    },
    "Fly's Eye Energetic Particle Sensor (FEEPS)": {
        "key": "feeps", "active": True,
        "desc": "Load data from the MMS Fly's Eye Energetic Particle Sensor (FEEPS)"
    },
    "Energetic Ion Spectrometer (EIS)": {
        "key": "eis", "active": True,
        "desc": "Load data from the MMS Energetic Ion Spectrometer (EIS)"
    },
    "Active Spacecraft Potential Control (ASPOC)": {
        "key": "aspoc", "active": True,
        "desc": "Load data from the MMS Active Spacecraft Potential Control (ASPOC)"
    },
    "Hot Plasma Composition Analyzer (HPCA)": {
        "key": "hpca", "active": True,
        "desc": "Load data from the MMS Hot Plasma Composition Analyzer (HPCA)"
    },
    "Mission Ephemeris Coordinates (MEC)": {
        "key": "mec", "active": True,
        "desc": "Load data from the MMS Mission Ephemeris and Coordinates files"
    },
    "Attitude and Ephemeris (STATE)": {
        "key": "state", "active": True,
        "desc": "Load the state (ephemeris and attitude) data"
    },
    "Tetrahedron Quality Factor (TQF)": {
        "key": "tqf", "active": True,
        "desc": "Load the MMS tetrahedron quality factor data"
    },
}


# Instrument parameter configuration for PySPEDAS
# Defines available rates, levels, and datatypes for each instrument
INSTRUMENT_CONFIG = {
    "fgm": {
        "rates": ["SRVY", "BRST", "FAST", "SLOW"],
        "levels": ["L2", "L1B", "QL"],
        "types": [],  # datatype not used for FGM
        "has_coord": True,
        "coords": ["GSM", "GSE"]
    },
    "fpi": {
        "rates": ["FAST", "BRST"],
        "levels": ["L2", "QL", "L1B"],
        "types": ["DIS-MOMS", "DES-MOMS", "DIS-MOMSAUX", "DES-MOMSAUX", 
                  "DIS-DIST", "DES-DIST", "DIS-PARTMOMS", "DES-PARTMOMS"],
        "has_coord": True,
        "coords": ["GSM", "GSE"]
    },
    "scm": {
        "rates": ["SRVY", "BRST", "FAST", "SLOW"],
        "levels": ["L2", "L1B"],
        "types": ["SCSRVY", "SCB", "SCF", "SCHB", "SCS", "SCM", "CAL"],
        "has_coord": False,
        "coords": []
    },
    "fsm": {
        "rates": ["BRST"],
        "levels": ["L3"],
        "types": ["8KHZ"],
        "has_coord": False,
        "coords": []
    },
    "edp": {
        "rates": ["FAST", "SRVY", "SLOW", "BRST"],  # Default: fast per PySPEDAS
        "levels": ["L2", "L1B", "QL"],
        "types": ["DCE", "DCV", "ACE", "HMFE"],  # Default: dce
        "has_coord": False,
        "coords": []
    },
    "edi": {
        "rates": ["SRVY", "FAST", "SLOW"],
        "levels": ["L2", "QL"],
        "types": ["EFIELD", "AMB"],
        "has_coord": False,
        "coords": []
    },
    "feeps": {
        "rates": ["SRVY", "BRST"],
        "levels": ["L2", "L1B"],
        "types": ["ELECTRON", "ION"],
        "has_coord": False,
        "coords": []
    },
    "eis": {
        "rates": ["SRVY", "BRST"],
        "levels": ["L2"],
        "types": ["EXTOF", "PHXTOF", "ELECTRONENERGY"],  # Default: extof per PySPEDAS
        "has_coord": False,
        "coords": []
    },
    "aspoc": {
        "rates": ["SRVY", "SITL"],
        "levels": ["L2"],
        "types": [],  # datatype ignored
        "has_coord": False,
        "coords": []
    },
    "hpca": {
        "rates": ["SRVY", "BRST"],
        "levels": ["L2"],
        "types": ["MOMENTS", "ION"],
        "has_coord": False,
        "coords": []
    },
    "mec": {
        "rates": ["SRVY", "BRST"],
        "levels": ["L2"],
        "types": ["EPHT89Q", "EPHT89D", "EPHTS04D"],
        "has_coord": False,
        "coords": []
    },
    "state": {
        "rates": [],  # no rate for STATE
        "levels": ["DEF", "PRED"],
        "types": ["POS", "VEL", "SPINRAS", "SPINDEC"],
        "has_coord": False,
        "coords": []
    },
    "tqf": {
        "rates": [],  # no rate for TQF
        "levels": [],
        "types": ["QUAL"],
        "has_coord": False,
        "coords": []
    },
}


def render_data_loader():
    """Render the main page data configuration wizard."""
    

    # Main title
    st.markdown("## Magnetospheric Multiscale (MMS) Turbulence Lab ◇")
    st.markdown("### Data Configuration ◈")
    st.caption("Select your data source and configure the download parameters.")
    
    # "What this app does" info box
    st.markdown(
        """
        <div style="
            max-width: 100%;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px 24px;
            margin: 20px 0;
        ">
            <div style="font-size: 1rem; font-weight: 600; color: rgba(248, 250, 252, 0.9); margin-bottom: 12px;">
                What this app does
            </div>
            <ul style="list-style: none; padding: 0; margin: 0; color: rgba(248, 250, 252, 0.6); font-size: 0.9rem; line-height: 1.7;">
                <li style="margin-bottom: 8px;">▸ Load magnetic field and plasma time series from NASA's <a href="https://mms.gsfc.nasa.gov" target="_blank" style="color: #a5b4fc;">Magnetospheric Multiscale (MMS)</a> mission via CDAWeb</li>
                <li style="margin-bottom: 8px;">▸ Compute <strong style="color: rgba(200, 210, 255, 0.85);">Welch Power Spectral Densities</strong> with configurable windowing, segment overlap, and detrending</li>
                <li>▸ Fit and compare <strong style="color: rgba(200, 210, 255, 0.85);">inertial- and kinetic-range spectral indices</strong> against reference slopes (Kolmogorov −5/3, kinetic −2.8)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
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
        "Select MMS Instrument",
        list(MMS_INSTRUMENTS.keys()),
        index=0,
        help="FGM: Magnetic field (nT) | FPI: Plasma moments (density, velocity) | SCM: AC magnetic fluctuations | EDP: Electric field"
    )
    
    instrument_info = MMS_INSTRUMENTS[instrument_name]
    instrument_key = instrument_info["key"]
    is_active = instrument_info["active"]
    instrument_desc = instrument_info.get("desc", "")
    
    # Display instrument description from dictionary
    if instrument_desc:
        st.caption(f"*{instrument_desc}*")


    
    # Time Range section
    st.markdown("### Time Range")
    
    # Show full data availability range
    from utils import get_data_time_range, get_mms_dataset_id
    dataset_id = get_mms_dataset_id('1', instrument_key, 'srvy' if instrument_key == 'fgm' else 'fast', 'l2')
    start_avail, end_avail = get_data_time_range(dataset_id)
    if start_avail and end_avail:
        st.caption(f"▫ Data available from: **{start_avail}** to **{end_avail}**")
    
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

    
    # Configuration section - dynamic based on instrument
    st.markdown("### Configuration")
    
    # Get instrument-specific config
    inst_config = INSTRUMENT_CONFIG.get(instrument_key, {})
    rates = inst_config.get("rates", [])
    levels = inst_config.get("levels", [])
    types = inst_config.get("types", [])
    has_coord = inst_config.get("has_coord", False)
    coords = inst_config.get("coords", [])
    
    # Determine number of columns based on available options
    col_count = 1  # Always have Probe
    if rates: col_count += 1
    if levels: col_count += 1
    if types: col_count += 1
    if has_coord and coords: col_count += 1
    
    cols = st.columns(col_count)
    col_idx = 0
    
    # Probe (always shown)
    with cols[col_idx]:
        probe = st.selectbox(
            "Probe", 
            ['1', '2', '3', '4'], 
            index=0,
            help="MMS constellation spacecraft (1-4). Probes maintain ~10-160 km tetrahedron formation."
        )
    col_idx += 1
    
    # Data Rate (if available)
    data_rate = None
    if rates:
        with cols[col_idx]:
            # Build rate help text based on instrument
            rate_help = {
                'fgm': "SRVY: 16 Hz survey | BRST: 128 Hz burst | FAST: 8 Hz | SLOW: 0.125 Hz",
                'fpi': "FAST: 4.5s (ions), 30ms (electrons) | BRST: 150ms (ions), 30ms (electrons)",
                'scm': "SRVY: 32 Hz | BRST: 8192 Hz (SCB mode)",
                'edp': "FAST: 32 Hz | BRST: 8192 Hz | SLOW: 8 Hz",
            }.get(instrument_key, "Sampling rate mode")
            data_rate_display = st.selectbox("Data Rate", rates, index=0, help=rate_help)
            data_rate = data_rate_display.lower()
        col_idx += 1
    
    # Level (if available)
    level = None
    if levels:
        with cols[col_idx]:
            level_help = "L2: Science-quality calibrated data | L1B: Calibrated, uncorrected | QL: Quick-look (near real-time)"
            level_display = st.selectbox("Level", levels, index=0, help=level_help)
            level = level_display.lower()
        col_idx += 1
    
    # Datatype (if available)
    datatype = None
    if types:
        with cols[col_idx]:
            # Build datatype help based on instrument
            type_help = {
                'fpi': "DIS-MOMS: Ion moments (density, velocity) | DES-MOMS: Electron moments | DIST: Distribution functions",
                'scm': "SCSRVY: Survey AC magnetic field | SCB: Burst waveform | SCHB: High-frequency burst",
                'edp': "DCE: DC E-field (mV/m) | DCV: Spacecraft potential | ACE: AC E-field | HMFE: High-freq E-field",
                'eis': "EXTOF: Energetic ions (20-500 keV) | PHXTOF: Protons (10-600 keV) | ELECTRONENERGY: Electrons",
            }.get(instrument_key, "Data product type")
            datatype_display = st.selectbox("Datatype", types, index=0, help=type_help)
            datatype = datatype_display.lower()
        col_idx += 1
    
    # Coordinates (if available for this instrument)
    coord = None
    if has_coord and coords:
        with cols[col_idx]:
            coord_help = "GSE: Geocentric Solar Ecliptic (X→Sun) | GSM: Geocentric Solar Magnetospheric (X→Sun, Z→dipole)"
            coord_display = st.selectbox("Coordinates", coords, index=0, help=coord_help)
            coord = coord_display.lower()
    
    st.markdown("")  # Spacer

    
    # Download button
    if st.button(f"Load {instrument_key.upper()} Data", type="primary", use_container_width=True):
        
        trange = format_trange(start_date, start_time, end_date, end_time)
        
        with st.spinner(f"Downloading MMS{probe} {instrument_key.upper()} data from NASA CDAWeb..."):
            try:
                # Import universal loader
                from downloader import load_mms_universal
                
                # Call universal loader with all parameters
                datasets = load_mms_universal(
                    instrument=instrument_key,
                    trange=trange,
                    probe=probe,
                    data_rate=data_rate if data_rate else 'srvy',
                    level=level if level else 'l2',
                    coord=coord if coord else 'gse',
                    datatype=datatype if datatype else ''
                )
                
                st.session_state.data = datasets
                st.session_state.data_loaded = True
                st.session_state.download_info = {
                    'probe': probe,
                    'data_rate': data_rate if data_rate else '',
                    'level': level if level else '',
                    'coord': coord.upper() if coord else '',
                    'datatype': datatype if datatype else '',
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
    
    st.markdown("### ▢ Upload CDF File")
    
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
        st.markdown("### Global Controls ◎")
        
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
            "Increasing this value improves resolution but increases memory usage."
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
        st.markdown("### Analysis Mode ▣")
        
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
    """Render Time Series Inspector view with Unified Publication Style."""
    from plots import plot_time_series, PLOTLY_CONFIG
    
    probe = info.get('probe', '?')
    coord = info.get('coord', 'GSE').upper()
    
    for key, df in datasets.items():
        if df is None or len(df) == 0:
            continue
        
        # --- Metadata Map (Instrument Logic) ---
        key_upper = key.upper()
        cols = [str(c).lower() for c in df.columns]
        
        # Default
        meta = {'label': key, 'unit': '', 'type': 'scalar'}
        
        if any(x in key_upper for x in ['FGM', 'SCM', 'FSM']):
             meta = {'label': r"$\mathbf{B}$", 'unit': "[nT]", 'type': 'vector'}
             
        elif any(x in key_upper for x in ['EDP', 'EDI']):
             meta = {'label': r"$\mathbf{E}$", 'unit': "[mV/m]", 'type': 'vector'}
             
        elif any(x in key_upper for x in ['MEC', 'STATE']):
             meta = {'label': r"$\mathbf{R}$", 'unit': "[km]", 'type': 'vector'}
             
        elif any(x in key_upper for x in ['FPI', 'DIS', 'DES']):
            # Distinguish Density vs Velocity based on column names
            is_density = any('dens' in c or 'number' in c for c in cols)
            if is_density:
                meta = {'label': r"$N$", 'unit': "[cm$^{-3}$]", 'type': 'scalar'}
            else:
                meta = {'label': r"$\mathbf{V}$", 'unit': "[km/s]", 'type': 'vector'}
                
        elif 'HPCA' in key_upper:
            # Usually density or flux, assuming density for dominant moments
            meta = {'label': r"$N$", 'unit': "[cm$^{-3}$]", 'type': 'scalar'}
            
        elif any(x in key_upper for x in ['FEEPS', 'EIS']):
            meta = {'label': r"$J$", 'unit': "[flux]", 'type': 'scalar'}

        # --- Dynamic Title Generation ---
        start_dt = df.index[0]
        end_dt = df.index[-1]
        
        if start_dt.date() == end_dt.date():
            date_str = start_dt.strftime('%d %B %Y')
            time_str = f"{start_dt.strftime('%H:%M:%S')} – {end_dt.strftime('%H:%M:%S')}"
            # e.g. "MMS 1 | B | GSE | 12 Jan 2024 | 12:00:00 - 14:00:00"
            title = f"MMS {probe} | {meta['label']} | {coord} | {date_str} | {time_str}"
        else:
            date_str = f"{start_dt.strftime('%d %b %Y %H:%M')} – {end_dt.strftime('%d %b %Y %H:%M')}"
            title = f"MMS {probe} | {meta['label']} | {coord} | {date_str}"
        
        # --- Subsampling ---
        if len(df) > subsample_pts:
            step = len(df) // subsample_pts
            plot_df = df.iloc[::step]
        else:
            plot_df = df
        
        # --- Plotting ---
        # Call the new unified plotter with metadata
        fig = plot_time_series(plot_df, meta, title=title)
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # --- Caption ---
        if len(df) > 1:
            try:
                dt = (df.index[1] - df.index[0]).total_seconds()
                fs = 1 / dt if dt > 0 else 0
                st.caption(f"{key}: {len(plot_df):,} of {len(df):,} points | {fs:.1f} Hz")
            except:
                st.caption(f"{key}: {len(plot_df):,} of {len(df):,} points")


def _plot_generic_vector(df, title: str, y_label: str, colors: dict):
    """Generic vector plot for E-field, position, etc."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for col in df.columns:
        color = colors.get(col, '#1f77b4')
        fig.add_trace(go.Scattergl(
            x=df.index, y=df[col],
            mode='lines', name=col,
            line=dict(color=color, width=1)
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right')
    )
    
    return fig


def _plot_generic_scalar(df, title: str, key: str):
    """Generic scalar plot for density, flux, current, etc."""
    import plotly.graph_objects as go
    from plots import SCALAR_COLOR, INSTRUMENT_UNITS
    
    fig = go.Figure()
    
    for col in df.columns:
        fig.add_trace(go.Scattergl(
            x=df.index, y=df[col],
            mode='lines', name=col,
            line=dict(color=SCALAR_COLOR, width=1.5)
        ))
    
    # Try to get units for y-axis
    units = INSTRUMENT_UNITS.get(key.lower(), '')
    y_label = f"{key} ({units})" if units else key
    
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='x unified',
        height=450,
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right')
    )
    
    return fig


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
        
        # Frequency range for custom fit
        f_pos = psd.frequencies[psd.frequencies > 0]
        if len(f_pos) > 0:
            f_min_data, f_max_data = float(f_pos.min()), float(f_pos.max())
            
            st.markdown("##### Custom Spectral Fit")
            st.caption("Select frequency range to fit your own spectral index (red line)")
            
            # Calculate safe default values - use 25% and 75% of log range
            log_f_min = np.log10(f_min_data)
            log_f_max = np.log10(f_max_data)
            log_range = log_f_max - log_f_min
            default_f_min = float(10 ** (log_f_min + log_range * 0.25))
            default_f_max = float(10 ** (log_f_min + log_range * 0.75))
            # Final clamp to ensure within bounds
            default_f_min = max(f_min_data, min(default_f_min, f_max_data))
            default_f_max = max(f_min_data, min(default_f_max, f_max_data))
            if default_f_min >= default_f_max:
                default_f_min = f_min_data
                default_f_max = f_max_data
            
            col1, col2 = st.columns(2)
            with col1:
                fit_f_min = st.slider(
                    "f_min [Hz]", 
                    min_value=0.0, 
                    max_value=f_max_data,
                    value=default_f_min,
                    format="%.2e",
                    key="psd_fit_fmin_main"
                )
            with col2:
                fit_f_max = st.slider(
                    "f_max [Hz]", 
                    min_value=0.0, 
                    max_value=f_max_data,
                    value=default_f_max,
                    format="%.2e",
                    key="psd_fit_fmax_main"
                )
            
            if fit_f_min >= fit_f_max:
                st.warning("f_min must be less than f_max")
                user_fit_range = None
            else:
                user_fit_range = (fit_f_min, fit_f_max)
        else:
            user_fit_range = None
        
        fig, fitted_slope = create_psd_plot(
            psd.frequencies, psd.power, 
            title=f"PSD: {selected_key} {selected_col}", 
            psd_units=units,
            user_fit_range=user_fit_range
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if fitted_slope is not None:
            st.success(f"**Fitted Spectral Index:** α = {fitted_slope:.3f}")
        
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
    
    # Unified Plotter
    plot_df = df[[selected_col]].iloc[::step]
    meta = {'label': selected_col, 'unit': '', 'type': 'scalar'}
    fig = plot_time_series(plot_df, meta, title=f"{selected_key}: {selected_col}")

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
            fig, _ = create_psd_plot(psd.frequencies, psd.power, psd_units=units, height=350)
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

