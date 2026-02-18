"""
MMS Turbulence Analysis Suite
==============================
Kinetic scale time series processing for space plasma physics.
"""

import streamlit as st
import streamlit.components.v1 as components
import time
from contextlib import contextmanager

# MUST be first Streamlit command
st.set_page_config(
    page_title="MMS Turbulence Laboratory",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded"
)




import numpy as np

# Local imports
from styles import apply_custom_css
from utils import (
    CDFLoader, extract_component, get_variable_metadata, 
    get_component_label, VariableMetadata
)
from physics import compute_psd_welch, compute_pdf, compute_statistics, compute_pvi


from plots import plot_time_series, create_psd_plot, create_pdf_plot, create_stats_display, PLOTLY_CONFIG, PSD_CONFIG

# ============================================================================
# Utilities
# ============================================================================

@contextmanager
def _timed(label: str, enabled: bool, container=None):
    """Simple timing helper for optional telemetry."""
    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        target = container if container is not None else st
        target.caption(f"Timing: {label} {elapsed:.2f}s")

def _subsample_df(df, target_pts: int):
    """Return a subsampled DataFrame with ~target_pts rows (uniformly spaced)."""
    if target_pts <= 0:
        return df
    n = len(df)
    if n <= target_pts:
        return df
    target = max(2, min(target_pts, n))
    idx = np.linspace(0, n - 1, num=target, dtype=int)
    # Ensure unique and sorted indices
    idx = np.unique(idx)
    return df.iloc[idx].copy()


# ============================================================================
# Mission Intelligence Modal
# ============================================================================

@st.dialog("MMS Mission Intelligence", width="large")
def view_mission_modal():
    """Display MMS mission details in a styled modal dialog."""
    
    # Custom CSS for modal styling
    st.markdown("""
    <style>
    .mission-modal * {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .mission-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .mission-header h3 {
        color: #a78bfa;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .mission-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .metric-box {
        background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(167,139,250,0.1) 100%);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
    }
    .specs-box {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #7c3aed;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .specs-box h4 {
        color: #c4b5fd;
        font-size: 0.9rem;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .specs-box ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .specs-box li {
        color: rgba(255,255,255,0.8);
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .science-objectives {
        background: linear-gradient(135deg, rgba(124,58,237,0.1) 0%, rgba(167,139,250,0.08) 100%);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }
    .science-objectives h4 {
        color: #c4b5fd;
        font-size: 1rem;
        margin-bottom: 12px;
    }
    .science-objectives li {
        color: rgba(255,255,255,0.85);
        padding: 8px 0;
        padding-left: 20px;
        position: relative;
    }
    .science-objectives li::before {
        content: "→";
        position: absolute;
        left: 0;
        color: #a78bfa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
    <div class="mission-modal">
        <div class="mission-header">
            <h3>🌌 NASA Magnetospheric Multiscale (MMS) Mission</h3>
            <p>
                MMS investigates the fundamental physics of <strong>magnetic reconnection</strong>, a universal process 
                that converts magnetic energy into particle kinetic energy and heat. It is the first mission dedicated 
                to studying the electron diffusion region at the microphysics scale.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics - Bento Row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">4</div>
            <div class="metric-label">Spacecraft<br>Tetrahedral Formation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">10 km</div>
            <div class="metric-label">Min Separation<br>Electron Scale</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">30 ms</div>
            <div class="metric-label">Electron Res.<br>Fast Plasma (FPI)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">25</div>
            <div class="metric-label">Sensors<br>11 Instruments/SC</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mission Specs & Scientific Objectives
    col_specs, col_obj = st.columns(2)

    with col_specs:
        st.markdown("""
        <div class="specs-box">
            <h4>🛰️ Mission Specifications</h4>
            <ul>
                <li><strong>Orbit:</strong> Highly elliptical Earth orbit (Day-side & Night-side).</li>
                <li><strong>Formation:</strong> Adjustable tetrahedron (10km - 400km) to resolve 3D structure.</li>
                <li><strong>Spin:</strong> 3 RPM spin-stabilized.</li>
                <li><strong>Goal:</strong> Unveil the microphysics of reconnection (decoupling, acceleration).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Context Image
        st.image("assets/mms_formation.jpg", caption="MMS Formation (NASA SVS)", use_container_width=True)

    with col_obj:
        st.markdown("""
        <div class="science-objectives">
            <h4>🔬 Scientific Objectives</h4>
            <ul>
                <li><strong>Reconnection Physics:</strong> What determines when it starts/stops? How does field lines break and reconnect?</li>
                <li><strong>Particle Acceleration:</strong> How are particles heated and accelerated to high energies?</li>
                <li><strong>Turbulence:</strong> What is the role of turbulence in the reconnection process?</li>
                <li><strong>Diffusion Region:</strong> Resolve the structure of the Electron Diffusion Region (EDR).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="specs-box" style="margin-top: 16px;">
            <h4>📡 Instrument Suite</h4>
             <ul>
                <li><strong>Fields (1000 Hz):</strong> FGM (Magnetic), EDP (Electric), SCM (Waves), EDI (E-Field/B-Field).</li>
                <li><strong>Plasma (FPI/HPCA):</strong> DES/DIS (30ms/150ms res), Ion Composition (H+, He+, O+).</li>
                <li><strong>Energetic Particles:</strong> FEEPS (Electrons), EIS (Ions).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Instrument Layout Image (Moved here)
        st.image("assets/mms_spacecraft.jpg", caption="MMS Spacecraft Instrument Layout", use_container_width=True)

    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent Discoveries (2023-2026)
    st.markdown("""
    <div class="specs-box" style="margin-top: 0px;">
        <h4>🚀 Recent Discoveries (2023-2026)</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <ul>
                    <li><strong>Reconnection Electric Fields (2023):</strong> Direct measurement of the non-ideal electric field terms driving magnetic flux transport, validating kinetic theory.</li>
                    <li><strong>Magnetosheath Turbulence (2024):</strong> Identification of "electron-only" reconnection events in highly turbulent magnetosheath plasma.</li>
                </ul>
            </div>
            <div>
                <ul>
                    <li><strong>Cold Ion Effects (2025):</strong> New evidence that cold plasma of ionospheric origin significantly alters the reconnection rate at the magnetopause.</li>
                    <li><strong>Machine Learning Catalogs:</strong> Automated detection of over 5000+ EDR candidates now enabling statistical studies of diffusion region structure.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Close button
    if st.button("Close", use_container_width=True, type="primary"):
        st.rerun()



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
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #12121b;
    min-height: 100vh;
    color: #e4e4ed;
    padding: 40px 30px;
}

.hero {
    text-align: center;
    padding: 20px 0 40px 0;
}

.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #a78bfa 70%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 14px;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: clamp(0.95rem, 1.8vw, 1.25rem);
    font-weight: 400;
    color: rgba(228, 228, 237, 0.45);
    letter-spacing: 0.01em;
    max-width: 560px;
    margin: 0 auto;
}

.divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.04);
    margin: 28px auto;
    max-width: 800px;
}

.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
    max-width: 1300px;
    margin: 0 auto;
    padding: 0 10px;
}

.glass-card {
    background: #1e1e34;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}

.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(124, 58, 237, 0.25);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35), 0 0 30px -10px rgba(124, 58, 237, 0.15);
}

.card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 10px;
    letter-spacing: -0.01em;
}

.card-body {
    font-size: 0.9rem;
    line-height: 1.65;
    color: rgba(228, 228, 237, 0.5);
}

.card-body strong {
    color: #a78bfa;
    font-weight: 500;
}

.info-box {
    max-width: 800px;
    margin: 0 auto 20px auto;
    background: #1e1e34;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    padding: 24px 28px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}

.info-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff;
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
    padding-left: 18px;
    margin-bottom: 9px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: rgba(228, 228, 237, 0.55);
}

.info-list li::before {
    content: '▸';
    position: absolute;
    left: 0;
    color: #7c3aed;
}

.info-list li:last-child {
    margin-bottom: 0;
}

.info-list a {
    color: #a78bfa;
    text-decoration: none;
    transition: color 0.2s ease;
}

.info-list a:hover {
    color: #c4b5fd;
    text-decoration: underline;
}

.info-list strong {
    color: rgba(196, 181, 253, 0.85);
    font-weight: 500;
}

.glass-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    height: 46px;
    background: rgba(124, 58, 237, 0.1);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 14px;
}

.glass-icon:hover {
    transform: translateY(-2px) scale(1.05);
    border-color: rgba(124, 58, 237, 0.3);
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.15);
}

.glass-icon .material-icons {
    font-size: 24px;
    color: #a78bfa;
}

.footer {
    text-align: center;
    padding: 40px 20px 20px 20px;
    color: rgba(228, 228, 237, 0.25);
    font-size: 0.9rem;
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

@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def cached_psd(data, time_data, fs_override: float = 0.0):
    data_arr = np.asarray(data)
    time_arr = np.asarray(time_data, dtype='datetime64[ns]')
    fs = fs_override if fs_override and fs_override > 0 else None
    return compute_psd_welch(data_arr, time_arr, fs_override=fs)

@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def cached_pdf(data, n_bins):
    return compute_pdf(np.asarray(data), n_bins=n_bins)

@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def cached_stats(data):
    return compute_statistics(np.asarray(data))

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
            # Skip CDFLoader objects
            if hasattr(df, 'get_time_data') and not hasattr(df, 'columns'):
                continue
            if df is None or (hasattr(df, 'empty') and df.empty):
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
            if sub:
                plot_df = _subsample_df(df, pts)
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
                with _timed("PSD compute", st.session_state.get("perf_telemetry", False)):
                    psd = cached_psd(data, time_data)
                
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
                with _timed("PDF compute", st.session_state.get("perf_telemetry", False)):
                    pdf = cached_pdf(clean_data, bins)
                units = "nT" if 'B' in selected_col else "km/s"
                fig = create_pdf_plot(pdf.bin_centers, pdf.density, xlabel=f"{selected_col} ({units})", log_y=logy)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))
        
        else:
            st.markdown(f"#### Summary: {selected_key} {selected_col}")
            try:
                with _timed("Stats compute", st.session_state.get("perf_telemetry", False)):
                    stats = cached_stats(clean_data)
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
                with _timed("PSD compute", st.session_state.get("perf_telemetry", False)):
                    psd = cached_psd(data, time_data)
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
                    with _timed("PDF compute", st.session_state.get("perf_telemetry", False)):
                        pdf = cached_pdf(clean_data, bins)
                    fig = create_pdf_plot(pdf.bin_centers, pdf.density, xlabel=f"{selected_col} (nT)", log_y=logy)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))
            with cs:
                st.markdown("##### Statistics")
                try:
                    with _timed("Stats compute", st.session_state.get("perf_telemetry", False)):
                        stats = cached_stats(clean_data)
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
                    with _timed("Stats compute", st.session_state.get("perf_telemetry", False)):
                        stats = cached_stats(clean_data)
                    for n, v in list(create_stats_display(stats).items())[:6]:
                        st.text(f"{n}: {v}")
                except Exception as e:
                    st.error(str(e))
            with c2:
                st.markdown("##### PSD")
                try:
                    with _timed("PSD compute", st.session_state.get("perf_telemetry", False)):
                        psd = cached_psd(data, time_data)
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

    "Hot Plasma Composition Analyzer (HPCA)": {
        "key": "hpca", "active": True,
        "desc": "Load data from the MMS Hot Plasma Composition Analyzer (HPCA)"
    }
}


# Instrument parameter configuration for PySPEDAS
# Defines available rates, levels, and datatypes for each instrument
INSTRUMENT_CONFIG = {
    "fgm": {
        "rates": ["SRVY", "BRST", "FAST", "SLOW"],
        "levels": ["L2", "L1B", "QL"],
        "types": [],  # datatype not used for FGM
        "has_coord": True,
        "coords": ["GSM", "GSE", "LMN"]
    },
    "fpi": {
        "rates": ["FAST", "BRST"],
        "levels": ["L2", "QL", "L1B"],
        "types": ["DIS-MOMS", "DES-MOMS", "DIS-MOMSAUX", "DES-MOMSAUX", 
                  "DIS-DIST", "DES-DIST", "DIS-PARTMOMS", "DES-PARTMOMS"],
        "has_coord": True,
        "coords": ["GSM", "GSE", "LMN"]
    },
    "scm": {
        "rates": ["SRVY", "BRST", "FAST", "SLOW"],
        "levels": ["L2", "L1B"],
        "types": ["SCSRVY", "SCB", "SCF", "SCHB", "SCS", "SCM", "CAL"],
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
}


def render_data_loader():
    """Render the main page data configuration wizard."""
    

    # Main title
    st.markdown("## Magnetospheric Multiscale (MMS) Turbulence Lab")
    
    # Restructured Intro Section
    st.info(
        """
        **Application Capabilities**
        *   **Multi-Instrument Analysis:** Load and visualize data from MMS instruments (FGM, FPI, SCM, etc.).
        *   **Spectral Analysis:** Compute Power Spectral Density (PSD) with advanced dual-slope fitting (Inertial/Kinetic ranges).
        *   **Orbit Visualization:** 2D projections of spacecraft trajectories (MMS1-4) in various coordinate systems.
        *   **Space Physics Tools:** Analyze turbulence, spectral indices, and plasma parameters in a unified interface.
        """, icon="🌌"
    )

    st.markdown("### Data Configuration")
    st.caption("Select your data source and configure the download parameters.")
    
    # "What this app does" info box

    
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
            <a href="http://doi.org/10.1007/s11214-015-0164-9" target="_blank" style="color: #a78bfa;">
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
    
    # Analysis Type Selection
    st.markdown("### Analysis Type")
    try:
        # Try new segmented control (Streamlit 1.39+)
        analysis_type = st.segmented_control(
            "Select Operation Mode",
            ["Time Series Analysis", "Orbit Plots"],
            selection_mode="single",
            default="Time Series Analysis"
        )
        if not analysis_type: analysis_type = "Time Series Analysis"
    except AttributeError:
        # Fallback
        analysis_type = st.radio(
            "Select Operation Mode",
            ["Time Series Analysis", "Orbit Plots"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # Instrument selection (Only for Time Series)
    if analysis_type == "Time Series Analysis":
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
    else:
        # Default for Orbit Plots
        instrument_key = 'fgm'


    
    # Time Range section
    st.markdown("### Time Range")
    
    # Show full data availability range
    from utils import get_data_time_range, get_mms_dataset_id
    dataset_id = get_mms_dataset_id('1', instrument_key, 'srvy' if instrument_key == 'fgm' else 'fast', 'l2')
    start_avail, end_avail = get_data_time_range(dataset_id)
    if start_avail and end_avail:
        st.caption(f"▫ Data available from: **{start_avail}** to **{end_avail}**")
    
    from datetime import date, time, timedelta, datetime
    
    # Parse dynamic data availability range from the instrument
    if start_avail and end_avail:
        try:
            mms_min_date = datetime.strptime(start_avail, "%Y-%m-%d").date()
            mms_max_date = datetime.strptime(end_avail, "%Y-%m-%d").date()
        except ValueError:
            # Fallback if parsing fails
            mms_min_date = date(2015, 9, 1)
            mms_max_date = date.today()
    else:
        # Fallback if no availability data
        mms_min_date = date(2015, 9, 1)
        mms_max_date = date.today()
    
    default_start = date.today() - timedelta(days=1)
    default_end = date.today() - timedelta(days=1)
    
    # Clamp defaults to valid range
    default_start = max(mms_min_date, min(default_start, mms_max_date))
    default_end = max(mms_min_date, min(default_end, mms_max_date))
    
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        start_date = st.date_input("Start Date", value=default_start, min_value=mms_min_date, max_value=mms_max_date)
    with t2:
        start_time = st.time_input("Start Time", value=time(12, 0))
    with t3:
        end_date = st.date_input("End Date", value=default_end, min_value=mms_min_date, max_value=mms_max_date)
    with t4:
        end_time = st.time_input("End Time", value=time(12, 30))

    
    
    if analysis_type == "Time Series Analysis":
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
                    with _timed("Download", st.session_state.get("perf_telemetry", False)):
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
            
        # Disclaimer Note (Time Series)
        st.markdown(
            "<div style='text-align: center; margin-top: 8px; opacity: 0.7; font-size: 0.85em;'>"
            "📝 <em>Note: Data will not be saved to your local device unless explicitly exported.</em>"
            "</div>",
            unsafe_allow_html=True
        )

    elif analysis_type == "Orbit Plots":
        # Configuration for Orbits
        st.markdown("### Orbit Configuration")
        
        c_orb1, c_orb2 = st.columns(2)
        with c_orb1:
            # Uppercase options
            plane = st.selectbox("Projection Plane", ['XY', 'XZ', 'YZ'], index=0, help="Plane to project the 3D orbit onto.")
        with c_orb2:
            # Uppercase options
            coord_sys = st.selectbox("Coordinates", ['GSE', 'GSM', 'SM', 'GEO'], index=0, help="Coordinate system for position data.")
            
        # Default probes as integers (will be handled by wrapper or logic)
        probes = st.multiselect("Select Probes", [1, 2, 3, 4], default=[1, 2, 3, 4], help="Select which MMS probes to plot.")
        
        st.markdown("") # Spacer
        
        if st.button("Generate Orbit Plot", type="primary", use_container_width=True):
            if not probes:
                st.error("Please select at least one spacecraft.")
            else:
                trange = format_trange(start_date, start_time, end_date, end_time)
                
                with st.spinner("Generating Orbit Plot..."):
                    try:
                        from plots import plot_mms_orbit_wrapper
                        
                        # Pass probes as comes (wrapper expects strings, but st.multiselect gives ints or strings?)
                        # st.multiselect given options [1, 2, 3, 4] (ints) will return ints.
                        # plot_mms_orbit_wrapper handles conversion to strings.
                        
                        fig = plot_mms_orbit_wrapper(
                            trange=trange,
                            probes=probes,
                            plane=plane.lower(),
                            coord=coord_sys.lower()
                        )
                        
                        if fig:
                            st.pyplot(fig)
                            st.success("Orbit plot generated successfully!")
                    except Exception as e:
                        st.error(f"Orbit plot generation failed: {e}")
            
        # Disclaimer Note (Orbit Plots)
        st.markdown(
            "<div style='text-align: center; margin-top: 8px; opacity: 0.7; font-size: 0.85em;'>"
            "📝 <em>Note: Data will not be saved to your local device unless explicitly exported.</em>"
            "</div>",
            unsafe_allow_html=True
        )

    
    # Footer disclaimer (centered) - removed as requested
    st.markdown("")


def render_upload_form():
    """Render the CDF file upload form."""
    
    st.markdown("### Upload CDF File")
    
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
        st.markdown("<div style='margin-top:-0.9rem'></div>", unsafe_allow_html=True)
        # Load new data button (only after data is loaded / sub-pages)
        data_loaded = st.session_state.get('data_loaded', False)
        if data_loaded:
            if st.button("Load New Data", use_container_width=True):
                st.session_state.data = None
                st.session_state.data_loaded = False
                st.session_state.download_info = {}
                st.session_state.upload_info = {}
                st.rerun()
        
        # Mission Info Button - opens modal
        if st.button("🚀 Mission Info", use_container_width=True, help="Learn about the MMS mission"):
            view_mission_modal()
        
        st.markdown("### Global Controls")
        st.checkbox(
            "Performance telemetry",
            value=st.session_state.get("perf_telemetry", False),
            key="perf_telemetry",
            help="Show timing captions for downloads and computations."
        )
        
        # Dynamic subsample control based on loaded data
        data = st.session_state.get('data', None)
        
        if data_loaded and data:
            # Safely calculate total length - only count pandas DataFrames
            import pandas as pd
            total_len = 0
            for item in data.values():
                if isinstance(item, pd.DataFrame):
                    total_len += len(item)
                elif hasattr(item, 'get_time_data'):
                    # CDFLoader - get length from time data
                    try:
                        time_data = item.get_time_data()
                        if time_data is not None:
                            total_len += len(time_data)
                    except Exception:
                        pass
            max_pts = max(1000, total_len) if total_len > 0 else 100000
            default_pts = min(30000, total_len) if total_len > 0 else 30000
        else:
            max_pts = 100000
            default_pts = 30000
            total_len = 0
        
        # Keep total length in session state for callbacks
        st.session_state["total_len"] = total_len

        # Detect data change and Force Update to 20%
        # If total_len changed (new data loaded), reset the subsample points to 20%
        last_len = st.session_state.get("last_total_len", 0)
        if total_len > 0 and total_len != last_len:
            st.session_state["last_total_len"] = total_len
            # Reset to 20%
            target_pts = int(total_len * 0.20)
            target_pts = max(100, min(target_pts, 100000)) # Clamp
            st.session_state["subsample_pts"] = target_pts
            st.session_state["subsample_pct"] = 20
        
        # Also init if missing
        if "last_total_len" not in st.session_state and total_len > 0:
             st.session_state["last_total_len"] = total_len

        # Single number input for subsample control
        min_pts = 100  # Allow smaller datasets
        # Calculate safe default based on requested 20%
        target_default_pts = int(total_len * 0.20) if total_len > 0 else 15000
        safe_default = max(min_pts, min(target_default_pts, max_pts))

        # Sync callbacks between percent slider and absolute points
        def _update_pts_from_pct():
            total = st.session_state.get("total_len", 0)
            pct = st.session_state.get("subsample_pct", 100)
            if total > 0:
                pts = int(total * (pct / 100.0))
                pts = max(min_pts, min(max_pts, pts))
                st.session_state["subsample_pts"] = pts

        def _update_pct_from_pts():
            total = st.session_state.get("total_len", 0)
            pts = st.session_state.get("subsample_pts", safe_default)
            if total > 0:
                pct = int(round((pts / total) * 100))
                pct = max(1, min(100, pct))
                st.session_state["subsample_pct"] = pct

        # Initialize percent if missing (set before widget creation to avoid warnings)
        if "subsample_pct" not in st.session_state:
            if total_len > 0:
                # Default to 20% or calculated, user asked for 20%.
                # The logic below calculated percentage based on safe_default. 
                # If safe_default is 10000 and total 100000, then 10%.
                # User wants 20% default.
                st.session_state["subsample_pct"] = 20
            else:
                st.session_state["subsample_pct"] = 20

        # Percent slider control
        st.slider(
            "Subsample (%)",
            min_value=1,
            max_value=100,
            step=1,
            disabled=(total_len == 0),
            key="subsample_pct",
            on_change=_update_pts_from_pct
        )

        subsample_pts = st.number_input(
            "Subsample Points",
            min_value=min_pts,
            max_value=max_pts,
            value=safe_default,
            step=1000,
            key="subsample_pts",
            on_change=_update_pct_from_pts
        )
        
        # Explanation text
        st.markdown(
            "<div style='font-size:0.78rem; padding:6px 10px; "
            "border:1px solid rgba(255,255,255,0.08); border-radius:8px; "
            "background: rgba(255,255,255,0.03);'>"
            "<span style='color:#ef4444; font-size:1.15rem; font-weight:700; margin-right:6px;'>❗</span>"
            "<strong>Note:</strong> Large datasets are subsampled to maintain performance. "
            "Increasing this value improves resolution but increases memory usage."
            "</div>",
            unsafe_allow_html=True
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
        

        
        # Analysis mode navigation
        st.markdown("### Analysis Mode")
        
        if data_loaded:
            analysis_mode = st.sidebar.radio(
                "Analysis Mode",
                ["Time Series", "Power Spectral Density", "PDF & Moments", "Partial Variance of Increments (PVI)", "Summary"],
                label_visibility="collapsed",
                key="analysis_mode"
            )
        else:
            st.info("Load data to enable analysis")
            analysis_mode = None
        
            # no divider here

        
        # Data Export section (only when data is loaded)
        if data_loaded and data:
            st.markdown("### Export Data")
            
            dataset_keys = list(data.keys())
            if len(dataset_keys) == 1:
                export_dataset = dataset_keys[0]

            else:
                export_dataset = st.selectbox("Dataset", dataset_keys, key="export_dataset")
            export_format = st.selectbox(
                "Format", 
                ["CSV (.csv)", "Text File (.txt)", "CDF (.cdf)"],
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
                elif "Text" in export_format:
                    txt_data = export_df.to_csv(sep='\t', index=True)
                    st.download_button(
                        label="Download TXT",
                        data=txt_data,
                        file_name=f"mms_{export_dataset.lower()}_data.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    # CDF export using PySPEDAS
                    try:
                        from downloader import download_cdf_pyspedas
                        with st.spinner("Preparing CDF export via PySPEDAS..."):
                            cdf_path, cdf_name = download_cdf_pyspedas(
                                st.session_state.get("download_info", {})
                            )
                        with open(cdf_path, "rb") as f:
                            cdf_bytes = f.read()
                        st.download_button(
                            label="Download CDF",
                            data=cdf_bytes,
                            file_name=cdf_name,
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"CDF export failed: {e}")
            
            # No bottom divider

    return analysis_mode, subsample_pts




def render_analysis(analysis_mode: str, subsample_pts: int):
    """Render the selected analysis view."""
    
    data = st.session_state.get('data', {})
    info = st.session_state.get('download_info', {})
    
    if not data:
        st.warning("No data loaded.")
        return
    
    if analysis_mode == "Time Series":
        render_time_series_analysis(data, info, subsample_pts)
    elif analysis_mode == "Power Spectral Density":
        render_psd_analysis(data, info, subsample_pts)
    elif analysis_mode == "PDF & Moments":
        render_pdf_analysis(data, info, subsample_pts)
    elif analysis_mode == "Partial Variance of Increments (PVI)":
        render_pvi_analysis(data, info)
    elif analysis_mode == "Summary":
        render_summary_analysis(data, info, subsample_pts)



def render_time_series_analysis(datasets: dict, info: dict, subsample_pts: int):
    """Render Time Series Inspector view with Unified Publication Style."""
    from plots import plot_time_series, PLOTLY_CONFIG
    
    probe = info.get('probe', '?')
    coord = info.get('coord', 'GSE').upper()
    
    for key, df in datasets.items():
        # Skip CDFLoader objects (they're handled separately)
        if hasattr(df, 'get_time_data') and not hasattr(df, 'columns'):
            continue
        if df is None:
            continue
        # Use DataFrame.empty for pandas check
        if hasattr(df, 'empty') and df.empty:
            continue
        
        # --- Metadata Map (Instrument Logic) ---
        key_upper = key.upper()
        cols = [str(c).lower() for c in df.columns]
        
        # Default
        meta = {'label': key, 'unit': '', 'type': 'scalar'}
        
        if any(x in key_upper for x in ['FGM', 'SCM', 'FSM']):
             meta = {'label': 'B', 'unit': '[nT]', 'type': 'vector'}
             
        elif any(x in key_upper for x in ['EDP', 'EDI']):
             meta = {'label': 'E', 'unit': '[mV/m]', 'type': 'vector'}
             
        elif any(x in key_upper for x in ['MEC', 'STATE']):
             meta = {'label': 'R', 'unit': '[km]', 'type': 'vector'}
             
        elif any(x in key_upper for x in ['FPI', 'DIS', 'DES']):
            # Distinguish Density vs Velocity based on column names
            is_density = any('dens' in c or 'number' in c for c in cols)
            if is_density:
                meta = {'label': 'N', 'unit': '[cm⁻³]', 'type': 'scalar'}
            else:
                meta = {'label': 'V', 'unit': '[km/s]', 'type': 'vector'}
                
        elif 'HPCA' in key_upper:
            # Usually density or flux, assuming density for dominant moments
            meta = {'label': 'N', 'unit': '[cm⁻³]', 'type': 'scalar'}
            
        elif any(x in key_upper for x in ['FEEPS', 'EIS']):
            meta = {'label': 'J', 'unit': '[flux]', 'type': 'scalar'}

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
        
        # --- Subsampling with Persistence ---
        # Cache subsampled data to avoid recomputing on every rerun
        cache_key = f"{key}_{subsample_pts}_{len(df)}"
        subsample_cache = st.session_state.setdefault('subsample_cache', {})
        if cache_key not in subsample_cache:
            subsample_cache[cache_key] = _subsample_df(df, subsample_pts)
        plot_df = subsample_cache[cache_key]
        
        # --- Plotting ---
        # Simple help text (avoiding Streamlit components with Material Icons font issues)
        st.caption("💡 **Tip:** Click legend items to show/hide • Drag to zoom • Double-click to reset • Toolbar in top-right for more")
        
        # Call the new unified plotter with metadata
        fig = plot_time_series(plot_df, meta, title=title)
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # --- Caption with Metadata and Cadence ---
        if len(df) > 1:
            try:
                # Calculate average cadence
                time_diffs = (df.index[1:] - df.index[:-1]).total_seconds()
                avg_dt = np.median(time_diffs)
                fs = 1 / avg_dt if avg_dt > 0 else 0
                
                # Metadata string
                instr = info.get('instrument', key).upper()
                rate = info.get('data_rate', '').upper()
                lvl = info.get('level', '').lower()
                
                meta_str = f"MMS{probe} {instr} ({rate}/{lvl}) | {coord}"
                stats_str = f"{len(plot_df):,} displayed / {len(df):,} total points | fs ≈ {fs:.2f} Hz (dt ≈ {avg_dt:.4f} s)"
                
                st.caption(f"**{meta_str}** — {stats_str}")
            except Exception:
                st.caption(f"{key}: {len(plot_df):,} of {len(df):,} points")
        
        # Subsample hint
        st.info("💡 **Tip:** Adjust **Subsample Points** in the sidebar (left panel) to change plot resolution.", icon="⚙️")


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


def render_psd_analysis(datasets: dict, info: dict, subsample_pts: int):
    """Render Power Spectral Density analysis with smart fitting and dual fit support."""
    import pandas as pd
    from physics import find_target_alpha_range
    
    st.markdown("### Power Spectral Density")
    st.caption("Calculated using the Welch (1967) method.")
    
    # Filter to only include DataFrames
    valid_datasets = {k: v for k, v in datasets.items() 
                      if isinstance(v, pd.DataFrame) and not v.empty}
    
    if not valid_datasets:
        st.warning("No valid data available for PSD analysis. Please load data first.")
        return
    
    dataset_keys = list(valid_datasets.keys())
    
    # =========================================================================
    # STRUCTURE-DEFINING WIDGETS (Outside Form - Instant Rerun)
    # =========================================================================
    
    # Dataset Selection
    if len(dataset_keys) == 1:
        selected_key = dataset_keys[0]
        st.caption(f"**Dataset:** {selected_key}")
    else:
        selected_key = st.selectbox("Dataset", dataset_keys, key="psd_dataset")
    
    df = valid_datasets[selected_key]
    columns = list(df.columns)
    
    # Variable Selection
    selected_col = st.selectbox("Variable", columns, key="psd_col")
    
    # Compute PSD (cached)
    df_psd = df
    if subsample_pts and subsample_pts < len(df):
        df_psd = _subsample_df(df, subsample_pts)

    data = df_psd[selected_col].values
    clean_data = data[~np.isnan(data)]
    time_data = df_psd.index.values.astype('datetime64[ns]')
    
    # Optional sampling frequency override for known MMS cadences
    instr = (info.get('instrument', '') or '').lower()
    rate = (info.get('data_rate', '') or '').lower()
    fs_override = None
    try:
        if instr == 'fgm' and rate == 'brst':
            fs_override = 128.0
    except Exception:
        fs_override = None

    try:
        with _timed("PSD compute", st.session_state.get("perf_telemetry", False)):
            psd = cached_psd(data, time_data, fs_override or 0.0)
    except Exception as e:
        st.error(f"PSD computation failed: {e}")
        return
    
    # Get frequency bounds
    f_pos = psd.frequencies[psd.frequencies > 0]
    if len(f_pos) == 0:
        st.error("No valid frequency data")
        return
        
    f_min_data, f_max_data = float(f_pos.min()), float(f_pos.max())
    
    # Units
    if 'B' in selected_col.upper():
        units = "nT²/Hz"
    elif 'V' in selected_col.upper():
        units = "km²/s²/Hz"
    elif 'E' in selected_col.upper():
        units = "mV²/m²/Hz"
    else:
        units = "a.u."
    
    # Default frequency ranges
    log_f_min = np.log10(f_min_data)
    log_f_max = np.log10(f_max_data)
    log_range = log_f_max - log_f_min
    default_fit1_fmin = float(10 ** (log_f_min + log_range * 0.15))
    default_fit1_fmax = float(10 ** (log_f_min + log_range * 0.45))
    default_fit2_fmin = float(10 ** (log_f_min + log_range * 0.55))
    default_fit2_fmax = float(10 ** (log_f_min + log_range * 0.85))
    
    # Fitting Mode (Outside Form - Instant Rerun)
    col_mode, col_dual = st.columns([2, 1])
    with col_mode:
        fit_mode = st.radio(
            "Fitting Mode",
            ["Manual Range", "Target Spectral Index"],
            horizontal=True,
            key="psd_fit_mode",
            help="Manual: Specify f_min/f_max. Target: Auto-find range matching α."
        )
    with col_dual:
        enable_dual_fit = st.checkbox(
            "Enable Fit 2",
            value=False,
            key="psd_dual_fit",
            help="Add second spectral slope fit"
        )
    
    # =========================================================================
    # VALUE INPUTS (Inside Form - Waits for Submit)
    # =========================================================================
    with st.form(key="psd_control_form"):
        
        if fit_mode == "Manual Range":
            # --- MANUAL MODE ---
            st.markdown("**Fit 1 (Red)**")
            c1, c2 = st.columns(2)
            with c1:
                fit1_fmin_str = st.text_input(
                    "f₁ min [Hz]",
                    value=f"{default_fit1_fmin:.4f}",
                    key="psd_fit1_fmin_input"
                )
            with c2:
                fit1_fmax_str = st.text_input(
                    "f₁ max [Hz]",
                    value=f"{default_fit1_fmax:.4f}",
                    key="psd_fit1_fmax_input"
                )
            target_alpha1 = None
            target_alpha2 = None
            
            if enable_dual_fit:
                st.markdown("**Fit 2 (Green)**")
                c3, c4 = st.columns(2)
                with c3:
                    fit2_fmin_str = st.text_input(
                        "f₂ min [Hz]",
                        value=f"{default_fit2_fmin:.4f}",
                        key="psd_fit2_fmin_input"
                    )
                with c4:
                    fit2_fmax_str = st.text_input(
                        "f₂ max [Hz]",
                        value=f"{default_fit2_fmax:.4f}",
                        key="psd_fit2_fmax_input"
                    )
            else:
                fit2_fmin_str, fit2_fmax_str = f"{default_fit2_fmin:.4f}", f"{default_fit2_fmax:.4f}"
                
        else:
            # --- TARGET SPECTRAL INDEX MODE ---
            fit1_fmin, fit1_fmax = default_fit1_fmin, default_fit1_fmax
            fit2_fmin, fit2_fmax = default_fit2_fmin, default_fit2_fmax
            
            if enable_dual_fit:
                c1, c2 = st.columns(2)
                with c1:
                    target_alpha1 = st.number_input(
                        "Target α₁ (Fit 1)", min_value=-5.0, max_value=0.0,
                        value=-5/3, step=0.1, format="%.2f",
                        key="psd_target_alpha1",
                        help="e.g., -1.67 for Kolmogorov"
                    )
                with c2:
                    target_alpha2 = st.number_input(
                        "Target α₂ (Fit 2)", min_value=-5.0, max_value=0.0,
                        value=-8/3, step=0.1, format="%.2f",
                        key="psd_target_alpha2",
                        help="e.g., -2.67 for Kinetic"
                    )
            else:
                target_alpha1 = st.number_input(
                    "Target α₁", min_value=-5.0, max_value=0.0,
                    value=-5/3, step=0.1, format="%.2f",
                    key="psd_target_alpha1_solo",
                    help="e.g., -1.67 for Kolmogorov, -2.67 for Kinetic"
                )
                target_alpha2 = None
        
        submitted = st.form_submit_button("Update Plot", use_container_width=True, type="primary")

    def _parse_float_in_range(val_str: str, min_val: float, max_val: float, label: str):
        try:
            val = float(val_str)
        except Exception:
            st.error(f"⚠️ {label} must be a valid number")
            return None
        if val < min_val or val > max_val:
            st.error(f"⚠️ {label} must be between {min_val:.4f} and {max_val:.4f} Hz")
            return None
        return val
    
    # =========================================================================
    # PLOTTING (always render - auto-load on page open)
    # =========================================================================
    fit1_range = None
    fit2_range = None
    
    # Compute Fit 1 range
    if fit_mode == "Manual Range":
        fit1_fmin = _parse_float_in_range(fit1_fmin_str, f_min_data, f_max_data, "f₁ min")
        fit1_fmax = _parse_float_in_range(fit1_fmax_str, f_min_data, f_max_data, "f₁ max")
        if fit1_fmin is not None and fit1_fmax is not None:
            if fit1_fmin >= fit1_fmax:
                st.error("⚠️ f₁ min must be < f₁ max")
            else:
                fit1_range = (fit1_fmin, fit1_fmax)
    else:
        try:
            f1_min, f1_max, _ = find_target_alpha_range(psd.frequencies, psd.power, target_alpha1)
            fit1_range = (f1_min, f1_max)
        except Exception as e:
            st.error(f"Fit 1 range error: {e}")
    
    # Compute Fit 2 range (if enabled)
    if enable_dual_fit:
        if fit_mode == "Manual Range":
            fit2_fmin = _parse_float_in_range(fit2_fmin_str, f_min_data, f_max_data, "f₂ min")
            fit2_fmax = _parse_float_in_range(fit2_fmax_str, f_min_data, f_max_data, "f₂ max")
            if fit2_fmin is not None and fit2_fmax is not None:
                if fit2_fmin >= fit2_fmax:
                    st.error("⚠️ f₂ min must be < f₂ max")
                else:
                    fit2_range = (fit2_fmin, fit2_fmax)
        else:
            try:
                f2_min, f2_max, _ = find_target_alpha_range(psd.frequencies, psd.power, target_alpha2)
                fit2_range = (f2_min, f2_max)
            except Exception as e:
                st.error(f"Fit 2 range error: {e}")
    
    # Plot
    st.caption("💡 **Tip:** Point on the plot to see exact values • Click legend items to show/hide • Drag to zoom • Double-click to reset • Toolbar in top-right for more")
    
    try:
        # Construct dynamic title & metadata
        probe = info.get('probe', '?')
        instr = info.get('instrument', 'FGM').upper()
        coord = info.get('coord', 'GSM').upper()
        rate = info.get('data_rate', 'srvy')
        level = info.get('level', 'l2')
        dataset_str = f"MMS{probe} {instr} {coord} ({rate}/{level})"
        plot_title = f"{dataset_str} | {selected_col}"

        fig, alpha1, alpha2 = create_psd_plot(
            psd.frequencies, psd.power, title=plot_title,
            psd_units=units, fit1_range=fit1_range, fit2_range=fit2_range
        )
        st.plotly_chart(fig, use_container_width=False, config=PSD_CONFIG)
        
        # =================================================================
        # RESULTS - Clean, readable metrics
        # =================================================================
        if alpha1 is not None or alpha2 is not None:
            st.markdown("#### Spectral Indices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if alpha1 is not None and fit1_range:
                    st.metric(
                        label="α₁ (Inertial)",
                        value=f"{alpha1:.3f}",
                        delta=f"{fit1_range[0]:.3f} – {fit1_range[1]:.3f} Hz",
                        delta_color="off"
                    )
            
            with col2:
                if alpha2 is not None and fit2_range:
                    st.metric(
                        label="α₂ (Kinetic)",
                        value=f"{alpha2:.3f}",
                        delta=f"{fit2_range[0]:.3f} – {fit2_range[1]:.3f} Hz",
                        delta_color="off"
                    )
        
        # Metadata Footer
        st.markdown(
            f"<div style='font-size: 0.9em; color: #555; margin-top: 10px;'>"
            f"<b>Dataset:</b> {dataset_str}<br>"
            f"<b>Sampling Frequency (Fs):</b> {psd.sampling_frequency:.1f} Hz | "
            f"<b>Window Size:</b> {psd.nperseg} | "
            f"<b>Total Points (N):</b> {len(clean_data):,}"
            "</div>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Plotting error: {e}")


def render_pdf_analysis(datasets: dict, info: dict, subsample_pts: int):
    """Render PDF & Moments analysis."""
    import pandas as pd
    import plotly.graph_objects as go
    # Ensure physics functions are available
    from physics import compute_pdf_robust, cached_stats, compute_kde
    
    st.markdown("### PDF & Moments")
    
    # Filter to only include DataFrames (not CDFLoader objects)
    valid_datasets = {k: v for k, v in datasets.items() 
                      if isinstance(v, pd.DataFrame) and not v.empty}
    
    if not valid_datasets:
        st.warning("No valid data available for PDF analysis. Please load data first.")
        return
    
    # 1. UI CLEANUP: Automatically use the active dataset (first one)
    selected_key = list(valid_datasets.keys())[0]
    
    df = valid_datasets[selected_key]
    if subsample_pts:
        df = _subsample_df(df, subsample_pts)
        
    columns = list(df.columns)
    
    # Variable Selector (Top)
    c_sel, _ = st.columns([1, 2])
    with c_sel:
        selected_col = st.selectbox("Variable", columns, key="pdf_col")
    
    data = df[selected_col].values
    clean_data = data[np.isfinite(data)]
    
    # --- SPLIT LAYOUT ---
    c_left, c_right = st.columns([2, 1])
    
    # === LEFT: VISUALIZATION & CONTROLS ===
    with c_left:
        with st.container(border=True):
            st.markdown("##### Visualization Settings")
            
            # Controls Row 1
            c_bins, c_log = st.columns([2, 1])
            bins = c_bins.slider("Bins", 20, 200, 50, key="pdf_bins")
            logy = c_log.checkbox("Log Y", key="pdf_logy")

            # Controls Row 2 (Plot Type & Kernel)
            c_type, c_kernel = st.columns([1.5, 1])
            plot_type = c_type.radio("Plot Type", ["Histogram", "KDE (PDF Line)", "Combined"], horizontal=True, key="pdf_type")
            
            kde_kernel = 'gaussian'
            if plot_type in ["KDE (PDF Line)", "Combined"]:
                kde_kernel = c_kernel.selectbox(
                    "Kernel Type", 
                    ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'],
                    key="pdf_kernel"
                )
        
        # Plotting Logic
        try:
            fig = go.Figure()
            units = "nT" if 'B' in selected_col else "km/s"
            
            # Histogram Calculation
            if plot_type in ["Histogram", "Combined"]:
                with _timed("PDF compute", st.session_state.get("perf_telemetry", False)):
                    pdf_res = compute_pdf_robust(clean_data, bins)
                bin_width = pdf_res.bin_centers[1] - pdf_res.bin_centers[0] if len(pdf_res.bin_centers) > 1 else 1.0
                
                fig.add_trace(go.Bar(
                    x=pdf_res.bin_centers, 
                    y=pdf_res.density,
                    name='Histogram',
                    marker_color='#1f77b4',
                    opacity=0.7 if plot_type == "Combined" else 1.0,
                    width=bin_width * 0.9
                ))
                
            # KDE Calculation
            if plot_type in ["KDE (PDF Line)", "Combined"]:
                with _timed("KDE compute", st.session_state.get("perf_telemetry", False)):
                    kde_x, kde_y = compute_kde(clean_data, kernel=kde_kernel, n_points=500)
                if len(kde_x) > 0:
                    fig.add_trace(go.Scatter(
                        x=kde_x, y=kde_y, mode='lines', name=f'KDE ({kde_kernel})',
                        line=dict(color='#d62728', width=2.5)
                    ))
    
            # Styles matches TS/PSD
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', family='Arial, sans-serif'),
                title=None,
                xaxis_title=dict(text=f"{selected_col} [{units}]", font=dict(size=16, color='black')),
                yaxis_title=dict(text="Probability Density", font=dict(size=16, color='black')),
                height=550, # Slightly smaller to fit layout
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,0.95)', bordercolor='rgba(0,0,0,0.3)', borderwidth=1,
                    font=dict(size=14, color='black')
                ),
                margin=dict(l=70, r=40, t=20, b=60),
            )
            
            grid_color = '#E5E5E5'
            fig.update_xaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, tickfont=dict(size=13, color='black'), showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, tickfont=dict(size=13, color='black'), showline=True, linewidth=1, linecolor='black', mirror=True, zeroline=True)
            
            if logy:
                fig.update_yaxes(type="log")
                
            st.plotly_chart(fig, use_container_width=True, config={'editable': False, 'displayModeBar': True})
            
        except Exception as e:
            st.error(f"Plotting error: {e}")

    # === RIGHT: STATISTICAL SUMMARY ===
    with c_right:
        with st.container(border=True):
            st.markdown("#### Statistical Summary")
            try:
                with _timed("Stats compute", st.session_state.get("perf_telemetry", False)):
                    stats = cached_stats(clean_data)
                
                # 2 Columns x 5 Rows
                sc1, sc2 = st.columns(2)
                
                # Column 1
                sc1.metric("Mean", f"{stats.mean:.4g}")
                sc1.metric("Median", f"{stats.median:.4g}")
                sc1.metric("Std Dev", f"{stats.std:.4g}")
                sc1.metric("Variance", f"{stats.variance:.4g}")
                sc1.metric("Skewness", f"{stats.skewness:.4f}")
                
                # Column 2
                sc2.metric("Kurtosis", f"{stats.kurtosis:.4f}")
                sc2.metric("Min", f"{stats.min_val:.4g}")
                sc2.metric("Max", f"{stats.max_val:.4g}")
                sc2.metric("Samples", f"{stats.n_samples:,}")
                sc2.metric("NaNs", f"{stats.n_nan:,}")
                
            except Exception as e:
                st.error(f"Stats Error: {e}")


def render_summary_analysis(datasets: dict, info: dict, subsample_pts: int):
    """Render Summary Statistics analysis."""
    import pandas as pd
    st.markdown("### Summary Statistics")
    
    # Filter to only include DataFrames (not CDFLoader objects)
    valid_datasets = {k: v for k, v in datasets.items() 
                      if isinstance(v, pd.DataFrame) and not v.empty}
    
    if not valid_datasets:
        st.warning("No valid data available for Summary analysis. Please load data first.")
        return
    
    dataset_keys = list(valid_datasets.keys())
    selected_key = st.selectbox("Dataset", dataset_keys, key="sum_dataset")
    
    df = valid_datasets[selected_key]
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
            with _timed("Stats compute", st.session_state.get("perf_telemetry", False)):
                stats = cached_stats(clean_data)
            for n, v in create_stats_display(stats).items():
                st.text(f"{n}: {v}")
        except Exception as e:
            st.error(str(e))
    
    with c2:
        st.markdown("##### Quick PSD")
        try:
            with _timed("PSD compute", st.session_state.get("perf_telemetry", False)):
                psd = cached_psd(data, time_data)
            units = "nT²/Hz" if 'B' in selected_col else "km²/s²/Hz"
            fig, _ = create_psd_plot(psd.frequencies, psd.power, psd_units=units, height=350)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))


def render_pvi_analysis(datasets: dict, info: dict):
    """
    Render Partial Variance of Increments (PVI) analysis.
    
    Detects intermittent coherent structures using vector increments.
    """
    import plotly.graph_objects as go
    from physics import compute_pvi
    from plots import PLOTLY_CONFIG, PUBLICATION_LAYOUT, COLORS, GRID_COLOR
    
    st.markdown("### Partial Variance of Increments (PVI)")
    
    # 1. Definition Section
    st.markdown(r"""
    The **Partial Variance of Increments (PVI)** is a method to identify coherent structures and quantify intermittency.
    
    $$PVI(t, \tau) = \frac{|\Delta \mathbf{B}(t, \tau)|}{\sqrt{\langle |\Delta \mathbf{B}(t, \tau)|^2 \rangle}}$$
    
    where $\Delta \mathbf{B}(t, \tau) = \mathbf{B}(t + \tau) - \mathbf{B}(t)$ is the vector increment of the magnetic field.
    Values of $PVI > \theta$ (typically 3) indicate non-Gaussian, intermittent events.
    """)
    
    # 2. Controls
    with st.container():
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            lag = st.slider(
                "Time Lag (τ)", 
                min_value=1, 
                max_value=100, 
                value=1,
                help="Separation between data points for increment calculation."
            )
        with c2:
            threshold = st.number_input(
                "Detection Threshold (θ)",
                min_value=0.0,
                value=3.0,
                step=0.1,
                help="PVI level above which structures are identified (Greco et al. 2018)."
            )
        # Empty c3 for spacing
            
    st.markdown("---")
    
    # 3. Processing & Visualization per Dataset
    for key, df in datasets.items():
        # Clean data (ensure numeric)
        # Assuming df has Bx, By, Bz columns OR BL, BM, BN.
        cols = df.columns
        
        # 1. Try Standard XYZ
        bx = next((c for c in cols if c.upper() in ['BX', 'VX', 'EX', 'X']), None)
        by = next((c for c in cols if c.upper() in ['BY', 'VY', 'EY', 'Y']), None)
        bz = next((c for c in cols if c.upper() in ['BZ', 'VZ', 'EZ', 'Z']), None)
        
        # 2. Try LMN (L->x, M->y, N->z)
        bl = next((c for c in cols if c in ['BL', 'VL', 'B_L']), None)
        bm = next((c for c in cols if c in ['BM', 'VM', 'B_M']), None)
        bn = next((c for c in cols if c in ['BN', 'VN', 'B_N']), None)
        
        if bx and by and bz:
            vector_cols = [bx, by, bz]
        elif bl and bm and bn:
            vector_cols = [bl, bm, bn]
        else:
            st.warning(f"Dataset {key} does not appear to have 3 vector components (XYZ or LMN). Skipping.")
            continue
            
        data_vectors = df[vector_cols].values
        clean_mask = np.isfinite(data_vectors).all(axis=1)
        clean_vectors = data_vectors[clean_mask]
        clean_time = df.index[clean_mask]
        
        if len(clean_vectors) < lag + 10:
            st.error(f"Not enough data in {key} for lag {lag}.")
            continue
            
        # Compute PVI
        with _timed("PVI compute", st.session_state.get("perf_telemetry", False)):
            pvi, kurtosis, rms = compute_pvi(clean_vectors, lag=lag)
        
        # Align time
        pvi_time = clean_time[:-lag]
        
        # --- Visualization ---
        st.subheader(f"PVI Series ({key})")
        # Standard Tip placed small right under title
        st.caption(f"💡 Tip: Adjust **Time Lag (τ)** to detect structures of different scales. (RMS: {rms:.2f} nT)")
        
        # Identify peaks
        peaks_mask = pvi > threshold
        n_peaks = np.sum(peaks_mask)
        max_pvi = np.max(pvi) if len(pvi) > 0 else 0
        
        # Plot
        fig = go.Figure()
        
        # Main PVI line (Black)
        fig.add_trace(go.Scatter(
            x=pvi_time,
            y=pvi,
            mode='lines',
            name='PVI',
            line=dict(color='black', width=1.5),
            hovertemplate='PVI: %{y:.2f}<extra></extra>'
        ))
        
        # Threshold Line (Standard Red)
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color=COLORS[3], 
            annotation_text=f"θ = {threshold}",
            annotation_position="top right",
            annotation_font=dict(color=COLORS[3])
        )
            
        # Apply Publication Layout STRICTLY
        layout = PUBLICATION_LAYOUT.copy()
        layout.update(
            title=dict(
                text=f"Partial Variance of Increments (Lag τ={lag})",
                font=dict(size=20, color='black'),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis_title=dict(
                text="Epoch [UTC]",
                font=dict(size=16, color='black')
            ),
            yaxis_title=dict(
                text="PVI Index",
                font=dict(size=16, color='black')
            ),
            height=450,
            showlegend=True
        )
        fig.update_layout(layout)
        
        # Match Grid Styling exactly
        grid_style = dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            tickfont=dict(size=13, color='black'),
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        )
        
        fig.update_xaxes(**grid_style)
        fig.update_xaxes(title_standoff=15)
        
        fig.update_yaxes(**grid_style)
        fig.update_yaxes(title_standoff=15, zeroline=True, zerolinecolor='rgba(0,0,0,0.3)')
        
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # --- Statistics Panel ---
        st.markdown("#### Statistics")
        sc1, sc2, sc3 = st.columns(3)
        
        with sc1:
            st.metric("Maximum PVI", f"{max_pvi:.2f}")
        with sc2:
            st.metric("Detected Structures", f"{n_peaks}", help=f"Number of points where PVI > {threshold}")
        with sc3:
            st.metric("Increment Kurtosis", f"{kurtosis:.2f}", help="Pearson Kurtosis of |ΔB|. Values > 3 indicate intermittency.")


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
