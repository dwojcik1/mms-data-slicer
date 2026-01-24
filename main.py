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
# Liquid Glass CSS
# ============================================================================

LIQUID_GLASS_CSS = """
<style>
/* ==========================================================================
   AURORA BACKGROUND
   ========================================================================== */
.stApp {
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(88, 28, 135, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(15, 82, 186, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(6, 78, 59, 0.1) 0%, transparent 60%),
        linear-gradient(180deg, #030712 0%, #0a0a1a 50%, #030712 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* ==========================================================================
   HERO SECTION
   ========================================================================== */
.hero-wrapper {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
}

.hero-glow {
    font-size: clamp(2.8rem, 7vw, 5rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.1;
    background: linear-gradient(
        135deg, 
        #f8fafc 0%, 
        #e0e7ff 30%, 
        #a5b4fc 60%, 
        #818cf8 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 30px rgba(129, 140, 248, 0.3));
    margin-bottom: 1rem;
}

.hero-sub {
    font-size: clamp(1.1rem, 2.5vw, 1.5rem);
    font-weight: 400;
    color: rgba(248, 250, 252, 0.5);
    letter-spacing: 0.02em;
}

/* ==========================================================================
   LIQUID GLASS CARDS - Target Streamlit's bordered containers
   ========================================================================== */
section[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.08) 0%,
        rgba(255, 255, 255, 0.02) 100%
    ) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 28px !important;
    box-shadow: 
        0 0 0 1px rgba(255, 255, 255, 0.05) inset,
        0 20px 50px -12px rgba(0, 0, 0, 0.5),
        0 0 80px -20px rgba(99, 102, 241, 0.15) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    overflow: visible !important;
}

section[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-8px) scale(1.02) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
    box-shadow: 
        0 0 0 1px rgba(255, 255, 255, 0.1) inset,
        0 30px 60px -15px rgba(0, 0, 0, 0.6),
        0 0 100px -10px rgba(99, 102, 241, 0.25) !important;
}

section[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 2rem !important;
}

/* ==========================================================================
   CARD CONTENT STYLING
   ========================================================================== */
.glass-card-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: rgba(248, 250, 252, 0.95);
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}

.glass-card-body {
    font-size: 0.95rem;
    line-height: 1.7;
    color: rgba(248, 250, 252, 0.55);
}

.glass-card-body b, .glass-card-body strong {
    color: rgba(165, 180, 252, 0.95);
    font-weight: 600;
}

/* LaTeX in cards */
.stLatex {
    margin-bottom: 0.75rem !important;
}

.stLatex mjx-container {
    color: rgba(248, 250, 252, 0.9) !important;
}

/* ==========================================================================
   FOOTER
   ========================================================================== */
.glass-footer-text {
    text-align: center;
    padding: 3rem 1rem;
    font-size: 1rem;
    color: rgba(248, 250, 252, 0.3);
    letter-spacing: 0.03em;
}

/* ==========================================================================
   SIDEBAR
   ========================================================================== */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(3, 7, 18, 0.98) 0%,
        rgba(10, 10, 26, 0.98) 100%
    ) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] * {
    color: rgba(248, 250, 252, 0.7) !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: rgba(248, 250, 252, 0.9) !important;
}

/* ==========================================================================
   GENERAL OVERRIDES
   ========================================================================== */
.stApp [data-testid="stHeader"] {
    background: transparent !important;
}

.stApp .block-container {
    max-width: 1400px !important;
    padding: 1rem 2rem 4rem 2rem !important;
}

/* Divider */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg, 
        transparent 0%, 
        rgba(255, 255, 255, 0.08) 20%, 
        rgba(255, 255, 255, 0.08) 80%, 
        transparent 100%
    ) !important;
    margin: 2.5rem auto !important;
    max-width: 900px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    backdrop-filter: blur(12px) !important;
}

[data-testid="stMetric"] label {
    color: rgba(248, 250, 252, 0.4) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: rgba(248, 250, 252, 0.9) !important;
}

/* Text */
.stMarkdown p, .stText {
    color: rgba(248, 250, 252, 0.75);
}

h1, h2, h3, h4, h5 {
    color: rgba(248, 250, 252, 0.95) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 12px !important;
    color: rgba(248, 250, 252, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
</style>
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
# Landing Page
# ============================================================================

def show_landing_page():
    """Render the liquid glass landing page."""
    
    # Hero
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="hero-glow">Turbulence Analysis Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Time series processing for space plasma physics</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Row 1
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        with st.container(border=True):
            st.latex(r"\mathcal{P}(f) \text{ — Spectral Analysis}")
            st.markdown(
                '<p class="glass-card-body"><b>Welch PSD</b> estimation. '
                'Spectral indices <b>α</b> in inertial and kinetic ranges. '
                'Reference slopes for Kolmogorov turbulence.</p>',
                unsafe_allow_html=True
            )
    
    with c2:
        with st.container(border=True):
            st.latex(r"P(\delta B_\tau) \text{ — Stochastic Dynamics}")
            st.markdown(
                '<p class="glass-card-body"><b>PDFs & Moments</b>. '
                'Kurtosis <b>κ</b> and Skewness <b>S</b> for non-Gaussianity. '
                'Structure functions for intermittency.</p>',
                unsafe_allow_html=True
            )
    
    with c3:
        with st.container(border=True):
            st.latex(r"\mathbf{J} \cdot \mathbf{E}' \text{ — Dissipation}")
            st.markdown(
                '<p class="glass-card-body"><b>Energy conversion</b> analysis. '
                'EDR/IDR signatures, Hall fields, and magnetic reconnection events.</p>',
                unsafe_allow_html=True
            )
    
    # Row 2
    c4, c5, c6 = st.columns(3, gap="medium")
    
    with c4:
        with st.container(border=True):
            st.latex(r"\text{PVI} \text{ — Coherent Structures}")
            st.markdown(
                '<p class="glass-card-body"><b>Partial Variance of Increments</b>. '
                'Detection of current sheets, flux ropes, and dipolarization fronts.</p>',
                unsafe_allow_html=True
            )
    
    with c5:
        with st.container(border=True):
            st.latex(r"\delta B_\perp / \delta B_\parallel \text{ — Wave Modes}")
            st.markdown(
                '<p class="glass-card-body"><b>Compressibility</b> and magnetic helicity. '
                'Identification of Kinetic Alfvén Waves (KAW).</p>',
                unsafe_allow_html=True
            )
    
    with c6:
        with st.container(border=True):
            st.latex(r"\mathbf{X}(t) \text{ — Signal Integrity}")
            st.markdown(
                '<p class="glass-card-body"><b>Stationarity tests</b> (ADF). '
                'Despiking and interpolation for data quality.</p>',
                unsafe_allow_html=True
            )
    
    st.divider()
    
    st.markdown(
        '<p class="glass-footer-text">Select a dataset from the sidebar to begin</p>',
        unsafe_allow_html=True
    )


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.markdown(LIQUID_GLASS_CSS, unsafe_allow_html=True)
    
    st.sidebar.markdown("### Configuration")
    st.sidebar.markdown("##### Data Input")
    uploaded_file = st.sidebar.file_uploader("CDF File", type=['cdf'], label_visibility="collapsed")
    
    if uploaded_file is None:
        show_landing_page()
        return
    
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
    
    if mode == "Time Series":
        st.markdown("### Time Series")
        
        with st.sidebar.expander("Settings", expanded=True):
            sel = st.multiselect("Variables", plottable_vars, 
                                 default=plottable_vars[:min(3, len(plottable_vars))], format_func=fmt)
            sub = st.checkbox("Subsample", value=True)
            pts = st.slider("Points", 1000, 50000, 10000) if sub else len(time_data)
        
        if not sel:
            st.caption("Select variables.")
            return
        
        for var in sel:
            data = loader.get_variable_data(var)
            if data is None: continue
            meta = var_metadata[var]
            ylabel = f"{meta['short_label']} ({meta['units']})" if meta['units'] else meta['short_label']
            t, d = time_data, data
            if sub and len(time_data) > pts:
                step = len(time_data) // pts
                t, d = time_data[::step], (data[::step] if len(data.shape) == 1 else data[::step, :])
            fig = create_time_series_plot(t, d, title=meta['label'], ylabel=ylabel, component_labels=meta.get('components'))
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.markdown("### Spectral Analysis")
        
        with st.sidebar.expander("Variable", expanded=True):
            cats = loader.get_physics_variables()
            non_empty = [k for k, v in cats.items() if v]
            if not non_empty:
                st.caption("No variables.")
                return
            cat = st.selectbox("Category", non_empty, format_func=lambda x: x.replace('_', ' ').title())
            var = st.selectbox("Variable", cats.get(cat, []), format_func=fmt)
        
        meta = var_metadata[var]
        vdata = loader.get_variable_data(var)
        vinfo = loader.classify_variable(var)
        if vdata is None:
            st.error("Load error.")
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
        
        if method == "PSD":
            st.markdown(f"#### PSD: {comp_label}")
            try:
                psd = cached_psd(tuple(adata.flatten()), tuple(time_data.astype('datetime64[ns]').astype(np.int64)))
                fig = create_psd_plot(psd.frequencies, psd.power, title=f"PSD: {meta['label']}", psd_units=meta['psd_units'])
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Fs: {psd.sampling_frequency:.2f} Hz | Nperseg: {psd.nperseg}")
            except Exception as e:
                st.error(str(e))
        
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
                st.markdown("##### Stats")
                try:
                    stats = cached_stats(tuple(adata.flatten()))
                    for n, v in create_stats_display(stats).items():
                        st.metric(n, v)
                except Exception as e:
                    st.error(str(e))
        
        else:
            st.markdown(f"#### Summary: {meta['label']}")
            step = max(1, len(time_data) // 8000)
            ylabel = f"{comp_label} ({meta['units']})" if meta['units'] else comp_label
            fig = create_time_series_plot(time_data[::step], adata[::step], title=meta['label'], ylabel=ylabel, height=350)
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
                st.markdown("##### PSD")
                try:
                    psd = cached_psd(tuple(adata.flatten()), tuple(time_data.astype('datetime64[ns]').astype(np.int64)))
                    fig = create_psd_plot(psd.frequencies, psd.power, psd_units=meta['psd_units'], height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(str(e))


if __name__ == "__main__":
    main()
