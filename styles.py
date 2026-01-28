"""
styles.py - Deep Space Theme (Aggressive Override)
====================================================
Forces dark theme across ALL Streamlit elements.
"""

import streamlit as st


def apply_custom_css():
    """
    Apply Deep Space theme with aggressive !important overrides.
    Targets ROOT DOM elements to eliminate white backgrounds.
    """
    st.markdown("""
    <style>
    /* ==========================================================================
       FONT IMPORT - SPACE GROTESK
       ========================================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global font application - exclude icon fonts */
    html, body, [class*="st-"], .stApp, .stApp * {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Preserve Material Icons for Streamlit UI icons */
    .material-icons, 
    [class*="icon"], 
    [data-testid*="icon"],
    .stSidebar button span,
    [aria-label*="Collapse"],
    [aria-label*="Expand"] {
        font-family: 'Material Icons', 'Material Symbols Rounded', sans-serif !important;
    }

    
    /* ==========================================================================
       ROOT APPLICATION - ELIMINATE WHITE BACKGROUND
       ========================================================================== */
    .stApp {
        background: linear-gradient(135deg, #030712 0%, #0b0f19 30%, #0d1020 70%, #030712 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* App view container */
    .stApp > div,
    .stApp [data-testid="stAppViewContainer"],
    .stApp [data-testid="stAppViewContainer"] > div,
    .main .block-container,
    [data-testid="stAppViewBlockContainer"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* ==========================================================================
       HEADER - MAKE TRANSPARENT
       ========================================================================== */
    header[data-testid="stHeader"],
    .stApp header,
    .stApp [data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* ==========================================================================
       SIDEBAR - PRONOUNCED DARK PANEL
       ========================================================================== */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    .stSidebar {
        background: #111625 !important;
        background-color: #111625 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 2px 0 15px rgba(0, 0, 0, 0.5) !important;
    }
    
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    
    /* ==========================================================================
       GLOBAL TEXT COLOR - WHITE ON DARK
       ========================================================================== */
    .stApp,
    .stApp *,
    .stMarkdown,
    .stMarkdown *,
    .stText,
    h1, h2, h3, h4, h5, h6,
    p, span, label, div,
    .stApp p,
    .stApp span,
    .stApp label {
        color: #E8E8E8 !important;
    }
    
    /* Headers brighter */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #F8FAFC !important;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5 {
        color: #D0D0D0 !important;
    }
    
    /* ==========================================================================
       FILE UPLOADER - DARK GLASS STYLE
       ========================================================================== */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.03) !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stFileUploader"]:hover > div,
    [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    
    /* Browse button */
    [data-testid="stFileUploader"] button,
    [data-testid="baseButton-secondary"] {
        background: rgba(129, 140, 248, 0.15) !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
        color: #F0F0F0 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: rgba(129, 140, 248, 0.3) !important;
        border-color: rgba(129, 140, 248, 0.5) !important;
    }
    
    /* ==========================================================================
       WIDGETS - ALL DARK
       ========================================================================== */
    /* Select boxes */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.04) !important;
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown menus */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="popover"] > div {
        background: rgba(20, 20, 40, 0.98) !important;
        background-color: rgba(20, 20, 40, 0.98) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-baseweb="menu"] li,
    [role="option"] {
        background: transparent !important;
    }
    
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover {
        background: rgba(129, 140, 248, 0.2) !important;
    }
    
    /* Checkbox and Radio */
    .stCheckbox, .stRadio {
        background: transparent !important;
    }
    
    /* Slider */
    .stSlider > div {
        background: transparent !important;
    }
    
    /* ==========================================================================
       EXPANDERS - GLASS STYLE
       ========================================================================== */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
    }
    
    .streamlit-expanderContent,
    [data-testid="stExpander"] > div > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-top: none !important;
    }
    
    /* ==========================================================================
       METRICS - GLASS CARDS
       ========================================================================== */
    [data-testid="stMetric"],
    [data-testid="stMetricValue"],
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }
    
    [data-testid="stMetric"] label {
        color: rgba(248, 250, 252, 0.5) !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        background: transparent !important;
        border: none !important;
    }
    
    /* ==========================================================================
       DIVIDERS
       ========================================================================== */
    hr, .stDivider {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08) 20%, rgba(255,255,255,0.08) 80%, transparent) !important;
    }
    
    /* ==========================================================================
       ALERTS
       ========================================================================== */
    .stAlert, [data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px !important;
    }
    
    /* ==========================================================================
       MAIN CONTAINER WIDTH
       ========================================================================== */
    .block-container {
        max-width: 95% !important;
        padding: 2rem !important;
    }
    
    /* ==========================================================================
       DATA CONFIG CONTAINER - CENTERED GLASS PANEL
       ========================================================================== */
    .data-config-container {
        max-width: 900px !important;
        margin: 0 auto !important;
        background: rgba(15, 20, 35, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .data-config-container h2,
    .data-config-container h3 {
        margin-top: 0 !important;
    }

    
    /* ==========================================================================
       SCROLLBAR
       ========================================================================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.3);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.15);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.25);
    }
    </style>
    """, unsafe_allow_html=True)


def get_responsive_columns():
    """Return appropriate column configuration."""
    return [1, 1, 1]
