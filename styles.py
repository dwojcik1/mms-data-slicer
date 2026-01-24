"""
styles.py - Deep Space Theme
==============================
Global dark theme with glassmorphism for MMS Turbulence Analysis Suite.
"""

import streamlit as st


def apply_custom_css():
    """
    Apply Deep Space theme globally across the entire Streamlit app.
    Matches the liquid glass landing page aesthetic.
    """
    st.markdown("""
    <style>
    /* ==========================================================================
       GLOBAL BACKGROUND - Deep Space Gradient
       ========================================================================== */
    .stApp {
        background: 
            radial-gradient(ellipse at 15% 15%, rgba(88, 28, 135, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 85% 85%, rgba(15, 82, 186, 0.12) 0%, transparent 50%),
            linear-gradient(135deg, #030712 0%, #0a0a1a 30%, #0d1020 60%, #030712 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Hide default header background */
    .stApp > header {
        background: transparent !important;
    }
    
    /* Main content area */
    .stApp [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    
    .stApp [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Block container */
    .stApp .block-container {
        background: transparent !important;
        max-width: 95% !important;
        padding: 2rem 2rem 4rem 2rem !important;
    }
    
    /* ==========================================================================
       SIDEBAR - Translucent Glass Panel
       ========================================================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            rgba(3, 7, 18, 0.92) 0%,
            rgba(10, 10, 26, 0.95) 100%
        ) !important;
        backdrop-filter: blur(20px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar content wrapper */
    [data-testid="stSidebarContent"] {
        background: transparent !important;
    }
    
    /* ==========================================================================
       GLOBAL TYPOGRAPHY - Light Text for Dark Background
       ========================================================================== */
    /* All text defaults */
    .stApp, .stApp p, .stApp span, .stApp div {
        color: rgba(248, 250, 252, 0.85);
    }
    
    /* Headers */
    .stApp h1 {
        color: rgba(248, 250, 252, 0.98) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: rgba(248, 250, 252, 0.95) !important;
        font-weight: 600 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p {
        color: rgba(248, 250, 252, 0.8) !important;
    }
    
    /* Caption text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: rgba(248, 250, 252, 0.5) !important;
    }
    
    /* Labels */
    .stApp label, .stSelectbox label, .stMultiSelect label {
        color: rgba(248, 250, 252, 0.75) !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5 {
        color: rgba(248, 250, 252, 0.95) !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(248, 250, 252, 0.7) !important;
    }
    
    /* ==========================================================================
       FILE UPLOADER - Dark Glass Style
       ========================================================================== */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"] > div:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: rgba(248, 250, 252, 0.6) !important;
    }
    
    /* Browse button */
    [data-testid="stFileUploader"] button {
        background: rgba(99, 102, 241, 0.2) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: rgba(248, 250, 252, 0.9) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: rgba(99, 102, 241, 0.35) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }
    
    /* ==========================================================================
       WIDGETS - Dark Theme Overrides
       ========================================================================== */
    /* Select boxes */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: rgba(248, 250, 252, 0.9) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Dropdown options */
    [data-baseweb="popover"] {
        background: rgba(20, 20, 35, 0.98) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    [data-baseweb="menu"] {
        background: transparent !important;
    }
    
    [data-baseweb="menu"] li {
        color: rgba(248, 250, 252, 0.8) !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background: rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Checkbox */
    .stCheckbox label span {
        color: rgba(248, 250, 252, 0.75) !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: rgba(248, 250, 252, 0.75) !important;
    }
    
    .stRadio > div {
        background: transparent !important;
    }
    
    /* Slider */
    .stSlider label {
        color: rgba(248, 250, 252, 0.75) !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: transparent !important;
    }
    
    /* ==========================================================================
       EXPANDERS - Glass Style
       ========================================================================== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        color: rgba(248, 250, 252, 0.85) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* ==========================================================================
       METRICS - Glass Cards
       ========================================================================== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stMetric"] label {
        color: rgba(248, 250, 252, 0.5) !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: rgba(248, 250, 252, 0.95) !important;
    }
    
    /* ==========================================================================
       DIVIDERS
       ========================================================================== */
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
        margin: 1.5rem 0 !important;
    }
    
    /* ==========================================================================
       ALERTS & MESSAGES
       ========================================================================== */
    .stAlert {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: rgba(248, 250, 252, 0.85) !important;
    }
    
    /* Error */
    [data-testid="stErrorMessage"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border-color: rgba(239, 68, 68, 0.3) !important;
    }
    
    /* Warning */
    [data-testid="stWarningMessage"] {
        background: rgba(245, 158, 11, 0.1) !important;
        border-color: rgba(245, 158, 11, 0.3) !important;
    }
    
    /* Info */
    [data-testid="stInfoMessage"] {
        background: rgba(59, 130, 246, 0.1) !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }
    
    /* ==========================================================================
       SPINNER
       ========================================================================== */
    .stSpinner > div {
        border-color: rgba(99, 102, 241, 0.3) !important;
        border-top-color: rgba(99, 102, 241, 0.9) !important;
    }
    
    /* ==========================================================================
       PLOTLY CHARTS - Dark Background
       ========================================================================== */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }
    
    /* ==========================================================================
       SCROLLBAR - Subtle Dark Style
       ========================================================================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
    
    /* ==========================================================================
       RESPONSIVE TWEAKS
       ========================================================================== */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        
        [data-testid="stSidebar"] {
            background: rgba(3, 7, 18, 0.98) !important;
        }
    }
    
    /* Large screens */
    @media (min-width: 1600px) {
        .block-container {
            max-width: 1400px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def get_responsive_columns():
    """Return appropriate column configuration."""
    return [1, 1, 1]
