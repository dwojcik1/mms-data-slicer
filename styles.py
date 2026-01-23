"""
styles.py - Responsive Design System
=====================================
Mobile-first to Ultrawide responsive CSS for MMS Data Slicer.
"""

import streamlit as st


def apply_custom_css():
    """
    Apply responsive CSS for all screen sizes.
    Targets: Smartphones, Laptops, Large Displays (40"+), 4K monitors.
    """
    st.markdown("""
    <style>
    /* ================================================================
       ULTRAWIDE FIX: Expand container to full width
       ================================================================ */
    .block-container {
        max-width: 95% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* For very large screens (4K, conference monitors) */
    @media (min-width: 2000px) {
        .block-container {
            max-width: 98% !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }
    }
    
    /* ================================================================
       RESPONSIVE TYPOGRAPHY: Scale with viewport
       ================================================================ */
    .main-header {
        font-size: clamp(1.8rem, 4vw, 3.5rem) !important;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: clamp(1rem, 2vw, 1.4rem) !important;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    
    /* Scale all headings responsively */
    h1 { font-size: clamp(1.5rem, 3vw, 2.5rem) !important; }
    h2 { font-size: clamp(1.3rem, 2.5vw, 2rem) !important; }
    h3 { font-size: clamp(1.1rem, 2vw, 1.5rem) !important; }
    
    /* Large screen enhancements */
    @media (min-width: 1600px) {
        .stMetric label { font-size: 1.1rem !important; }
        .stMetric [data-testid="stMetricValue"] { font-size: 2rem !important; }
    }
    
    /* ================================================================
       MOBILE OPTIMIZATIONS
       ================================================================ */
    @media (max-width: 768px) {
        /* Larger touch targets for radio buttons */
        .stRadio > div {
            gap: 0.75rem !important;
        }
        .stRadio label {
            padding: 0.75rem 1rem !important;
            font-size: 1rem !important;
        }
        
        /* Larger multiselect items */
        .stMultiSelect [data-baseweb="tag"] {
            padding: 0.5rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Compact sidebar on mobile */
        [data-testid="stSidebar"] {
            min-width: 280px !important;
        }
        
        /* Stack columns vertically */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        
        /* Reduce padding */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
        }
        
        /* Larger buttons */
        .stButton button {
            padding: 0.75rem 1.5rem !important;
            font-size: 1rem !important;
        }
    }
    
    /* ================================================================
       HIDE STREAMLIT DEFAULTS FOR CLEANER LOOK
       ================================================================ */
    /* Hide hamburger menu on mobile */
    #MainMenu {
        visibility: hidden;
    }
    
    /* Hide footer */
    footer {
        visibility: hidden;
    }
    
    /* Hide "Made with Streamlit" */
    footer:after {
        visibility: hidden;
    }
    
    /* ================================================================
       SIDEBAR ENHANCEMENTS
       ================================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    [data-testid="stSidebar"] .stRadio label {
        background: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.25rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #e3f2fd;
    }
    
    /* ================================================================
       METRIC CARDS
       ================================================================ */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3d7ab0;
    }
    
    /* ================================================================
       EXPANDER STYLING
       ================================================================ */
    .streamlit-expanderHeader {
        font-size: clamp(0.95rem, 1.5vw, 1.1rem) !important;
        font-weight: 600;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    /* ================================================================
       CHART CONTAINERS
       ================================================================ */
    [data-testid="stPlotlyChart"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Ensure charts fill container on all screens */
    .js-plotly-plot, .plotly {
        width: 100% !important;
    }
    
    /* ================================================================
       TABLET OPTIMIZATIONS
       ================================================================ */
    @media (min-width: 769px) and (max-width: 1199px) {
        .block-container {
            max-width: 90% !important;
        }
    }
    
    /* ================================================================
       DARK MODE SUPPORT (if user prefers)
       ================================================================ */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(90deg, #5d9ad0 0%, #3d7ab0 50%, #2d5a87 100%);
            -webkit-background-clip: text;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def get_responsive_columns():
    """
    Return appropriate column configuration based on screen context.
    For Streamlit, we can't detect screen size server-side,
    but we design for natural wrapping.
    """
    return [1, 1, 1]  # Equal columns that wrap naturally
