"""
styles.py - Dark SaaS Dashboard Theme
=======================================
Comprehensive visual overhaul: rounded shapes, pill buttons,
card containers, sidebar navigation, and purple accent palette.
Matches modern dark dashboard aesthetic.
"""

import streamlit as st


@st.cache_resource
def _get_cached_css() -> str:
    """Return cached CSS string to avoid re-parsing on every rerun."""
    return """
    <style>
    /* ==========================================================================
       USING STREAMLIT DEFAULT FONTS (Source Sans Pro, Source Serif, Source Code)
       No custom font imports - preserves default rendering
       ========================================================================== */

    /* ==========================================================================
       ROOT APPLICATION - DARK BACKGROUND
       ========================================================================== */
    .stApp {
        background: #12121b !important;
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
       HEADER - TRANSPARENT
       ========================================================================== */
    header[data-testid="stHeader"],
    .stApp header,
    .stApp [data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* ==========================================================================
       SIDEBAR - DARK PANEL WITH REFINED STYLING
       ========================================================================== */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    .stSidebar {
        background: #1a1a2e !important;
        background-color: #1a1a2e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background: transparent !important;
        background-color: transparent !important;
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
        color: #a0a0b8 !important;
    }
    
    /* ==========================================================================
       GLOBAL TEXT COLOR
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
        color: #e4e4ed !important;
    }
    
    /* Headers - bright white, heavier weight */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    .stApp h1 {
        font-size: 1.8rem !important;
    }
    
    /* ==========================================================================
       BUTTONS - PILL SHAPED WITH PURPLE GRADIENT
       ========================================================================== */
    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #a855f7 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3), 0 1px 3px rgba(0,0,0,0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.01em !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 50%, #9333ea 100%) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45), 0 2px 6px rgba(0,0,0,0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Secondary/default buttons */
    .stButton > button,
    [data-testid="baseButton-secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e4e4ed !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.5rem !important;
        transition: all 0.25s ease !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover {
        background: rgba(124, 58, 237, 0.12) !important;
        border-color: rgba(124, 58, 237, 0.3) !important;
        color: #ffffff !important;
    }

    /* Download button */
    [data-testid="stDownloadButton"] > button {
        background: rgba(124, 58, 237, 0.1) !important;
        border: 1px solid rgba(124, 58, 237, 0.25) !important;
        border-radius: 12px !important;
    }

    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(124, 58, 237, 0.2) !important;
        border-color: rgba(124, 58, 237, 0.4) !important;
    }

    /* ==========================================================================
       SELECT BOXES / DROPDOWNS - ROUNDED DARK
       ========================================================================== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    [data-baseweb="select"] > div {
        background: #22223a !important;
        background-color: #22223a !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        min-height: 44px !important;
    }

    .stSelectbox > div > div:hover,
    [data-baseweb="select"] > div:hover {
        border-color: rgba(124, 58, 237, 0.3) !important;
    }

    .stSelectbox > div > div:focus-within,
    [data-baseweb="select"] > div:focus-within {
        border-color: rgba(124, 58, 237, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
    }
    
    /* Dropdown menus */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="popover"] > div {
        background: #1e1e34 !important;
        background-color: #1e1e34 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5) !important;
        overflow: hidden !important;
    }
    
    [data-baseweb="menu"] li,
    [role="option"] {
        background: transparent !important;
        border-radius: 8px !important;
        margin: 2px 6px !important;
        transition: background 0.15s ease !important;
    }
    
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover {
        background: rgba(124, 58, 237, 0.15) !important;
    }

    /* ==========================================================================
       TEXT INPUTS / NUMBER INPUTS - ROUNDED DARK
       ========================================================================== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #22223a !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        color: #e4e4ed !important;
        padding: 0.5rem 0.75rem !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: rgba(124, 58, 237, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
    }

    /* Number input step buttons */
    .stNumberInput > div > div > div > button {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        color: #a0a0b8 !important;
    }
    
    .stNumberInput > div > div > div > button:hover {
        background: rgba(124, 58, 237, 0.15) !important;
        border-color: rgba(124, 58, 237, 0.3) !important;
        color: #ffffff !important;
    }

    /* Date input */
    [data-testid="stDateInput"] > div > div > input,
    .stDateInput > div > div > input {
        background: #22223a !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        color: #e4e4ed !important;
    }

    /* Time input select */
    [data-testid="stTimeInput"] [data-baseweb="select"] > div {
        background: #22223a !important;
        border-radius: 12px !important;
    }
    
    /* ==========================================================================
       RADIO BUTTONS - SEGMENTED PILL CONTROLS
       ========================================================================== */
    .stRadio > div {
        background: transparent !important;
    }

    /* Segmented control wrapper (horizontal radio) */
    .stRadio [role="radiogroup"] {
        gap: 4px !important;
    }

    .stRadio [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        padding: 6px 16px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    .stRadio [role="radiogroup"] label:hover {
        background: rgba(124, 58, 237, 0.1) !important;
        border-color: rgba(124, 58, 237, 0.25) !important;
    }

    /* Active/selected radio - purple filled */
    .stRadio [role="radiogroup"] label[data-checked="true"],
    .stRadio [role="radiogroup"] label:has(input:checked) {
        background: rgba(124, 58, 237, 0.2) !important;
        border-color: rgba(124, 58, 237, 0.4) !important;
    }
    
    /* Radio input circles - purple */
    .stRadio [role="radiogroup"] input[type="radio"] {
        accent-color: #7c3aed !important;
    }

    /* Checkbox */
    .stCheckbox {
        background: transparent !important;
    }

    .stCheckbox [data-testid="stCheckbox"] span[role="checkbox"] {
        border-radius: 6px !important;
    }

    /* ==========================================================================
       SLIDERS - PURPLE TRACK
       ========================================================================== */
    .stSlider > div {
        background: transparent !important;
    }

    /* Slider track */
    .stSlider [data-testid="stSlider"] > div > div > div {
        background: rgba(255, 255, 255, 0.08) !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #7c3aed !important;
        border-color: #7c3aed !important;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.4) !important;
    }

    /* Sidebar spacing */
    section[data-testid="stSidebar"] [data-testid="stSlider"] {
        margin-bottom: -0.35rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] {
        margin-top: -0.15rem !important;
    }

    /* ==========================================================================
       TOGGLE / SWITCH
       ========================================================================== */
    [data-testid="stCheckbox"] label span[data-testid="stCheckbox"] {
        border-radius: 6px !important;
    }

    /* ==========================================================================
       FILE UPLOADER - ROUNDED DARK DROPZONE
       ========================================================================== */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: #1e1e34 !important;
        background-color: #1e1e34 !important;
        border: 2px dashed rgba(124, 58, 237, 0.2) !important;
        border-radius: 16px !important;
    }
    
    [data-testid="stFileUploader"]:hover > div,
    [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(124, 58, 237, 0.06) !important;
        border-color: rgba(124, 58, 237, 0.35) !important;
    }
    
    /* Browse button in uploader */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ==========================================================================
       EXPANDERS - DARK ROUNDED CARDS
       ========================================================================== */
    [data-testid="stExpander"] {
        background: #1e1e34 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }

    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: transparent !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 12px 16px !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] summary:hover {
        background: rgba(124, 58, 237, 0.06) !important;
    }
    
    .streamlit-expanderContent,
    [data-testid="stExpander"] > div > div {
        background: transparent !important;
        border: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 12px 16px !important;
    }
    
    /* ==========================================================================
       METRICS - DARK CARD TILES
       ========================================================================== */
    [data-testid="stMetric"],
    [data-testid="stMetricValue"],
    [data-testid="metric-container"] {
        background: #1e1e34 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.2rem !important;
    }

    /* Bento Box Common Styles */
    .metric-box,
    .specs-box,
    .science-objectives {
        background: #1e1e34;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-box:hover,
    .specs-box:hover,
    .science-objectives:hover {
        transform: translateY(-2px);
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 0.85rem;
        color: rgba(248, 250, 252, 0.5) !important;
        line-height: 1.4;
    }

    .specs-box h4,
    .science-objectives h4 {
        margin-top: 0;
        margin-bottom: 12px;
        color: #c4b5fd !important;
        font-size: 1.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 8px;
    }

    .specs-box ul,
    .science-objectives ul {
        list-style-type: none;
        padding-left: 0;
        margin: 0;
        flex-grow: 1;
    }

    .specs-box li,
    .science-objectives li {
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: rgba(248, 250, 252, 0.7) !important;
        padding-left: 14px;
        position: relative;
    }

    .specs-box li::before,
    .science-objectives li::before {
        content: "•";
        color: #7c3aed;
        position: absolute;
        left: 0;
        font-weight: bold;
    }
    
    [data-testid="stMetric"] label {
        color: rgba(248, 250, 252, 0.45) !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        background: transparent !important;
        border: none !important;
        font-weight: 700 !important;
    }
    
    /* ==========================================================================
       TABS - ROUNDED SEGMENTED STYLE
       ========================================================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 14px !important;
        padding: 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        color: #a0a0b8 !important;
        border: none !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(124, 58, 237, 0.08) !important;
        color: #ffffff !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(124, 58, 237, 0.2) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Tab underline - hide default, we use background fill instead */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ==========================================================================
       DIVIDERS
       ========================================================================== */
    hr, .stDivider {
        border: none !important;
        height: 1px !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    
    /* ==========================================================================
       ALERTS / INFO BOXES
       ========================================================================== */
    .stAlert,
    [data-testid="stAlert"] {
        background: #1e1e34 !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* ==========================================================================
       MAIN CONTAINER WIDTH
       ========================================================================== */
    .block-container {
        max-width: 95% !important;
        padding: 2rem !important;
    }
    
    /* ==========================================================================
       DATA CONFIG CONTAINER - DARK CARD PANEL
       ========================================================================== */
    .data-config-container {
        max-width: 900px !important;
        margin: 0 auto !important;
        background: #1e1e34 !important;
        border-radius: 20px !important;
        padding: 32px 36px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2) !important;
    }
    
    .data-config-container h2,
    .data-config-container h3 {
        margin-top: 0 !important;
    }

    
    /* ==========================================================================
       GLASSMORPHIC ICONS - Dashboard Style
       ========================================================================== */
    .glass-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 14px;
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
            rgba(124, 58, 237, 0.1) 0%,
            rgba(167, 139, 250, 0.08) 100%
        );
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: inherit;
    }
    
    .glass-icon:hover {
        transform: translateY(-2px) scale(1.05);
        border-color: rgba(124, 58, 237, 0.35);
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.2);
    }
    
    .glass-icon:hover::before {
        opacity: 1;
    }
    
    .glass-icon .material-icons,
    .glass-icon span {
        font-size: 24px !important;
        color: #a78bfa !important;
        -webkit-text-fill-color: #a78bfa !important;
    }
    
    /* Smaller inline icon badge */
    .glass-icon-sm {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.12);
        border-radius: 8px;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .glass-icon-sm span {
        font-size: 15px !important;
        color: #a78bfa !important;
        -webkit-text-fill-color: #a78bfa !important;
    }

    /* ==========================================================================
       TOOLTIPS
       ========================================================================== */
    [data-testid="stTooltipIcon"] {
        color: #a0a0b8 !important;
    }

    /* ==========================================================================
       CAPTIONS - MUTED TEXT
       ========================================================================== */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: rgba(160, 160, 184, 0.7) !important;
    }
    .stCaption *, [data-testid="stCaptionContainer"] * {
        color: rgba(160, 160, 184, 0.7) !important;
    }

    /* ==========================================================================
       SCROLLBAR
       ========================================================================== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(124, 58, 237, 0.25);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(124, 58, 237, 0.4);
    }

    /* ==========================================================================
       COLUMNS - CARD GAP SPACING
       ========================================================================== */
    [data-testid="stHorizontalBlock"] {
        gap: 12px !important;
    }

    /* ==========================================================================
       DIALOG / MODAL OVERRIDE
       ========================================================================== */
    [data-testid="stModal"] > div {
        background: #1a1a2e !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5) !important;
    }

    /* ==========================================================================
       SPINNER
       ========================================================================== */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }

    /* ==========================================================================
       LINKS
       ========================================================================== */
    a {
        color: #a78bfa !important;
    }
    a:hover {
        color: #c4b5fd !important;
    }

    /* ==========================================================================
       PROGRESS BAR
       ========================================================================== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
        border-radius: 4px !important;
    }

    /* ==========================================================================
       MULTISELECT TAGS
       ========================================================================== */
    [data-baseweb="tag"] {
        background: rgba(124, 58, 237, 0.15) !important;
        border: 1px solid rgba(124, 58, 237, 0.25) !important;
        border-radius: 8px !important;
    }

    /* ==========================================================================
       FORM BORDERS
       ========================================================================== */
    [data-testid="stForm"] {
        background: #1e1e34 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    </style>
    """


def apply_custom_css():
    """
    Apply Dark SaaS Dashboard theme.
    Uses cached CSS string for performance.
    """
    st.markdown(_get_cached_css(), unsafe_allow_html=True)


def get_responsive_columns():
    """Return appropriate column configuration."""
    return [1, 1, 1]
