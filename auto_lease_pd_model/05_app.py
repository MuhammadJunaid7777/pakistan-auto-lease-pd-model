import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AutoLease PD Model",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    /* ── Base ── */
    .stApp { background-color: #F5F7FA; }
    .stApp, .stMarkdown, p, div { color: #1A1A2E !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1A1A2E;
        border-right: 3px solid #E63946;
    }

    [data-testid="stSidebar"] * { 
        color: #F5F7FA !important; 
    }

    /* INPUT inside sidebar */
    [data-testid="stSidebar"] input { 
        color: #FFFFFF !important; 
        background-color: #0F3460 !important;
    }

    /* SELECT BOX (main container) */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #0F3460 !important;
        color: #FFFFFF !important;
    }

    /* ─────────────────────────────────────────────
       ✅ FIX: DROPDOWN VISIBILITY ISSUE (IMPORTANT)
       ───────────────────────────────────────────── */

    /* Dropdown overlay (prevents hidden behind UI) */
    div[data-baseweb="popover"] {
        z-index: 9999 !important;
    }

    /* Dropdown menu container */
    div[data-baseweb="menu"] {
        background-color: #0F3460 !important;
        border: 1px solid #2D3561 !important;
    }

    /* Dropdown list */
    div[data-baseweb="menu"] ul {
        background-color: #0F3460 !important;
    }

    /* Dropdown items (THIS WAS THE MAIN ISSUE) */
    div[data-baseweb="menu"] li {
        color: #FFFFFF !important;
        background-color: #0F3460 !important;
    }

    /* Hover state */
    div[data-baseweb="menu"] li:hover {
        background-color: #16213E !important;
        color: #FFFFFF !important;
    }

    /* Selected item */
    div[aria-selected="true"] {
        background-color: #16213E !important;
        color: #FFD166 !important;
    }

    /* Sidebar labels */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #BFC9D4 !important;
        font-size: 0.8rem !important;
    }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 60%, #0F3460 100%);
        padding: 28px 32px 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        border-left: 5px solid #E63946;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 900;
        color: #FFFFFF !important;
        letter-spacing: -1px;
        margin: 0;
        line-height: 1.1;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #BFC9D4 !important;
        margin-top: 6px;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .accent-bar {
        width: 60px;
        height: 3px;
        background: #E63946;
        margin: 10px 0 0 0;
        border-radius: 2px;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E0E6ED;
        border-radius: 10px;
        padding: 22px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .metric-label {
        font-size: 0.72rem;
        color: #6B7A8D !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value-blue { font-size: 2.2rem; font-weight: 900; color: #0F3460 !important; }
    .metric-value-green { font-size: 2.2rem; font-weight: 900; color: #2D6A4F !important; }
    .metric-value-orange { font-size: 2.2rem; font-weight: 900; color: #E07B39 !important; }
    .metric-value-red { font-size: 2.2rem; font-weight: 900; color: #E63946 !important; }

    /* ── PD Score ── */
    .pd-low   { color: #2D6A4F !important; font-size: 3.2rem; font-weight: 900; line-height:1; }
    .pd-med   { color: #E07B39 !important; font-size: 3.2rem; font-weight: 900; line-height:1; }
    .pd-high  { color: #C1121F !important; font-size: 3.2rem; font-weight: 900; line-height:1; }
    .pd-vhigh { color: #9D0208 !important; font-size: 3.2rem; font-weight: 900; line-height:1; }

    /* ── Grade Badges ── */
    .grade-a { background:#D8F3DC; color:#1B4332; padding:6px 18px; border-radius:20px; font-weight:800; font-size:1rem; border:2px solid #2D6A4F; }
    .grade-b { background:#D0E8FF; color:#023E8A; padding:6px 18px; border-radius:20px; font-weight:800; font-size:1rem; border:2px solid #0077B6; }
    .grade-c { background:#FFE8CC; color:#7D3C00; padding:6px 18px; border-radius:20px; font-weight:800; font-size:1rem; border:2px solid #E07B39; }
    .grade-d { background:#FFD6D6; color:#7D0000; padding:6px 18px; border-radius:20px; font-weight:800; font-size:1rem; border:2px solid #E63946; }

    /* ── Decision Boxes ── */
    .decision-approve {
        background: #D8F3DC;
        border: 2px solid #2D6A4F;
        border-left: 6px solid #2D6A4F;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 16px 0;
    }

    .decision-caution {
        background: #FFF3CD;
        border: 2px solid #E07B39;
        border-left: 6px solid #E07B39;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 16px 0;
    }

    .decision-reject {
        background: #FFD6D6;
        border: 2px solid #E63946;
        border-left: 6px solid #E63946;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 16px 0;
    }

    /* ── Section Labels ── */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #E63946 !important;
        font-weight: 700;
        border-bottom: 2px solid #E0E6ED;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }

    .sidebar-section {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #E63946 !important;
        font-weight: 700;
        border-bottom: 1px solid #2D3561;
        padding-bottom: 5px;
        margin: 14px 0 8px 0;
    }

    /* ── DBR Status ── */
    .dbr-ok   { color: #2D6A4F !important; font-weight: 700; font-size: 0.82rem; }
    .dbr-warn { color: #E63946 !important; font-weight: 700; font-size: 0.82rem; }

    /* ── eCIB Guide ── */
    .ecib-guide {
        background: #0F3460;
        border: 1px solid #E63946;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.8rem;
        line-height: 1.9;
        color: #F5F7FA !important;
    }

    .ecib-guide b { color: #FFD166 !important; }

    /* ── Panels ── */
    .flag-panel,
    .summary-panel {
        background: #FFFFFF;
        border: 1px solid #E0E6ED;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)