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
    [data-testid="stSidebar"] * { color: #F5F7FA !important; }
            [data-testid="stSidebar"] input { 
    color: #FFFFFF !important; 
    background-color: #0F3460 !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background-color: #0F3460 !important;
    color: #FFFFFF !important;
}
            /* Dropdown options */
div[data-baseweb="popover"] ul {
    background-color: #0F3460 !important;
}

div[data-baseweb="popover"] li {
    color: #FFFFFF !important;
    background-color: #0F3460 !important;
}

div[data-baseweb="popover"] li:hover {
    color: #FFFFFF !important;
    background-color: #1A4A7A !important;
}

div[data-baseweb="popover"] * {
    color: #FFFFFF !important;
}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #BFC9D4 !important; font-size: 0.8rem !important; }

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

    /* ── Risk Flag Panel ── */
    .flag-panel {
        background: #FFFFFF;
        border: 1px solid #E0E6ED;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* ── Summary Panel ── */
    .summary-panel {
        background: #FFFFFF;
        border: 1px solid #E0E6ED;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Load & Train Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_excel('Pakistan_Auto_Lease_PD_Dataset.xlsx', sheet_name='Loan_Data')
    df = df.drop('Borrower_ID', axis=1)
    df['Guarantor'] = df['Guarantor'].map({'Yes': 1, 'No': 0})
    df['Previous_Relationship'] = df['Previous_Relationship'].map({'Yes': 1, 'No': 0})
    df['eCIB_Category'] = df['eCIB_Category'].map({'A':1,'B':2,'C':3,'E':4,'D':5})
    df = pd.get_dummies(df, columns=['Borrower_Type','Product_Type'], drop_first=True)
    df = df.drop('LTV_Percent', axis=1)
    X = df.drop('Default', axis=1)
    y = df['Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    model = LogisticRegression(max_iter=5000, random_state=42, solver='lbfgs', class_weight='balanced')
    model.fit(X_train, y_train)
    return model, scaler

model, scaler = load_model()

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="main-title">🚗 AutoLease PD Model</p>
    <p class="sub-title">Pakistan Vehicle Lease — Individual Borrower Credit Assessment</p>
    <div class="accent-bar"></div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:1.1rem; font-weight:800; color:#FFFFFF; margin-bottom:4px">📋 Borrower Input Panel</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#2D3561; margin:4px 0 12px 0"/>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Borrower Profile</p>', unsafe_allow_html=True)
    borrower_type = st.selectbox('Borrower Type', ['Salaried','Proprietor','Agriculturist'])
    age = st.slider('Age', 22, 65, 35)

    st.markdown('<p class="sidebar-section">Income & Obligations</p>', unsafe_allow_html=True)
    income = st.number_input('Net Monthly Income (PKR)', min_value=50000, max_value=2000000, value=150000, step=10000, format="%d")
    existing_obligations = st.number_input('Existing Monthly Obligations (PKR)', min_value=0, max_value=500000, value=0, step=5000, format="%d")
    proposed_installment = st.number_input('Proposed Lease Installment (PKR)', min_value=5000, max_value=500000, value=30000, step=5000, format="%d")

    st.markdown('<p class="sidebar-section">eCIB Assessment</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ecib-guide">
    Use eCIB report to determine grade:<br>
    ✦ Writeoff or settlement? → <b>Grade D</b><br>
    ✦ 90+ days overdue? → <b>Grade D</b><br>
    ✦ 30–89 days overdue? → <b>Grade C</b><br>
    ✦ 1–29 days overdue? → <b>Grade B</b><br>
    ✦ Previously overdue, cleared? → <b>Grade B</b><br>
    ✦ Clean — no overdue ever? → <b>Grade A</b><br>
    ✦ No record on eCIB? → <b>Grade E</b>
    </div>
    """, unsafe_allow_html=True)
    ecib = st.selectbox('Select eCIB Grade', ['A — Clean','B — Minor','C — Watchlist','D — Default','E — No Record'])

    st.markdown('<p class="sidebar-section">Stability</p>', unsafe_allow_html=True)
    stability = st.slider('Years in Job / Business', 1, 30, 5)

    st.markdown('<p class="sidebar-section">Lease Structure</p>', unsafe_allow_html=True)
    down_payment = st.slider('Down Payment %', 10, 40, 20)
    lease_rate = st.slider('Lease Rate % (Annual)', 22.0, 34.0, 26.0, step=0.5,
                            help="Higher lease rate = higher monthly burden = higher default risk")
    product_type = st.selectbox('Product Type', ['Direct Lease','Sale & Leaseback'])

    st.markdown('<p class="sidebar-section">Risk Mitigants</p>', unsafe_allow_html=True)
    guarantor = st.selectbox('Guarantor / Co-Borrower', ['Yes','No'],
                              help="Guarantor Yes reduces default risk")
    previous_relationship = st.selectbox('Previous Relationship', ['Yes','No'])

# ── Calculations ──────────────────────────────────────────────────────
dbr = (existing_obligations + proposed_installment) / income * 100

ecib_map_input = {'A — Clean':1,'B — Minor':2,'C — Watchlist':3,'E — No Record':4,'D — Default':5}
ecib_num       = ecib_map_input[ecib]
guarantor_num  = 0 if guarantor == 'Yes' else 1
prev_rel_num   = 1 if previous_relationship == 'Yes' else 0
type_prop      = 1 if borrower_type == 'Proprietor' else 0
type_sal       = 1 if borrower_type == 'Salaried' else 0
prod_slb       = 1 if product_type == 'Sale & Leaseback' else 0

input_data   = np.array([[age, income, existing_obligations, proposed_installment,
                           dbr, ecib_num, stability, down_payment, lease_rate,
                           guarantor_num, prev_rel_num, type_prop, type_sal, prod_slb]])
input_scaled = scaler.transform(input_data)
pd_score     = model.predict_proba(input_scaled)[0][1] * 100

if pd_score < 10:
    grade = 'A — Low Risk';   grade_class = 'grade-a'; score_class = 'pd-low'
elif pd_score < 25:
    grade = 'B — Acceptable'; grade_class = 'grade-b'; score_class = 'pd-med'
elif pd_score < 50:
    grade = 'C — Caution';    grade_class = 'grade-c'; score_class = 'pd-high'
else:
    grade = 'D — High Risk';  grade_class = 'grade-d'; score_class = 'pd-vhigh'

# ── Results ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    pd_color = 'metric-value-green' if pd_score < 25 else 'metric-value-orange' if pd_score < 50 else 'metric-value-red'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Probability of Default</div>
        <div class="{score_class}">{pd_score:.1f}%</div>
        <div style="margin-top:10px"><span class="{grade_class}">{grade}</span></div>
    </div>""", unsafe_allow_html=True)

with col2:
    dbr_class = 'dbr-warn' if dbr > 40 else 'dbr-ok'
    dbr_flag  = '▲ ABOVE 40% LIMIT' if dbr > 40 else '✔ WITHIN 40% LIMIT'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Debt Burden Ratio</div>
        <div class="metric-value-blue">{dbr:.1f}%</div>
        <div class="{dbr_class}" style="margin-top:8px">{dbr_flag}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Borrower Summary</div>
        <div style="text-align:left; margin-top:8px; font-size:0.85rem; color:#1A1A2E !important; line-height:1.8">
            <b>Type:</b> {borrower_type}<br>
            <b>Income:</b> PKR {income:,}<br>
            <b>eCIB:</b> {ecib.split('—')[0].strip()}<br>
            <b>Guarantor:</b> {guarantor}
        </div>
    </div>""", unsafe_allow_html=True)

# ── Decision ──────────────────────────────────────────────────────────
if pd_score < 25:
    st.markdown(f"""
    <div class="decision-approve">
        <b style="color:#1B4332; font-size:1.15rem">✅ RECOMMEND APPROVAL</b><br>
        <span style="color:#1B4332; font-size:0.9rem">
        Borrower presents acceptable credit risk. PD of <b>{pd_score:.1f}%</b> is within threshold.
        Proceed with standard documentation and verification.
        </span>
    </div>""", unsafe_allow_html=True)
elif pd_score < 50:
    st.markdown(f"""
    <div class="decision-caution">
        <b style="color:#7D3C00; font-size:1.15rem">⚡ SENIOR REVIEW REQUIRED</b><br>
        <span style="color:#7D3C00; font-size:0.9rem">
        Elevated risk detected. PD of <b>{pd_score:.1f}%</b> requires senior credit officer review.
        Consider additional collateral or guarantor before approval.
        </span>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="decision-reject">
        <b style="color:#7D0000; font-size:1.15rem">🚫 RECOMMEND REJECTION</b><br>
        <span style="color:#7D0000; font-size:0.9rem">
        High default probability of <b>{pd_score:.1f}%</b>. Borrower does not meet minimum credit standards.
        Application should be declined or significantly restructured.
        </span>
    </div>""", unsafe_allow_html=True)

# ── Risk Flags ────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Key Risk Flags</p>', unsafe_allow_html=True)

flags = []
if dbr > 40:           flags.append(("⚠", f"DBR {dbr:.1f}% exceeds 40% policy limit", "#C1121F"))
if ecib_num == 5:      flags.append(("⚠", "eCIB Grade D — Previous default history detected", "#C1121F"))
if ecib_num == 4:      flags.append(("⚠", "eCIB Grade E — No credit record, unknown risk", "#E07B39"))
if ecib_num == 3:      flags.append(("⚠", "eCIB Grade C — Borrower on watchlist", "#E07B39"))
if prod_slb:           flags.append(("⚠", "Sale & Leaseback — Liquidity stress signal", "#E07B39"))
if down_payment < 15:  flags.append(("⚠", "Low down payment — Weak skin in the game", "#E07B39"))
if guarantor_num == 1: flags.append(("ℹ", "No guarantor — Consider requiring one", "#0077B6"))
if not flags:          flags.append(("✅", "No major risk flags detected", "#2D6A4F"))

st.markdown('<div class="flag-panel">', unsafe_allow_html=True)
for icon, msg, color in flags:
    st.markdown(f'<p style="color:{color}; margin:5px 0; font-size:0.88rem; font-weight:600">{icon} {msg}</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown('<br><p style="color:#9BAAB8; font-size:0.72rem; text-align:center">Pakistan Auto Lease PD Model v1.0 — Built on synthetic data for demonstration purposes only</p>', unsafe_allow_html=True)