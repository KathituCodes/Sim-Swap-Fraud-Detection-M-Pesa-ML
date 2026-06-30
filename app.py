import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PAGE CONFIG — must be the very first Streamlit call
st.set_page_config(
    page_title="M-Pesa Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SAFARICOM BRAND COLORS
SAF_GREEN      = "#30B54A"
SAF_DARK_GREEN = "#006633"
SAF_RED        = "#E2001A"
SAF_LIGHT      = "#F0FAF2"
SAF_GREY       = "#F5F5F5"
SAF_DARK_TEXT  = "#1A1A1A"
SAF_MID_TEXT   = "#555555"
WHITE          = "#FFFFFF"

st.markdown(f"""
<style>
    .main {{ background-color: #F8FAF8; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {SAF_DARK_GREEN} 0%, #004d26 100%);
    }}
    [data-testid="metric-container"] {{
        background-color: {WHITE};
        border: 1px solid #E0E0E0;
        border-left: 5px solid {SAF_GREEN};
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .fraud-alert {{
        background: linear-gradient(135deg, #FFF0F0, #FFE5E5);
        border: 2px solid {SAF_RED};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(226,0,26,0.15);
    }}
    .legit-alert {{
        background: linear-gradient(135deg, #F0FAF2, #E8F5E9);
        border: 2px solid {SAF_GREEN};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(48,181,74,0.15);
    }}
    .info-box {{
        background-color: {SAF_LIGHT};
        border-left: 5px solid {SAF_GREEN};
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 14px;
        color: {SAF_DARK_GREEN};
        line-height: 1.6;
    }}
    .warning-box {{
        background-color: #FFF8E1;
        border-left: 5px solid #FFA000;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 14px;
        color: #5D4037;
        line-height: 1.6;
    }}
    .section-header {{
        background: linear-gradient(90deg, {SAF_DARK_GREEN}, {SAF_GREEN});
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 16px;
        letter-spacing: 0.3px;
    }}
    .stat-card {{
        background: white;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #E8F5E9;
        border-top: 4px solid {SAF_GREEN};
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        height: 100%;
    }}
    .stNumberInput label, .stSlider label, .stSelectbox label,
    .stNumberInput p, .stSlider p, .stSelectbox p,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p {{
        font-weight: 600 !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }}
    .stSlider [data-testid="stWidgetLabel"],
    .stNumberInput [data-testid="stWidgetLabel"],
    .stSelectbox [data-testid="stWidgetLabel"] {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, {SAF_DARK_GREEN}, {SAF_GREEN});
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 32px;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(48,181,74,0.3);
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE — persists values across Streamlit reruns
# This is required for the Quick Test buttons to work correctly.
# When a button is clicked, we write the test values into
# session state. The input widgets then read from session state
# as their default values on the next rerun.
# ─────────────────────────────────────────────────────────────
def init_state():
    if 'amount'   not in st.session_state:
        st.session_state.amount          = 500.0
        st.session_state.sender_before   = 5000.0
        st.session_state.sender_after    = 4500.0
        st.session_state.receiver_before = 1000.0
        st.session_state.receiver_after  = 1500.0
        st.session_state.hour            = 14
        st.session_state.txn_type        = "peer"
        st.session_state.device          = "smartphone"
        st.session_state.region          = "Nairobi"
        st.session_state.day             = "Tue"
        # Scorecard inputs
        st.session_state.woe_amount      = 500.0
        st.session_state.woe_s_before    = 5000.0
        st.session_state.woe_s_after     = 4500.0

init_state()

def load_fraud():
    """Load a clear SIM swap fraud example into session state.
    Fraud pattern: nearly entire balance drained in one transaction.
    Sender starts with KES 5,000, sends KES 4,900, leaves only KES 50.
    sender_balance_ratio = 50/5001 = 0.01 (near zero = account drained)
    is_balance_wipeout = 1 (balance below KES 100)
    is_high_value = 1 (amount 4900 > 3800 threshold)
    """
    st.session_state.amount          = 4900.0
    st.session_state.sender_before   = 5000.0
    st.session_state.sender_after    = 50.0
    st.session_state.receiver_before = 1000.0
    st.session_state.receiver_after  = 5900.0
    st.session_state.hour            = 2
    st.session_state.txn_type        = "peer"
    st.session_state.device          = "smartphone"
    st.session_state.region          = "Nairobi"
    st.session_state.day             = "Tue"
    st.session_state.woe_amount      = 4900.0
    st.session_state.woe_s_before    = 5000.0
    st.session_state.woe_s_after     = 50.0

def load_legit():
    """Load a clear legitimate transaction example into session state.
    Legitimate pattern: small routine payment, most of balance remains.
    Sender starts with KES 25,000, sends KES 300, leaves KES 24,700.
    sender_balance_ratio = 24700/25001 = 0.988 (near 1 = almost nothing spent)
    is_balance_wipeout = 0
    is_high_value = 0
    """
    st.session_state.amount          = 300.0
    st.session_state.sender_before   = 25000.0
    st.session_state.sender_after    = 24700.0
    st.session_state.receiver_before = 3000.0
    st.session_state.receiver_after  = 3300.0
    st.session_state.hour            = 14
    st.session_state.txn_type        = "paybill"
    st.session_state.device          = "smartphone"
    st.session_state.region          = "Nairobi"
    st.session_state.day             = "Wed"
    st.session_state.woe_amount      = 300.0
    st.session_state.woe_s_before    = 25000.0
    st.session_state.woe_s_after     = 24700.0


# LOAD MODELS
@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgboost_fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    woe_model = joblib.load("models/woe_scorecard_model.pkl")
    with open("models/woe_bins.pkl", "rb") as f:
        woe_bins = pickle.load(f)
    return xgb_model, scaler, woe_model, woe_bins

xgb_model, scaler, woe_model, woe_bins = load_models()


# FEATURE ENGINEERING
def engineer_features(amount, sender_bal_before, sender_bal_after,
                      receiver_bal_before, receiver_bal_after,
                      hour, transaction_type, device_type,
                      region, day_of_week):

    balance_depletion_rate = amount / (sender_bal_before + 1)
    is_high_value          = 1 if amount > 3800 else 0
    is_balance_wipeout     = 1 if sender_bal_after < 100 else 0
    sender_balance_ratio   = sender_bal_after / (sender_bal_before + 1)

    txn_peer       = 1 if transaction_type == "peer"       else 0
    txn_till       = 1 if transaction_type == "till"       else 0
    dev_smartphone = 1 if device_type      == "smartphone" else 0
    reg_kisumu     = 1 if region           == "Kisumu"     else 0
    reg_mombasa    = 1 if region           == "Mombasa"    else 0
    reg_nairobi    = 1 if region           == "Nairobi"    else 0
    reg_nakuru     = 1 if region           == "Nakuru"     else 0
    day_mon        = 1 if day_of_week      == "Mon"        else 0
    day_sat        = 1 if day_of_week      == "Sat"        else 0
    day_sun        = 1 if day_of_week      == "Sun"        else 0
    day_thu        = 1 if day_of_week      == "Thu"        else 0
    day_tue        = 1 if day_of_week      == "Tue"        else 0
    day_wed        = 1 if day_of_week      == "Wed"        else 0

    features = {
        'amount':                    amount,
        'sender_balance_before':     sender_bal_before,
        'sender_balance_after':      sender_bal_after,
        'receiver_balance_before':   receiver_bal_before,
        'receiver_balance_after':    receiver_bal_after,
        'hour':                      hour,
        'transaction_type_peer':     txn_peer,
        'transaction_type_till':     txn_till,
        'device_type_smartphone':    dev_smartphone,
        'region_Kisumu':             reg_kisumu,
        'region_Mombasa':            reg_mombasa,
        'region_Nairobi':            reg_nairobi,
        'region_Nakuru':             reg_nakuru,
        'day_of_week_Mon':           day_mon,
        'day_of_week_Sat':           day_sat,
        'day_of_week_Sun':           day_sun,
        'day_of_week_Thu':           day_thu,
        'day_of_week_Tue':           day_tue,
        'day_of_week_Wed':           day_wed,
        'balance_depletion_rate':    balance_depletion_rate,
        'is_high_value':             is_high_value,
        'is_balance_wipeout':        is_balance_wipeout,
        'sender_balance_ratio':      sender_balance_ratio,
    }

    return pd.DataFrame([features]), {
        'balance_depletion_rate': balance_depletion_rate,
        'is_high_value':          is_high_value,
        'is_balance_wipeout':     is_balance_wipeout,
        'sender_balance_ratio':   sender_balance_ratio
    }


def prepare_woe_input(amount, sender_bal_after, sender_balance_ratio, is_high_value):
    """
    Manually applies WoE bin transformation without scorecardpy.
    Replicates sc.woebin_ply() using saved bin boundaries from woe_bins.pkl.
    scorecardpy is not used at runtime because it requires pkg_resources
    which is unavailable on Python 3.11+ (Streamlit Cloud environment).
    """
    def get_woe(value, bin_df):
        for _, row in bin_df.iterrows():
            bin_range = str(row['bin'])
            if bin_range == 'missing':
                continue
            bin_range = bin_range.replace('[-inf', '(-inf')
            try:
                lower    = float(bin_range.split(',')[0].strip('([)]-inf ').replace('-inf', '-999999999'))
                upper    = float(bin_range.split(',')[1].strip('([)] inf').replace('inf', '999999999'))
                low_inc  = '[' in bin_range.split(',')[0]
                high_inc = ']' in bin_range.split(',')[1]
                lower_check = value >= lower if low_inc  else value > lower
                upper_check = value <= upper if high_inc else value < upper
                if lower_check and upper_check:
                    return float(row['woe'])
            except Exception:
                continue
        return 0.0

    # Build WoE scores for each feature using saved bin boundaries
    amount_woe               = get_woe(amount,               woe_bins['amount'])
    is_high_value_woe        = get_woe(is_high_value,        woe_bins['is_high_value'])
    sender_balance_ratio_woe = get_woe(sender_balance_ratio, woe_bins['sender_balance_ratio'])
    sender_balance_after_woe = get_woe(sender_bal_after,     woe_bins['sender_balance_after'])

    # Column order must exactly match woe_model.feature_names_in_
    # ['amount_woe', 'is_high_value_woe', 'sender_balance_ratio_woe', 'sender_balance_after_woe']
    df = pd.DataFrame([{
        'amount_woe':               amount_woe,
        'is_high_value_woe':        is_high_value_woe,
        'sender_balance_ratio_woe': sender_balance_ratio_woe,
        'sender_balance_after_woe': sender_balance_after_woe,
    }])
    return df


# SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:40px;'>🔐</div>
        <div style='color:white; font-size:18px; font-weight:800;
                    letter-spacing:0.5px; margin-top:8px;'>
            M-Pesa Fraud Detector
        </div>
        <div style='color:#A8D5B5; font-size:11px; margin-top:4px;'>
            by Kat.Codes
        </div>
    </div>
    <hr style='border-color:#004d26; margin:12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color:#A8D5B5; font-size:12px; font-weight:600; margin-bottom:8px;'>NAVIGATE</p>",
                unsafe_allow_html=True)

    page = st.radio(
        label="",
        options=["🏠 Home", "🔍 Check a Transaction", "📋 Scorecard Audit", "📊 Model Results"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#004d26; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='color:#A8D5B5; font-size:12px; font-weight:600; margin-bottom:8px;'>QUICK TEST</p>",
                unsafe_allow_html=True)

    # on_click loads the values into session state BEFORE the page reruns
    # This ensures the input widgets always read the correct values
    st.button("⚠️ Load Fraud Example",  use_container_width=True, on_click=load_fraud)
    st.button("✅ Load Safe Example",    use_container_width=True, on_click=load_legit)

    st.markdown("<hr style='border-color:#004d26; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='color:#A8D5B5; font-size:11px; line-height:1.9;'>
        <p style='color:white; font-size:12px; font-weight:600; margin-bottom:6px;'>DATASET</p>
        📱 120,000 M-Pesa transactions<br>
        ⚠️ 3,510 fraud cases (2.93%)<br>
        ⚖️ 33:1 class imbalance<br><br>
        <p style='color:white; font-size:12px; font-weight:600; margin-bottom:6px;'>PERFORMANCE</p>
        🎯 XGBoost Precision: 1.00<br>
        🔎 WoE Scorecard Recall: 0.69<br>
        📈 Both models AUC: ~0.83<br><br>
        <p style='color:white; font-size:12px; font-weight:600; margin-bottom:6px;'>CONTEXT</p>
        💸 KSh 810M lost in 2024<br>
        📊 344% increase YoY (CBK)
    </div>
    """, unsafe_allow_html=True)


# PAGE: HOME
if page == "🏠 Home":

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {SAF_DARK_GREEN} 0%, {SAF_GREEN} 100%);
                padding: 36px 32px; border-radius: 16px; margin-bottom: 28px;
                box-shadow: 0 8px 24px rgba(0,102,51,0.25);'>
        <div style='display:flex; align-items:center; gap:16px;'>
            <div style='font-size:48px;'>🔐</div>
            <div>
                <h1 style='color:white; margin:0; font-size:28px; font-weight:800;'>
                    M-Pesa SIM Swap Fraud Detection
                </h1>
                <p style='color:#C8EDD0; margin:8px 0 0 0; font-size:15px; line-height:1.5;'>
                    Real-time fraud detection powered by XGBoost and WoE Scorecard models.
                    Built to protect M-Pesa users from SIM swap attacks.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions Analysed", "120,000")
    c2.metric("Fraud Cases",           "3,510",  delta="2.93% of total", delta_color="inverse")
    c3.metric("XGBoost Precision",     "1.00",   delta="Zero false positives")
    c4.metric("WoE Recall",            "0.69",   delta="Highest fraud catch rate")
    c5.metric("All Models AUC",        "~0.83",  delta="Strong separation")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">What does this tool do?</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="stat-card">
            <div style='font-size:32px; margin-bottom:12px;'>🔍</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>Check a Transaction</b>
            <p style='color:{SAF_MID_TEXT}; font-size:13px; margin-top:8px; line-height:1.6;'>
                Enter transaction details and get an instant fraud probability score
                from the XGBoost model. Every flag is a genuine fraud case
                with zero false positives.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="stat-card">
            <div style='font-size:32px; margin-bottom:12px;'>📋</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>Scorecard Audit</b>
            <p style='color:{SAF_MID_TEXT}; font-size:13px; margin-top:8px; line-height:1.6;'>
                Run the WoE Logistic Regression Scorecard for a fully explainable
                fraud decision. Every flag can be justified feature by feature
                to a CBK regulator or auditor.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="stat-card">
            <div style='font-size:32px; margin-bottom:12px;'>📊</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>Model Results</b>
            <p style='color:{SAF_MID_TEXT}; font-size:13px; margin-top:8px; line-height:1.6;'>
                Compare all four models side by side. See how XGBoost, LightGBM,
                Logistic Regression, and the WoE Scorecard perform against
                each other on the same dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">How the Two-Layer System Works</div>', unsafe_allow_html=True)

    lay1, lay2 = st.columns(2)
    with lay1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FFF0F0, #FFE5E5);
                    border: 1.5px solid {SAF_RED}; border-radius: 12px;
                    padding: 20px; height:100%;'>
            <div style='font-size:28px;'>⚡</div>
            <b style='color:{SAF_RED}; font-size:16px;'>Layer 1 — Real-Time Blocking</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                <b>Model:</b> XGBoost<br>
                <b>Precision:</b> 1.00 — zero false positives<br>
                <b>Use case:</b> Automated transaction blocking<br><br>
                When XGBoost flags a transaction, block it immediately.
                Every single flag is a confirmed fraud case.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with lay2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {SAF_LIGHT}, #E8F5E9);
                    border: 1.5px solid {SAF_GREEN}; border-radius: 12px;
                    padding: 20px; height:100%;'>
            <div style='font-size:28px;'>📋</div>
            <b style='color:{SAF_DARK_GREEN}; font-size:16px;'>Layer 2 — Regulatory Audit</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                <b>Model:</b> WoE Logistic Regression Scorecard<br>
                <b>Recall:</b> 0.69 — catches the most fraud<br>
                <b>Use case:</b> CBK compliance and audit trails<br><br>
                When a regulator asks why a transaction was blocked,
                the scorecard explains every contributing feature.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>Why this matters:</b> The CBK's 2024 Financial Sector Stability Report documented
        KSh 810 million in mobile banking losses, a 344% increase in a single year, driven
        primarily by SIM swap fraud. Existing defenses are reactive. This system detects
        fraud at the moment of the transaction before the money leaves the account.
    </div>
    """, unsafe_allow_html=True)


# PAGE: CHECK A TRANSACTION
elif page == "🔍 Check a Transaction":

    st.markdown(f"""
    <h2 style='color:{SAF_DARK_GREEN}; margin-bottom:4px;'>🔍 Check a Transaction</h2>
    <p style='color:{SAF_MID_TEXT}; margin-bottom:20px;'>
        Enter the transaction details below and click <b>Check Transaction</b>
        to get an instant fraud probability score from the XGBoost model.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b>XGBoost Layer:</b> This model has perfect Precision (1.00).
        Every transaction it flags as fraud is a genuine fraud case.
        Zero innocent customers have been wrongly blocked in testing.
        Use the <b>Quick Test</b> buttons in the sidebar to try a pre-built example.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown(f'<div class="section-header">💳 Transaction Details</div>',
                    unsafe_allow_html=True)

        amount = st.number_input(
            "Transaction Amount (KES)",
            min_value=1.0, max_value=500000.0,
            value=st.session_state.amount, step=100.0,
            key="input_amount"
        )
        sender_before = st.number_input(
            "Sender Balance Before (KES)",
            min_value=0.0, max_value=1000000.0,
            value=st.session_state.sender_before, step=500.0,
            key="input_sb"
        )
        sender_after = st.number_input(
            "Sender Balance After (KES)",
            min_value=0.0, max_value=1000000.0,
            value=st.session_state.sender_after, step=500.0,
            key="input_sa"
        )
        receiver_before = st.number_input(
            "Receiver Balance Before (KES)",
            min_value=0.0, max_value=1000000.0,
            value=st.session_state.receiver_before, step=500.0,
            key="input_rb"
        )
        receiver_after = st.number_input(
            "Receiver Balance After (KES)",
            min_value=0.0, max_value=1000000.0,
            value=st.session_state.receiver_after, step=500.0,
            key="input_ra"
        )
        hour = st.slider(
            "Hour of Transaction (0 = Midnight, 23 = 11pm)",
            0, 23,
            value=st.session_state.hour,
            key="input_hour"
        )

        st.markdown(f'<div class="section-header">📱 Transaction Context</div>',
                    unsafe_allow_html=True)

        txn_list   = ["peer", "till", "paybill"]
        dev_list   = ["smartphone", "feature"]
        reg_list   = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]
        day_list   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        txn_type = st.selectbox(
            "Transaction Type",
            txn_list,
            index=txn_list.index(st.session_state.txn_type),
            key="input_txn",
            help="peer = person to person | till = business till | paybill = bill payment"
        )
        device = st.selectbox(
            "Device Type",
            dev_list,
            index=dev_list.index(st.session_state.device),
            key="input_device",
            help="smartphone = Android or iOS | feature = basic USSD phone"
        )
        region = st.selectbox(
            "Region",
            reg_list,
            index=reg_list.index(st.session_state.region),
            key="input_region"
        )
        day = st.selectbox(
            "Day of Week",
            day_list,
            index=day_list.index(st.session_state.day),
            key="input_day"
        )

        check_btn = st.button("🔍 Check Transaction", type="primary",
                              use_container_width=True)

    with col_result:
        st.markdown(f'<div class="section-header">🎯 Detection Result</div>',
                    unsafe_allow_html=True)

        if check_btn:
            X_input, eng = engineer_features(
                amount, sender_before, sender_after,
                receiver_before, receiver_after,
                hour, txn_type, device, region, day
            )

            scale_cols = ['amount', 'sender_balance_before', 'sender_balance_after',
                          'receiver_balance_before', 'receiver_balance_after',
                          'balance_depletion_rate', 'sender_balance_ratio']
            X_scaled = X_input.copy()
            X_scaled[scale_cols] = scaler.transform(X_input[scale_cols])

            prob       = xgb_model.predict_proba(X_scaled)[0][1]
            prediction = xgb_model.predict(X_scaled)[0]
            pct        = round(prob * 100, 1)

            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <div style='font-size:48px;'>⚠️</div>
                    <h2 style='color:{SAF_RED}; margin:8px 0;'>FRAUDULENT</h2>
                    <h3 style='color:{SAF_RED}; margin:4px 0; font-size:22px;'>
                        Fraud Probability: {pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        This transaction matches SIM swap fraud patterns.<br>
                        <b>Recommended action: Block and flag for investigation.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <div style='font-size:48px;'>✅</div>
                    <h2 style='color:{SAF_DARK_GREEN}; margin:8px 0;'>LEGITIMATE</h2>
                    <h3 style='color:{SAF_DARK_GREEN}; margin:4px 0; font-size:22px;'>
                        Fraud Probability: {pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        No SIM swap fraud pattern detected.<br>
                        <b>Recommended action: Allow transaction to proceed.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 1.5))
            bar_color = SAF_RED if prediction == 1 else SAF_GREEN
            ax.barh(['Risk Level'], [pct],       color=bar_color, alpha=0.85, height=0.5)
            ax.barh(['Risk Level'], [100 - pct], left=[pct], color='#E0E0E0',
                    alpha=0.5, height=0.5)
            ax.axvline(x=50, color='orange', linestyle='--', linewidth=1.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Fraud Probability (%)')
            label_x = min(pct / 2, 45) if pct > 5 else 5
            ax.text(label_x, 0, f'{pct}%',
                    ha='center', va='center', color='white',
                    fontweight='bold', fontsize=12)
            ax.set_title('Fraud Probability Score', fontsize=11,
                         color=SAF_DARK_GREEN, fontweight='bold')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">🧬 SIM Swap Behavioral Signals</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <p style='color:#555; font-size:13px; margin-bottom:12px;'>
                These four engineered features capture the account drainage
                pattern of a SIM swap attack.
            </p>
            """, unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            dr = eng['balance_depletion_rate']
            br = eng['sender_balance_ratio']
            s1.metric("Balance Depletion Rate",
                      f"{dr:.4f}",
                      delta="⚠️ High — account drain" if dr > 1 else "✅ Normal")
            s1.metric("Sender Balance Ratio",
                      f"{br:.4f}",
                      delta="⚠️ Low — account drained" if br < 0.1 else "✅ Normal")
            s2.metric("High Value Transaction",
                      "YES ⚠️" if eng['is_high_value'] else "NO ✅",
                      delta="Risk signal" if eng['is_high_value'] else "Normal")
            s2.metric("Balance Wipeout",
                      "YES ⚠️" if eng['is_balance_wipeout'] else "NO ✅",
                      delta="Critical risk" if eng['is_balance_wipeout'] else "Normal")
        else:
            st.markdown(f"""
            <div style='text-align:center; padding:60px 20px;
                        background:{SAF_LIGHT}; border-radius:12px;
                        border: 2px dashed {SAF_GREEN};'>
                <div style='font-size:48px; margin-bottom:16px;'>🔍</div>
                <h3 style='color:{SAF_DARK_GREEN};'>Ready to check a transaction</h3>
                <p style='color:{SAF_MID_TEXT}; font-size:14px;'>
                    Fill in the transaction details on the left<br>
                    and click <b>Check Transaction</b>.<br><br>
                    Or use the <b>Quick Test</b> buttons in the sidebar<br>
                    to load a sample fraud or legitimate transaction.
                </p>
            </div>
            """, unsafe_allow_html=True)


# PAGE: SCORECARD AUDIT
elif page == "📋 Scorecard Audit":

    st.markdown(f"""
    <h2 style='color:{SAF_DARK_GREEN}; margin-bottom:4px;'>📋 Scorecard Audit</h2>
    <p style='color:{SAF_MID_TEXT}; margin-bottom:20px;'>
        The WoE Logistic Regression Scorecard explains every fraud decision
        feature by feature. Use this for regulatory reporting and CBK audit trails.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b>WoE Scorecard Layer:</b> Unlike XGBoost, every decision this model makes
        can be explained to a regulator. Higher Recall (0.69) means it catches more
        fraud than XGBoost, at the cost of more false alarms (Precision: 0.23).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_woe_form, col_woe_result = st.columns([1, 1], gap="large")

    with col_woe_form:
        st.markdown(f'<div class="section-header">💳 Transaction Inputs</div>',
                    unsafe_allow_html=True)

        woe_amount = st.number_input(
            "Amount (KES)",
            min_value=1.0, max_value=500000.0,
            value=st.session_state.woe_amount, step=100.0,
            key="woe_input_amount"
        )
        woe_s_before = st.number_input(
            "Sender Balance Before (KES)",
            min_value=0.0, max_value=1000000.0,
            value=st.session_state.woe_s_before, step=500.0,
            key="woe_input_sb"
        )
        woe_s_after = st.number_input(
            "Sender Balance After (KES)",
            min_value=0.0, max_value=1000000.0,
            value=st.session_state.woe_s_after, step=500.0,
            key="woe_input_sa"
        )

        woe_btn = st.button("📋 Run Scorecard Audit", type="primary",
                            use_container_width=True)

        st.markdown(f"""
        <div class="warning-box" style='margin-top:16px;'>
            <b>How WoE scoring works:</b><br>
            Each feature value is mapped to a WoE score based on how common
            that value is in fraud vs legitimate transactions.<br><br>
            <b>Positive WoE</b> = this value appears more in fraud<br>
            <b>Negative WoE</b> = this value appears more in legitimate
        </div>
        """, unsafe_allow_html=True)

    with col_woe_result:
        st.markdown(f'<div class="section-header">🎯 Scorecard Result</div>',
                    unsafe_allow_html=True)

        if woe_btn:
            woe_is_high = 1 if woe_amount > 3800 else 0
            woe_ratio   = woe_s_after / (woe_s_before + 1)

            woe_input = prepare_woe_input(
                woe_amount, woe_s_after, woe_ratio, woe_is_high
            )

            woe_prob = woe_model.predict_proba(woe_input)[0][1]
            woe_pred = woe_model.predict(woe_input)[0]
            woe_pct  = round(woe_prob * 100, 1)

            if woe_pred == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <div style='font-size:48px;'>⚠️</div>
                    <h2 style='color:{SAF_RED}; margin:8px 0;'>FRAUD FLAG</h2>
                    <h3 style='color:{SAF_RED}; margin:4px 0; font-size:20px;'>
                        Scorecard Probability: {woe_pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        Scorecard flags this transaction for regulatory review.<br>
                        <b>Recommended: Queue for credit officer audit.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <div style='font-size:48px;'>✅</div>
                    <h2 style='color:{SAF_DARK_GREEN}; margin:8px 0;'>CLEAR</h2>
                    <h3 style='color:{SAF_DARK_GREEN}; margin:4px 0; font-size:20px;'>
                        Scorecard Probability: {woe_pct}%
                    </h3>
                    <p style='color:#555; margin:12px 0 0 0; font-size:13px;'>
                        Scorecard does not flag this transaction.<br>
                        <b>No regulatory action required.</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">📊 Feature Score Breakdown</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <p style='color:#555; font-size:13px; margin-bottom:12px;'>
                Each row is one feature. Positive WoE pushed toward fraud.
                Negative WoE pushed toward legitimate.
            </p>
            """, unsafe_allow_html=True)

            woe_display = woe_input.T.reset_index()
            woe_display.columns = ['Feature', 'WoE Score']
            woe_display['WoE Score'] = woe_display['WoE Score'].round(4)
            woe_display['Signal'] = woe_display['WoE Score'].apply(
                lambda x: '🔴 Fraud signal' if x > 0 else '🟢 Legitimate signal'
            )
            woe_display['Feature'] = (
                woe_display['Feature']
                .str.replace('_woe', '', regex=False)
                .str.replace('_', ' ', regex=False)
                .str.title()
            )
            st.dataframe(woe_display, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">⚖️ Model Coefficients</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <p style='color:#555; font-size:13px; margin-bottom:12px;'>
                A longer red bar means that feature pushes more strongly toward fraud.
            </p>
            """, unsafe_allow_html=True)

            coef_df = pd.DataFrame({
                'Feature':     ['Sender Balance After', 'Sender Balance Ratio',
                                 'Amount', 'Is High Value'],
                'Coefficient': [0.5879, 0.4465, 0.1336, -0.0050]
            })
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            colors = [SAF_RED if c > 0 else SAF_GREEN for c in coef_df['Coefficient']]
            ax2.barh(coef_df['Feature'], coef_df['Coefficient'],
                     color=colors, alpha=0.85, edgecolor='white', height=0.5)
            ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
            ax2.set_xlabel('Coefficient Value')
            ax2.set_title('Red = pushes toward fraud | Green = toward legitimate',
                          fontsize=10, color=SAF_MID_TEXT)
            fig2.patch.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        else:
            st.markdown(f"""
            <div style='text-align:center; padding:60px 20px;
                        background:{SAF_LIGHT}; border-radius:12px;
                        border: 2px dashed {SAF_GREEN};'>
                <div style='font-size:48px; margin-bottom:16px;'>📋</div>
                <h3 style='color:{SAF_DARK_GREEN};'>Ready for scorecard audit</h3>
                <p style='color:{SAF_MID_TEXT}; font-size:14px;'>
                    Enter the transaction amount and balance details<br>
                    on the left and click <b>Run Scorecard Audit</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)


# PAGE: MODEL RESULTS
elif page == "📊 Model Results":

    st.markdown(f"""
    <h2 style='color:{SAF_DARK_GREEN}; margin-bottom:4px;'>📊 Model Performance Results</h2>
    <p style='color:{SAF_MID_TEXT}; margin-bottom:20px;'>
        All four models were trained and tested on the same 120,000-record dataset
        using an 80/20 stratified split. Here is how they compare.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        All four models achieve nearly identical AUC-ROC scores (~0.83).
        The difference is entirely in <b>how</b> they flag fraud, not whether they can detect it.
        No single model wins on every metric. That is why both XGBoost and the WoE Scorecard
        are deployed together.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">📋 Full Comparison Scorecard</div>',
                unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        'Model':           ['Logistic Regression', 'WoE Scorecard', 'LightGBM', 'XGBoost'],
        'Type':            ['Baseline', 'Regulatory', 'High Performance', 'Primary'],
        'Precision':       [0.5219, 0.2280, 0.9603, 1.0000],
        'Recall':          [0.6624, 0.6852, 0.6538, 0.6553],
        'F1-Score':        [0.5838, 0.3421, 0.7780, 0.7917],
        'AUC-ROC':         [0.8343, 0.8250, 0.8267, 0.8303],
        'False Positives': [426,    1629,   19,     0],
        'Fraud Caught':    [465,    481,    459,    460],
    })

    def highlight_best(s):
        if s.name in ['Precision', 'F1-Score', 'AUC-ROC', 'Fraud Caught', 'Recall']:
            return ['background-color:#E8F5E9; font-weight:bold;'
                    if v == s.max() else '' for v in s]
        if s.name == 'False Positives':
            return ['background-color:#E8F5E9; font-weight:bold;'
                    if v == s.min() else '' for v in s]
        return ['' for _ in s]

    st.dataframe(
        comp_df.style.apply(highlight_best,
                            subset=['Precision', 'Recall', 'F1-Score', 'AUC-ROC',
                                    'False Positives', 'Fraud Caught']),
        use_container_width=True, hide_index=True
    )
    st.markdown("<p style='color:#888; font-size:12px;'>Green = best value in that column</p>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_ch1, col_ch2 = st.columns(2)
    models_s = ['LR Baseline', 'WoE Scorecard', 'LightGBM', 'XGBoost']
    x        = np.arange(len(models_s))
    w        = 0.35
    clrs     = ['#888888', SAF_DARK_GREEN, '#FFA000', SAF_RED]

    with col_ch1:
        st.markdown(f'<div class="section-header">Precision vs Recall</div>',
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(x - w/2, comp_df['Precision'], w, color=clrs, alpha=0.9,  edgecolor='white')
        ax3.bar(x + w/2, comp_df['Recall'],    w, color=clrs, alpha=0.45, edgecolor='white')
        ax3.axhline(y=0.80, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
        ax3.set_xticks(x)
        ax3.set_xticklabels(models_s, rotation=10, fontsize=9)
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1.15)
        ax3.legend(['Precision (solid)', 'Recall (transparent)'], fontsize=9)
        ax3.set_title('Precision vs Recall by Model', fontsize=11,
                      color=SAF_DARK_GREEN, fontweight='bold')
        fig3.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_ch2:
        st.markdown(f'<div class="section-header">False Positives vs Fraud Caught</div>',
                    unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.bar(x - w/2, comp_df['False Positives'], w,
                label='False Positives (innocent customers flagged)',
                color=SAF_RED, alpha=0.75, edgecolor='white')
        ax4.bar(x + w/2, comp_df['Fraud Caught'], w,
                label='Fraud Caught',
                color=SAF_GREEN, alpha=0.85, edgecolor='white')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models_s, rotation=10, fontsize=9)
        ax4.set_ylabel('Number of Transactions')
        ax4.legend(fontsize=9)
        ax4.set_title('False Positives vs Fraud Caught\n(out of 24,000 test transactions)',
                      fontsize=10, color=SAF_DARK_GREEN, fontweight='bold')
        fig4.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">🚀 Deployment Recommendation</div>',
                unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FFF0F0, #FFE5E5);
                    border:1.5px solid {SAF_RED}; border-radius:12px; padding:20px;'>
            <b style='color:{SAF_RED}; font-size:15px;'>⚡ Layer 1 — XGBoost</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                Precision 1.00 | F1 0.79 | False Positives: <b>0</b><br><br>
                Block transactions automatically. Zero innocent customers
                will be wrongly stopped. Every flag is genuine fraud.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {SAF_LIGHT}, #E8F5E9);
                    border:1.5px solid {SAF_GREEN}; border-radius:12px; padding:20px;'>
            <b style='color:{SAF_DARK_GREEN}; font-size:15px;'>📋 Layer 2 — WoE Scorecard</b>
            <p style='color:{SAF_DARK_TEXT}; font-size:13px; margin-top:10px; line-height:1.7;'>
                Recall 0.69 | Fraud Caught: <b>481</b> | CBK compliant<br><br>
                Explain every blocked transaction to a regulator with
                a full feature-by-feature audit trail.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="warning-box">
        <b>PSI Monitoring:</b> All four WoE features confirmed STABLE at deployment
        (PSI below 0.001). In production, recompute PSI monthly on incoming transactions.
        Alert at PSI above 0.10. Retrain the model immediately if PSI exceeds 0.25.
    </div>
    """, unsafe_allow_html=True)
