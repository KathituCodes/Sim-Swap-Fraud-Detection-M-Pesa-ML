import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="M-Pesa SIM Swap Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────
NAVY   = "#1F3864"
TEAL   = "#2E75B6"
RED    = "#C00000"
GREEN  = "#375623"
ORANGE = "#E97132"
PURPLE = "#7B3F8C"

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }

    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 4px solid #2E75B6;
        border-radius: 6px;
        padding: 12px 16px;
    }

    .fraud-alert {
        background-color: #FFE5E5;
        border: 2px solid #C00000;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }

    .legit-alert {
        background-color: #E8F5E9;
        border: 2px solid #375623;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }

    .insight-box {
        background-color: #EBF3FB;
        border-left: 4px solid #2E75B6;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 14px;
        color: #1F3864;
    }

    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #E97132;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 14px;
        color: #7B3F00;
    }

    .section-header {
        background: linear-gradient(90deg, #1F3864, #2E75B6);
        color: white;
        padding: 10px 18px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 16px;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """
    Load all four model artifacts.
    @st.cache_resource means models are loaded once
    and reused across all user sessions.
    """
    xgb_model      = joblib.load("xgboost_fraud_model.pkl")
    scaler         = joblib.load("scaler.pkl")
    woe_model      = joblib.load("woe_scorecard_model.pkl")

    with open("woe_bins.pkl", "rb") as f:
        woe_bins   = pickle.load(f)

    return xgb_model, scaler, woe_model, woe_bins


xgb_model, scaler, woe_model, woe_bins = load_models()


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING FUNCTION
# Replicates the exact same features built in the notebook
# ─────────────────────────────────────────────────────────────
def engineer_features(amount, sender_bal_before, sender_bal_after,
                       receiver_bal_before, receiver_bal_after,
                       hour, transaction_type, device_type,
                       region, day_of_week):
    """
    Engineer all features from raw transaction inputs.
    Must match exactly what was done in notebook Phase 1.
    """
    # Engineered features
    balance_depletion_rate = amount / (sender_bal_before + 1)
    is_high_value          = 1 if amount > 3800 else 0
    is_balance_wipeout     = 1 if sender_bal_after < 100 else 0
    sender_balance_ratio   = sender_bal_after / (sender_bal_before + 1)

    # One-hot encoding — must match get_dummies output from notebook
    # transaction_type: drop_first=True drops 'paybill', keeps 'peer' and 'till'
    txn_peer = 1 if transaction_type == "peer" else 0
    txn_till = 1 if transaction_type == "till" else 0

    # device_type: drop_first=True drops 'feature', keeps 'smartphone'
    dev_smartphone = 1 if device_type == "smartphone" else 0

    # region: drop_first=True drops 'Eldoret', keeps others
    reg_kisumu  = 1 if region == "Kisumu"  else 0
    reg_mombasa = 1 if region == "Mombasa" else 0
    reg_nairobi = 1 if region == "Nairobi" else 0
    reg_nakuru  = 1 if region == "Nakuru"  else 0

    # day_of_week: drop_first=True drops 'Fri', keeps others
    day_mon = 1 if day_of_week == "Mon" else 0
    day_sat = 1 if day_of_week == "Sat" else 0
    day_sun = 1 if day_of_week == "Sun" else 0
    day_thu = 1 if day_of_week == "Thu" else 0
    day_tue = 1 if day_of_week == "Tue" else 0
    day_wed = 1 if day_of_week == "Wed" else 0

    # Build feature dict in the exact column order from training
    features = {
        'amount':                    amount,
        'sender_balance_before':     sender_bal_before,
        'sender_balance_after':      sender_bal_after,
        'receiver_balance_before':   receiver_bal_before,
        'receiver_balance_after':    receiver_bal_after,
        'hour':                      hour,
        'balance_depletion_rate':    balance_depletion_rate,
        'is_high_value':             is_high_value,
        'is_balance_wipeout':        is_balance_wipeout,
        'sender_balance_ratio':      sender_balance_ratio,
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
    }

    return pd.DataFrame([features]), {
        'balance_depletion_rate': balance_depletion_rate,
        'is_high_value':          is_high_value,
        'is_balance_wipeout':     is_balance_wipeout,
        'sender_balance_ratio':   sender_balance_ratio
    }


def prepare_woe_input(amount, sender_bal_before, sender_bal_after,
                      sender_balance_ratio, is_high_value):
    """
    Apply WoE transformation manually using saved bin boundaries.
    Replicates sc.woebin_ply() without requiring scorecardpy at runtime.
    """
    def get_woe(value, bin_df):
        """Look up the WoE value for a given raw input using bin boundaries."""
        for _, row in bin_df.iterrows():
            bin_range = str(row['bin'])
            # Handle missing bin
            if bin_range == 'missing':
                continue
            # Parse bin boundaries from scorecardpy format e.g. [-inf, 500.0)
            bin_range = bin_range.replace('[-inf', '(-inf')
            try:
                lower = float(bin_range.split(',')[0].strip('([)]-inf ').replace('-inf', '-999999999'))
                upper = float(bin_range.split(',')[1].strip('([)] inf').replace('inf', '999999999'))
                low_inc  = '[' in bin_range.split(',')[0]
                high_inc = ']' in bin_range.split(',')[1]

                if low_inc:
                    lower_check = value >= lower
                else:
                    lower_check = value > lower

                if high_inc:
                    upper_check = value <= upper
                else:
                    upper_check = value < upper

                if lower_check and upper_check:
                    return float(row['woe'])
            except Exception:
                continue
        # Return 0 if no bin matched
        return 0.0

    result = {}
    feature_map = {
        'amount':               amount,
        'sender_balance_after': sender_bal_after,
        'sender_balance_ratio': sender_balance_ratio,
        'is_high_value':        is_high_value,
    }

    for feature, value in feature_map.items():
        if feature in woe_bins:
            woe_val = get_woe(value, woe_bins[feature])
            result[f'{feature}_woe'] = woe_val

    return pd.DataFrame([result])


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='color:white; font-size:20px; font-weight:700;
                padding-bottom:6px; border-bottom:1px solid #2E75B6;
                margin-bottom:16px;'>
        🔐 SIM Swap Detector
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#90CAF9; font-size:12px;'>Urbanus Kathitu | Kat.Codes</p>",
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border-color:#2E75B6;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#90CAF9; font-size:11px; line-height:1.7;'>
        <b style='color:white;'>Dataset</b><br>
        120,000 M-Pesa transactions<br>
        3,510 fraud cases (2.93%)<br>
        33.2:1 class imbalance<br><br>
        <b style='color:white;'>Models</b><br>
        XGBoost (Precision: 1.00)<br>
        WoE Scorecard (Recall: 0.69)<br><br>
        <b style='color:white;'>Context</b><br>
        KSh 810M lost to SIM swap<br>
        fraud in Kenya in 2024<br>
        344% increase YoY (CBK)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2E75B6;'>", unsafe_allow_html=True)

    # Quick test buttons
    st.markdown(
        "<p style='color:#BBDEFB; font-size:13px; font-weight:600;'>Quick Test Cases</p>",
        unsafe_allow_html=True
    )

    fraud_test    = st.button("Load Fraud Test Case")
    legit_test    = st.button("Load Legitimate Test Case")


# ─────────────────────────────────────────────────────────────
# DEFAULT AND TEST CASE VALUES
# ─────────────────────────────────────────────────────────────
# Fraud test case: large amount draining a small balance
if fraud_test:
    default_amount          = 4900.0
    default_sender_before   = 5000.0
    default_sender_after    = 50.0
    default_receiver_before = 1000.0
    default_receiver_after  = 5900.0
    default_hour            = 2
    default_txn_type        = "peer"
    default_device          = "smartphone"
    default_region          = "Nairobi"
    default_day             = "Tue"
# Legitimate test case: small routine transaction
elif legit_test:
    default_amount          = 500.0
    default_sender_before   = 25000.0
    default_sender_after    = 24500.0
    default_receiver_before = 3000.0
    default_receiver_after  = 3500.0
    default_hour            = 14
    default_txn_type        = "paybill"
    default_device          = "smartphone"
    default_region          = "Nairobi"
    default_day             = "Wed"
else:
    default_amount          = 500.0
    default_sender_before   = 5000.0
    default_sender_after    = 4500.0
    default_receiver_before = 1000.0
    default_receiver_after  = 1500.0
    default_hour            = 14
    default_txn_type        = "peer"
    default_device          = "smartphone"
    default_region          = "Nairobi"
    default_day             = "Tue"


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1F3864,#2E75B6);
            padding:28px 24px; border-radius:10px; margin-bottom:24px;'>
    <h1 style='color:white; margin:0; font-size:26px;'>
        🔐 M-Pesa SIM Swap Fraud Detection System
    </h1>
    <p style='color:#BBDEFB; margin:8px 0 0 0; font-size:14px;'>
        Two-layer detection: XGBoost real-time blocking + WoE Scorecard regulatory reporting.
        Built on 120,000 synthetic M-Pesa transactions with a 33:1 class imbalance.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL PERFORMANCE KPIs
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("XGBoost Precision", "1.00",  delta="Zero false positives")
c2.metric("XGBoost Recall",    "0.66",  delta="66% fraud caught")
c3.metric("XGBoost F1",        "0.79")
c4.metric("WoE Recall",        "0.69",  delta="Catches more fraud")
c5.metric("WoE Precision",     "0.23",  delta="Higher false alarms")
c6.metric("All Models AUC",    "~0.83", delta="Consistent separation")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Real-Time Fraud Check (XGBoost)",
    "📋 Regulatory Scorecard (WoE)",
    "📊 Model Comparison"
])


# ═══════════════════════════════════════════════════════════
# TAB 1: XGBOOST REAL-TIME FRAUD CHECK
# ═══════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <div class="insight-box">
        <b>XGBoost Layer:</b> Real-time fraud blocking with perfect precision (1.00).
        Every transaction flagged by this model is a genuine fraud case.
        Zero false positives confirmed across 24,000 test transactions.
        Use this layer for automated transaction blocking decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown(
            '<div class="section-header">Transaction Details</div>',
            unsafe_allow_html=True
        )

        amount = st.number_input(
            "Transaction Amount (KES)",
            min_value=1.0, max_value=500000.0,
            value=default_amount, step=100.0
        )
        sender_bal_before = st.number_input(
            "Sender Balance Before (KES)",
            min_value=0.0, max_value=1000000.0,
            value=default_sender_before, step=500.0
        )
        sender_bal_after = st.number_input(
            "Sender Balance After (KES)",
            min_value=0.0, max_value=1000000.0,
            value=default_sender_after, step=500.0
        )
        receiver_bal_before = st.number_input(
            "Receiver Balance Before (KES)",
            min_value=0.0, max_value=1000000.0,
            value=default_receiver_before, step=500.0
        )
        receiver_bal_after = st.number_input(
            "Receiver Balance After (KES)",
            min_value=0.0, max_value=1000000.0,
            value=default_receiver_after, step=500.0
        )
        hour = st.slider("Hour of Transaction (0 = Midnight)", 0, 23,
                         value=default_hour)

        st.markdown(
            '<div class="section-header">Transaction Context</div>',
            unsafe_allow_html=True
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["peer", "till", "paybill"],
            index=["peer", "till", "paybill"].index(default_txn_type)
        )
        device_type = st.selectbox(
            "Device Type",
            ["smartphone", "feature"],
            index=["smartphone", "feature"].index(default_device)
        )
        region = st.selectbox(
            "Region",
            ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"],
            index=["Nairobi", "Mombasa", "Kisumu",
                   "Nakuru", "Eldoret"].index(default_region)
        )
        day_of_week = st.selectbox(
            "Day of Week",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            index=["Mon", "Tue", "Wed", "Thu",
                   "Fri", "Sat", "Sun"].index(default_day)
        )

        check_button = st.button("🔍 Check Transaction", type="primary",
                                 use_container_width=True)

    with col_result:
        st.markdown(
            '<div class="section-header">Detection Result</div>',
            unsafe_allow_html=True
        )

        if check_button:
            # Engineer features
            X_input, engineered = engineer_features(
                amount, sender_bal_before, sender_bal_after,
                receiver_bal_before, receiver_bal_after,
                hour, transaction_type, device_type,
                region, day_of_week
            )

            # Scale numerical features
            scale_cols = ['amount', 'sender_balance_before',
                          'sender_balance_after', 'receiver_balance_before',
                          'receiver_balance_after', 'balance_depletion_rate',
                          'sender_balance_ratio']
            X_scaled = X_input.copy()
            X_scaled[scale_cols] = scaler.transform(X_input[scale_cols])

            # Predict
            fraud_prob  = xgb_model.predict_proba(X_scaled)[0][1]
            prediction  = xgb_model.predict(X_scaled)[0]
            fraud_pct   = round(fraud_prob * 100, 1)

            # Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h2 style='color:#C00000; margin:0;'>⚠️ FRAUDULENT</h2>
                    <h3 style='color:#C00000; margin:8px 0;'>
                        Fraud Probability: {fraud_pct}%
                    </h3>
                    <p style='color:#444; margin:0; font-size:13px;'>
                        This transaction matches SIM swap fraud patterns.<br>
                        Recommended action: Block and flag for investigation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <h2 style='color:#375623; margin:0;'>✅ LEGITIMATE</h2>
                    <h3 style='color:#375623; margin:8px 0;'>
                        Fraud Probability: {fraud_pct}%
                    </h3>
                    <p style='color:#444; margin:0; font-size:13px;'>
                        No SIM swap fraud pattern detected.<br>
                        Recommended action: Allow transaction to proceed.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            st.markdown(
                '<div class="section-header">Fraud Probability Gauge</div>',
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(8, 2))
            bar_color = RED if prediction == 1 else GREEN
            ax.barh(['Fraud Risk'], [fraud_pct],
                    color=bar_color, alpha=0.8, height=0.4)
            ax.barh(['Fraud Risk'], [100 - fraud_pct],
                    left=[fraud_pct], color='#E0E0E0',
                    alpha=0.5, height=0.4)
            ax.axvline(x=50, color='orange', linestyle='--',
                       linewidth=1.5, label='50% threshold')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Fraud Probability (%)')
            ax.text(fraud_pct / 2, 0, f'{fraud_pct}%',
                    ha='center', va='center',
                    color='white', fontweight='bold', fontsize=12)
            ax.legend(loc='lower right', fontsize=8)
            ax.set_title('Fraud Probability Score', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Engineered signals
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">SIM Swap Behavioral Signals</div>',
                unsafe_allow_html=True
            )

            sig_col1, sig_col2 = st.columns(2)
            sig_col1.metric(
                "Balance Depletion Rate",
                f"{engineered['balance_depletion_rate']:.4f}",
                delta="High = account drain" if engineered['balance_depletion_rate'] > 1 else "Normal"
            )
            sig_col1.metric(
                "Sender Balance Ratio",
                f"{engineered['sender_balance_ratio']:.4f}",
                delta="Low = account drained" if engineered['sender_balance_ratio'] < 0.1 else "Normal"
            )
            sig_col2.metric(
                "High Value Transaction",
                "YES" if engineered['is_high_value'] else "NO",
                delta="Risk signal" if engineered['is_high_value'] else "Normal"
            )
            sig_col2.metric(
                "Balance Wipeout",
                "YES" if engineered['is_balance_wipeout'] else "NO",
                delta="Critical risk" if engineered['is_balance_wipeout'] else "Normal"
            )

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#888;'>
                <h3>Enter transaction details and click</h3>
                <h3>Check Transaction to see the result.</h3>
                <br>
                <p>Or use the Quick Test Cases in the sidebar<br>
                to load a pre-built fraud or legitimate example.</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: WoE REGULATORY SCORECARD
# ═══════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="insight-box">
        <b>WoE Scorecard Layer:</b> Regulatory and audit-ready model.
        Unlike XGBoost, every decision made by this model can be explained
        feature by feature to a credit officer or CBK regulator.
        Higher recall (0.69) catches more fraud but at the cost of
        more false positives (Precision: 0.23).
        Use this layer for audit trails and regulatory reporting.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <b>How WoE scoring works:</b> Each feature is converted into a
        Weight of Evidence score based on pre-computed bin boundaries.
        Positive WoE means that value is associated with fraud.
        Negative WoE means it is associated with legitimate transactions.
        The logistic regression model combines these scores into a
        final fraud probability that is fully auditable.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_woe_input, col_woe_result = st.columns([1, 1])

    with col_woe_input:
        st.markdown(
            '<div class="section-header">Transaction Inputs (Scorecard)</div>',
            unsafe_allow_html=True
        )

        woe_amount           = st.number_input(
            "Amount (KES)",
            min_value=1.0, max_value=500000.0,
            value=default_amount, step=100.0,
            key="woe_amount"
        )
        woe_sender_before    = st.number_input(
            "Sender Balance Before (KES)",
            min_value=0.0, max_value=1000000.0,
            value=default_sender_before, step=500.0,
            key="woe_sender_before"
        )
        woe_sender_after     = st.number_input(
            "Sender Balance After (KES)",
            min_value=0.0, max_value=1000000.0,
            value=default_sender_after, step=500.0,
            key="woe_sender_after"
        )

        woe_button = st.button("📋 Run Scorecard", type="primary",
                               use_container_width=True)

    with col_woe_result:
        st.markdown(
            '<div class="section-header">Scorecard Result</div>',
            unsafe_allow_html=True
        )

        if woe_button:
            # Compute engineered features needed for WoE
            woe_is_high_value       = 1 if woe_amount > 3800 else 0
            woe_sender_bal_ratio    = woe_sender_after / (woe_sender_before + 1)
            woe_is_balance_wipeout  = 1 if woe_sender_after < 100 else 0

            # Prepare WoE input
            woe_input = prepare_woe_input(
                woe_amount,
                woe_sender_before,
                woe_sender_after,
                woe_sender_bal_ratio,
                woe_is_high_value
            )

            # Drop is_balance_wipeout if present
            if 'is_balance_wipeout' in woe_input.columns:
                woe_input = woe_input.drop(columns=['is_balance_wipeout'])

            # Predict
            woe_prob       = woe_model.predict_proba(woe_input)[0][1]
            woe_prediction = woe_model.predict(woe_input)[0]
            woe_pct        = round(woe_prob * 100, 1)

            if woe_prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h2 style='color:#C00000; margin:0;'>⚠️ FRAUD FLAG</h2>
                    <h3 style='color:#C00000; margin:8px 0;'>
                        Scorecard Probability: {woe_pct}%
                    </h3>
                    <p style='color:#444; margin:0; font-size:13px;'>
                        Scorecard flags this transaction for review.<br>
                        Recommended action: Queue for credit officer audit.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    <h2 style='color:#375623; margin:0;'>✅ CLEAR</h2>
                    <h3 style='color:#375623; margin:8px 0;'>
                        Scorecard Probability: {woe_pct}%
                    </h3>
                    <p style='color:#444; margin:0; font-size:13px;'>
                        Scorecard does not flag this transaction.<br>
                        Recommended action: No regulatory action required.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # WoE feature breakdown
            st.markdown(
                '<div class="section-header">WoE Feature Breakdown</div>',
                unsafe_allow_html=True
            )

            woe_display = woe_input.T.reset_index()
            woe_display.columns = ['Feature (WoE)', 'WoE Score']
            woe_display['WoE Score'] = woe_display['WoE Score'].round(4)
            woe_display['Signal'] = woe_display['WoE Score'].apply(
                lambda x: 'Fraud signal' if x > 0 else 'Legitimate signal'
            )

            st.dataframe(woe_display, use_container_width=True,
                         hide_index=True)

            st.markdown("""
            <div class="insight-box">
                <b>How to read WoE scores:</b>
                Positive WoE means this feature value is more common
                in fraud transactions than legitimate ones.
                Negative WoE means the opposite.
                The logistic regression model multiplies each WoE score
                by its coefficient to produce the final probability.
                This makes every decision fully explainable to a regulator.
            </div>
            """, unsafe_allow_html=True)

            # Coefficients chart
            st.markdown(
                '<div class="section-header">Model Coefficients</div>',
                unsafe_allow_html=True
            )
            coef_data = {
                'Feature':     ['sender_balance_after_woe',
                                 'sender_balance_ratio_woe',
                                 'amount_woe',
                                 'is_high_value_woe'],
                'Coefficient': [0.5879, 0.4465, 0.1336, -0.0050]
            }
            coef_df = pd.DataFrame(coef_data)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            colors = [RED if c > 0 else TEAL for c in coef_df['Coefficient']]
            ax2.barh(coef_df['Feature'], coef_df['Coefficient'],
                     color=colors, alpha=0.85, edgecolor='white')
            ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
            ax2.set_title('WoE Scorecard Coefficients\n'
                          'Red = pushes toward fraud | Blue = toward legitimate',
                          fontsize=11)
            ax2.set_xlabel('Coefficient Value')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#888;'>
                <h3>Enter transaction details and click</h3>
                <h3>Run Scorecard to see the regulatory result.</h3>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
with tab3:

    st.markdown(f"<h2 style='color:{NAVY};'>Four-Model Performance Comparison</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        All four models were trained and evaluated on the same 120,000-record
        dataset using an 80/20 stratified split. The AUC-ROC scores are nearly
        identical across all models (0.825 to 0.834), meaning all four have
        roughly the same overall ability to separate fraud from legitimate.
        The critical difference is in how they flag fraud, not what they detect.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model comparison table
    comparison_data = {
        'Model':     ['Logistic Regression (Baseline)',
                      'WoE Scorecard (Regulatory)',
                      'LightGBM',
                      'XGBoost (Primary)'],
        'Type':      ['Interpretable', 'Interpretable',
                      'High Performance', 'High Performance'],
        'Precision': [0.5219, 0.2280, 0.9603, 1.0000],
        'Recall':    [0.6624, 0.6852, 0.6538, 0.6553],
        'F1-Score':  [0.5838, 0.3421, 0.7780, 0.7917],
        'AUC-ROC':   [0.8343, 0.8250, 0.8267, 0.8303],
    }
    comp_df = pd.DataFrame(comparison_data)

    def highlight_best(s):
        styles = []
        for v in s:
            if s.name in ['Precision', 'F1-Score', 'AUC-ROC']:
                styles.append(
                    'background-color: #E8F5E9; font-weight: bold;'
                    if v == s.max() else ''
                )
            elif s.name == 'Recall':
                styles.append(
                    'background-color: #E8F5E9; font-weight: bold;'
                    if v == s.max() else ''
                )
            else:
                styles.append('')
        return styles

    styled_comp = comp_df.style.apply(
        highlight_best,
        subset=['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    )
    st.dataframe(styled_comp, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart comparison
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(
            '<div class="section-header">Precision vs Recall Tradeoff</div>',
            unsafe_allow_html=True
        )
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        models_short = ['LR Baseline', 'WoE Scorecard', 'LightGBM', 'XGBoost']
        x      = np.arange(len(models_short))
        width  = 0.35
        colors_prec = [TEAL, PURPLE, ORANGE, RED]

        bars1 = ax3.bar(x - width/2,
                        comp_df['Precision'], width,
                        label='Precision', color=colors_prec,
                        alpha=0.85, edgecolor='white')
        bars2 = ax3.bar(x + width/2,
                        comp_df['Recall'], width,
                        label='Recall', color=colors_prec,
                        alpha=0.45, edgecolor='white')

        ax3.set_title('Precision vs Recall by Model', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(models_short, rotation=15, fontsize=9)
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1.15)
        ax3.legend(['Precision (solid)', 'Recall (transparent)'])
        ax3.axhline(y=0.80, color='black', linestyle='--',
                    linewidth=0.8, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_chart2:
        st.markdown(
            '<div class="section-header">F1-Score and AUC-ROC</div>',
            unsafe_allow_html=True
        )
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        bars3 = ax4.bar(x - width/2,
                        comp_df['F1-Score'], width,
                        label='F1-Score', color=colors_prec,
                        alpha=0.85, edgecolor='white')
        bars4 = ax4.bar(x + width/2,
                        comp_df['AUC-ROC'], width,
                        label='AUC-ROC', color=colors_prec,
                        alpha=0.45, edgecolor='white')

        ax4.set_title('F1-Score vs AUC-ROC by Model', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(models_short, rotation=15, fontsize=9)
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1.15)
        ax4.legend(['F1-Score (solid)', 'AUC-ROC (transparent)'])
        ax4.axhline(y=0.80, color='black', linestyle='--',
                    linewidth=0.8, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # Deployment recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">Deployment Architecture Recommendation</div>
    """, unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown(f"""
        <div style='border:1px solid #E0E0E0; border-top:4px solid {RED};
                    border-radius:6px; padding:16px; background:white;'>
            <b style='color:{NAVY}; font-size:15px;'>
                Layer 1: Real-Time Blocking
            </b><br><br>
            <b style='color:{RED};'>XGBoost</b><br>
            <span style='color:#444; font-size:13px;'>
                Precision: 1.00 — zero false positives.<br>
                Every fraud flag is a genuine fraud case.<br>
                Suitable for automated transaction blocking<br>
                without manual review overhead.<br><br>
                When XGBoost flags a transaction, block it.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown(f"""
        <div style='border:1px solid #E0E0E0; border-top:4px solid {PURPLE};
                    border-radius:6px; padding:16px; background:white;'>
            <b style='color:{NAVY}; font-size:15px;'>
                Layer 2: Regulatory Reporting
            </b><br><br>
            <b style='color:{PURPLE};'>WoE Scorecard</b><br>
            <span style='color:#444; font-size:13px;'>
                Every decision is explainable feature by feature.<br>
                Satisfies CBK interpretability requirements.<br>
                Credit officers can audit flagged transactions<br>
                with full WoE score breakdowns.<br><br>
                When CBK asks why a transaction was blocked,<br>
                the scorecard provides the audit trail.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
        <b>PSI Monitoring:</b> All four WoE features show PSI values below 0.001
        between training and test sets, confirming no distribution drift.
        In production, PSI should be recomputed monthly on incoming transaction
        distributions. Alert at PSI > 0.10. Retrain at PSI > 0.25.
    </div>
    """, unsafe_allow_html=True)
