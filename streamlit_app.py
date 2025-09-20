
import json, joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="AMI–IABP One-Year Mortality Risk Calculator",
                   layout="wide", page_icon="")

# ── Custom styling (labels bigger + bold) ──────────────────────
st.markdown("""
<style>
.block-container {padding-top:1rem;}

/* Number input labels (continuous vars) */
div[data-testid="stNumberInput"] label {
  font-size:1.35rem !important;
  font-weight:800 !important;
  margin-bottom:8px !important;
  line-height:1.2 !important;
  display:block !important;
  white-space:normal !important;
}

/* Radio button labels (binary vars) */
div[data-testid="stRadio"] label {
  font-size:1.2rem !important;
  font-weight:800 !important;
}

.badge{display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;font-weight:700;}
.badge.low{background:#E8F6EE;border:1px solid #57C785;color:#1D774C;}
.badge.inter{background:#FFF6E0;border:1px solid #F9B300;color:#8A5A00;}
.badge.high{background:#FFEAEA;border:1px solid #EF6262;color:#7E1A1A;}
.metric-row{display:flex;align-items:center;gap:12px;margin:6px 0 2px;}
.metric-num{font-weight:800;font-size:2rem;line-height:1;}
.badge.big{font-size:1.2rem;padding:0.25rem 0.7rem;}
</style>
""", unsafe_allow_html=True)

st.title("AMI–IABP One-Year Mortality Risk Calculator")
st.caption("Prediction of one-year mortality in AMI patients supported with IABP (logistic regression; internal training, external validation).")

# ── Load model + metadata ─────────────────────────────────────
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()

model_path = APP_DIR / "model_log_reg.joblib"
meta_path  = APP_DIR / "features.json"

model = joblib.load(model_path)
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

train_order   = meta["features"]
binary_feats  = set(meta["binary_features"])
labels        = meta["labels"]
thr_low       = float(meta["risk_thresholds"]["low"])
thr_high      = float(meta["risk_thresholds"]["high"])

def pretty(code: str) -> str:
    return labels.get(code, code.replace("_"," ").title())

# ── Clinical groupings ────────────────────────────────────────
UI_GROUPS = [
    ("Demographics & Hospitalization", ["age", "ICU_LOS"]),
    ("Hematology", ["hemoglobin_min", "hemoglobin_max", "rbc_count_max"]),
    ("Renal Function", ["creatinine_min", "creatinine_max", "eGFR_CKD_EPI_21"]),
    ("Inflammatory / Immune Markers", ["neutrophils_abs_min", "eosinophils_abs_max",
                                       "neutrophils_pct_min", "eosinophils_pct_max"]),
    ("Metabolic & Biochemistry", ["sodium_max", "lactate_max", "AST_min"]),
    ("Hemodynamics", ["dbp_post_iabp"]),
    ("Therapies & Interventions", ["beta_blocker_use", "ticagrelor_use", "invasive_ventilation"]),
]

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Notes")
    st.caption("Interpret results in clinical context; do not use in isolation.")

# ── Inputs ────────────────────────────────────────────────────
st.subheader("Patient Inputs")
inputs = {}
for header, group in UI_GROUPS:
    st.markdown(f"### {header}")
    bin_list = [f for f in group if f in binary_feats]
    num_list = [f for f in group if f not in binary_feats]

    if bin_list:
        for f in bin_list:
            choice = st.radio(pretty(f), ["No", "Yes"], horizontal=True, key=f"bin_{f}")
            inputs[f] = 1 if choice == "Yes" else 0

    if num_list:
        cols = st.columns(3)
        for i, f in enumerate(num_list):
            c = cols[i % 3]
            inputs[f] = float(c.number_input(pretty(f), value=0.0, format="%.3f", key=f"num_{f}"))

for f in train_order:
    inputs.setdefault(f, 0 if f in binary_feats else 0.0)

# ── Predict ───────────────────────────────────────────────────
st.markdown("---")
if st.button("Calculate Risk", type="primary", use_container_width=True):
    X = pd.DataFrame([[inputs.get(f, 0) for f in train_order]], columns=train_order)
    for f in train_order:
        X[f] = X[f].astype(int) if f in binary_feats else pd.to_numeric(X[f], errors="coerce").fillna(0.0)

    try:
        risk = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        score = float(getattr(model, "decision_function")(X)[0])
        risk  = 1.0 / (1.0 + np.exp(-score))

    if risk < thr_low:
        badge = f'<span class="badge low">Low Risk (&lt; {int(thr_low*100)}%)</span>'
    elif risk < thr_high:
        badge = f'<span class="badge inter">Intermediate Risk ({int(thr_low*100)}–{int(thr_high*100)}%)</span>'
    else:
        badge = f'<span class="badge high">High Risk (≥ {int(thr_high*100)}%)</span>'

    risk_pct = f"{risk*100:.1f}%"
    badge_big = badge.replace('class="badge ', 'class="badge big ', 1)
    st.markdown(
        f'<div class="metric-row"><div class="metric-num">{risk_pct}</div>{badge_big}</div>'
        '<div style="font-weight:600;margin-top:2px;">Predicted 1-Year Mortality</div>',
        unsafe_allow_html=True
    )

    st.info("This estimate is intended to support, not replace, clinical judgment, and should be interpreted within the broader clinical context.")

st.markdown("---")
st.caption("AMI–IABP One-Year Mortality Risk Calculator • Streamlit App• Developed by Zampawala et al., 2025")
