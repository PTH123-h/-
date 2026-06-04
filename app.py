from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [APP_DIR / "models" / "rf_model.pkl", APP_DIR / "rf_model.pkl"]
FEATURES = ["CREA", "HCT", "MCHC", "MCV", "NLR"]

FEATURE_INFO = {
    "CREA": {
        "name": "Creatinine",
        "unit": "umol/L",
        "min": 30.8,
        "max": 442.2,
        "default": 81.95,
        "step": 0.1,
        "format": "%.1f",
        "expected_low": 46.685,
        "expected_high": 160.5,
    },
    "HCT": {
        "name": "Hematocrit",
        "unit": "%",
        "min": 18.0,
        "max": 59.9,
        "default": 38.6,
        "step": 0.1,
        "format": "%.1f",
        "expected_low": 29.085,
        "expected_high": 46.3,
    },
    "MCHC": {
        "name": "Mean corpuscular hemoglobin concentration",
        "unit": "g/L",
        "min": 252.0,
        "max": 351.0,
        "default": 320.5,
        "step": 1.0,
        "format": "%.0f",
        "expected_low": 296.0,
        "expected_high": 338.15,
    },
    "MCV": {
        "name": "Mean corpuscular volume",
        "unit": "fL",
        "min": 54.9,
        "max": 108.5,
        "default": 92.65,
        "step": 0.1,
        "format": "%.1f",
        "expected_low": 68.785,
        "expected_high": 102.46,
    },
    "NLR": {
        "name": "Neutrophil-to-lymphocyte ratio",
        "unit": "ratio",
        "min": 0.296,
        "max": 123.429,
        "default": 7.229,
        "step": 0.001,
        "format": "%.3f",
        "expected_low": 2.069,
        "expected_high": 38.846,
    },
}


st.set_page_config(page_title="T2RF Risk Calculator", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .header-band {
        border-bottom: 1px solid #d9e2ec;
        padding: 0.4rem 0 1.1rem 0;
        margin-bottom: 1.2rem;
    }
    .header-band h1 {
        font-size: 2.15rem;
        line-height: 1.15;
        margin: 0 0 0.35rem 0;
        color: #111827;
        font-weight: 800;
        letter-spacing: 0;
    }
    .header-band p {
        color: #475569;
        font-size: 1rem;
        margin: 0;
    }
    .metric-box {
        border: 1px solid #d5dee8;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        background: #fbfdff;
    }
    .probability {
        font-size: 3.1rem;
        line-height: 1.05;
        font-weight: 850;
        color: #0f172a;
        margin: 0.15rem 0 0.35rem 0;
    }
    .risk-high {
        color: #b42318;
        font-size: 1.2rem;
        font-weight: 800;
    }
    .risk-low {
        color: #067647;
        font-size: 1.2rem;
        font-weight: 800;
    }
    .caption-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    div.stButton > button:first-child {
        width: 100%;
        min-height: 3rem;
        border-radius: 7px;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_payload() -> tuple[object, float, dict[str, object]]:
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Cannot find models/rf_model.pkl.")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("The RF model file is not in the expected payload format.")
    threshold = float(payload.get("threshold", 0.5))
    features = payload.get("features", FEATURES)
    if list(features) != FEATURES:
        raise ValueError(f"Model features do not match calculator features: {features}")
    return payload["model"], threshold, payload


def make_input_frame(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)


def predict_probability(model: object, x_input: pd.DataFrame) -> float:
    probability = np.asarray(model.predict_proba(x_input)[:, 1], dtype=float)
    return float(probability[0])


def render_number_input(feature: str) -> float:
    info = FEATURE_INFO[feature]
    return float(
        st.number_input(
            f"{feature} ({info['unit']})",
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["default"]),
            step=float(info["step"]),
            format=str(info["format"]),
            help=f"{info['name']}; training-set median {info['default']} {info['unit']}.",
        )
    )


def expected_range_warnings(values: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    for feature, value in values.items():
        info = FEATURE_INFO[feature]
        low = float(info["expected_low"])
        high = float(info["expected_high"])
        if value < low or value > high:
            warnings.append(f"{feature}: value is outside the training 5th-95th percentile range ({low:g}-{high:g}).")
    return warnings


st.markdown(
    """
    <div class="header-band">
      <h1>T2RF Risk Calculator</h1>
      <p>Random Forest prediction tool for type 2 respiratory failure risk in elderly patients with acute COPD and coronary artery disease.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, threshold, payload = load_model_payload()

left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.subheader("Laboratory Inputs")
    st.caption("Enter routine admission laboratory indicators, then click Calculate Risk.")

    with st.form("rf_risk_calculator"):
        values: dict[str, float] = {}
        col_a, col_b = st.columns(2)
        for index, feature in enumerate(FEATURES):
            with col_a if index % 2 == 0 else col_b:
                values[feature] = render_number_input(feature)

        submitted = st.form_submit_button("Calculate Risk", type="primary")

    warnings = expected_range_warnings(values)
    if warnings:
        with st.expander("Input range notes", expanded=False):
            for item in warnings:
                st.warning(item)

with right:
    st.subheader("Prediction Result")
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)

    if submitted:
        x_input = make_input_frame(values)
        probability = predict_probability(model, x_input)
        risk_group = "High risk" if probability >= threshold else "Low risk"
        st.markdown(f'<div class="probability">{probability * 100:.1f}%</div>', unsafe_allow_html=True)
        st.progress(min(max(probability, 0.0), 1.0))
        if risk_group == "High risk":
            st.markdown('<div class="risk-high">High risk</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">Low risk</div>', unsafe_allow_html=True)
        st.write(f"Risk threshold: `{threshold:.4f}`")
        with st.expander("Entered values", expanded=False):
            st.dataframe(x_input, hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="probability">--</div>', unsafe_allow_html=True)
        st.write("Waiting for input.")
        st.progress(0.0)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class="caption-note">
        This calculator is intended for research presentation and clinical discussion only.
        It is not a standalone diagnostic, treatment, or triage tool.
        </p>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Model: Random Forest; predictors: CREA, HCT, MCHC, MCV, and NLR. "
    "The model uses the current exploratory development cohort and should be validated prospectively before clinical use."
)
