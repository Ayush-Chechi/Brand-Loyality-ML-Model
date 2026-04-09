from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import streamlit as st

from src.brand_loyalty.config import CFG
from src.brand_loyalty.data import canonicalize_features
from src.brand_loyalty.train import train

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "models" / "random_forest_tuned.joblib"
DATASET_PATH = Path(CFG.dataset_path)
REPORT_PATH = ARTIFACTS_DIR / "reports" / "project_experiment_metadata.json"

QUESTION_LABELS = {
    "Brand": "Which smartphone brand do you currently use?",
    "Usage Duration": "How long have you been using this brand?",
    "Experience": "Have you had a good overall experience with this brand?",
    "Discount Influence": "Do discounts or offers influence your smartphone purchase decision?",
    "Peer Influence": "Do your friends mostly use the same smartphone brand?",
    "Decision Factor": "What is the MOST important factor influencing your smartphone choice?",
    "Social Engagement": "How often do you engage with this smartphone brand online?",
    "Price Importance": "How important is price when choosing a smartphone?",
}


@st.cache_data(show_spinner=False)
def load_options(dataset_path: Path) -> Dict[str, List[str]]:
    df = pd.read_excel(dataset_path)
    rename_map = {v: k for k, v in CFG.source_feature_columns.items()}
    df = df.rename(columns=rename_map)
    out: Dict[str, List[str]] = {}
    for c in CFG.all_feature_columns:
        vals = (
            df[c]
            .dropna()
            .astype(str)
            .str.replace("\u2013", "-", regex=False)
            .str.strip()
            .unique()
            .tolist()
        )
        out[c] = sorted(vals)
    return out


@st.cache_data(show_spinner=False)
def load_report(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _predict_with_confidence(input_dict: Dict[str, Any]) -> Tuple[str, float]:
    pipeline = joblib.load(MODEL_PATH)
    X = canonicalize_features(pd.DataFrame([input_dict], columns=CFG.all_feature_columns))[CFG.all_feature_columns]
    proba = pipeline.predict_proba(X)[0]
    pred_id = int(pipeline.predict(X)[0])
    label = CFG.loyalty_binary_id_to_name[pred_id]
    return label, float(max(proba) * 100.0)


def _result_card(label: str, confidence: float) -> None:
    color = {"Loyal": "#22c55e", "Not Loyal": "#ef4444"}.get(label, "#64748b")
    st.markdown(
        f"""
        <div style="border:1px solid rgba(148,163,184,.35); border-left:8px solid {color}; border-radius:12px; padding:.9rem 1rem; margin-top:.7rem;">
          <h3 style="margin:0 0 .3rem 0;">Predicted Loyalty Class</h3>
          <p style="margin:.1rem 0; font-size:1.08rem;"><strong>{label}</strong></p>
          <p style="margin:.1rem 0;">Confidence Score: <strong>{confidence:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_form(options: Dict[str, List[str]]) -> Dict[str, Any]:
    st.subheader("Input Survey")
    data: Dict[str, Any] = {}
    for i, key in enumerate(CFG.all_feature_columns, start=1):
        st.markdown(f"**{i}. {QUESTION_LABELS[key]}**")
        if key in ("Experience", "Peer Influence"):
            data[key] = st.selectbox(" ", ["Yes", "No"], key=f"q_{key}", label_visibility="collapsed")
        else:
            data[key] = st.selectbox(" ", options.get(key, []), key=f"q_{key}", label_visibility="collapsed")
    return data


def main() -> None:
    st.set_page_config(page_title="Brand Loyalty Predictor", layout="wide")
    st.title("Smartphone Brand Loyalty Prediction")
    st.caption("Leakage-free binary ML system: Loyal vs Not Loyal")
    st.markdown(
        """
        <style>
        div.stButton > button:first-child { width:100%; min-height:3rem; font-weight:700; }
        div[data-testid="stSelectbox"] { margin-bottom:.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pipeline Control")
        st.write(f"Dataset: `{DATASET_PATH.as_posix()}`")
        st.write(f"Model: `{MODEL_PATH.as_posix()}`")
        if st.button("Train / Refresh Artifacts"):
            with st.spinner("Training models and generating plots/report..."):
                train(str(DATASET_PATH), str(ARTIFACTS_DIR))
            st.success("Training complete.")

    if not DATASET_PATH.exists():
        st.error("`project dataset.xlsx` not found in project root.")
        st.stop()

    options = load_options(DATASET_PATH)
    report = load_report(REPORT_PATH)
    left, right = st.columns([2, 3], gap="large")

    with left:
        inputs = build_form(options)
        if st.button("Predict Loyalty", type="primary", disabled=not MODEL_PATH.exists()):
            lbl, conf = _predict_with_confidence(inputs)
            st.session_state["pred"] = (lbl, conf)
        if st.session_state.get("pred"):
            lbl, conf = st.session_state["pred"]
            _result_card(lbl, conf)

    with right:
        st.subheader("Experimental Results & Insights")
        if report:
            st.markdown("### Dataset Overview")
            st.write(f"- Samples: **500**")
            st.write(f"- Leakage removed: **{report.get('excluded_feature', 'N/A')}**")

            st.markdown("### Model Performance Table")
            perf = pd.DataFrame(report.get("models", []))
            if not perf.empty:
                st.dataframe(perf, width="stretch", hide_index=True)

            st.markdown("### Overfitting Check")
            over = report.get("overfitting_check", {})
            train_acc = over.get("train_accuracy_rf")
            test_acc = over.get("test_accuracy_rf")
            gap = None if train_acc is None or test_acc is None else float(train_acc) - float(test_acc)
            st.write(f"- Train Accuracy (RF): **{train_acc:.4f}**" if train_acc is not None else "- Train Accuracy (RF): N/A")
            st.write(f"- Test Accuracy (RF): **{test_acc:.4f}**" if test_acc is not None else "- Test Accuracy (RF): N/A")
            st.write(f"- Gap: **{gap:.4f}**" if gap is not None else "- Gap: N/A")

            st.markdown("### Feature Importance (Random Forest)")
            for i, row in enumerate(report.get("rf_top_features", [])[:5], start=1):
                st.write(f"{i}. `{row['feature']}` -> **{row['importance']:.4f}**")

            st.markdown("### Behavioral Insights")
            st.write("- Stronger **experience** and **usage duration** scores increase loyalty likelihood.")
            st.write("- **Price-discount interactions** influence medium vs low loyalty boundaries.")
            st.write("- **Social engagement + peer context** contributes to loyalty differentiation.")
            st.caption("Confidence score is the model's predicted probability for the selected class.")
        else:
            st.info("No project report found yet. Use 'Train / Refresh Artifacts' in the sidebar.")

    st.divider()
    st.subheader("Plots")
    pdir = ARTIFACTS_DIR / "plots"
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        f = pdir / "class_distribution.png"
        if f.exists():
            st.image(str(f), caption="Class Distribution", width="stretch")
    with p2:
        f = pdir / "model_comparison_accuracy.png"
        if f.exists():
            st.image(str(f), caption="Model Comparison", width="stretch")
    with p3:
        f = pdir / "random_forest_tuned_cm.png"
        if f.exists():
            st.image(str(f), caption="RF Confusion Matrix", width="stretch")
    with p4:
        f = pdir / "random_forest_feature_importance.png"
        if f.exists():
            st.image(str(f), caption="RF Feature Importance", width="stretch")


if __name__ == "__main__":
    main()

