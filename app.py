import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# ==============================================================
# CUSTOM CSS
# ==============================================================

st.markdown("""
<style>
    .risk-low {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-left: 5px solid #52b788;
        border-radius: 10px;
        padding: 18px 22px;
        color: white;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #7b3f00, #c47c00);
        border-left: 5px solid #ffd166;
        border-radius: 10px;
        padding: 18px 22px;
        color: white;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .risk-high {
        background: linear-gradient(135deg, #6b1a1a, #c1121f);
        border-left: 5px solid #ff6b6b;
        border-radius: 10px;
        padding: 18px 22px;
        color: white;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .clinical-note {
        background-color: #1e2130;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.78rem;
        color: #a0aec0;
        margin-top: 4px;
    }
    .warn-box {
        background-color: #3d2200;
        border-left: 4px solid #ffa500;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.82rem;
        color: #ffd580;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        border-bottom: 2px solid #3a3f5c;
        padding-bottom: 5px;
        margin-bottom: 14px;
    }
    .model-badge {
        background-color: #1a202c;
        border: 1px solid #3a3f5c;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.82rem;
        color: #a0aec0;
        margin-bottom: 10px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# SESSION STATE
# ==============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==============================================================
# LOAD MODEL
# ==============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/heart_best_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Error loading model: " + str(e))
    st.stop()

# ==============================================================
# HELPER - FEATURE IMPORTANCE FOR XGBOOST PIPELINE
# Model structure detected:
#   Pipeline -> preprocessor (ColumnTransformer) -> XGBClassifier
#   num cols (StandardScaler): Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
#   cat cols (OneHotEncoder):  Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
# ==============================================================

ORIGINAL_FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]


def get_feature_importance_df(pipeline):
    """
    Extract feature importances from the XGBoost Pipeline.
    Returns a DataFrame with columns [Feature, Importance] aggregated
    back to the 11 original input features.
    """
    step_names   = list(pipeline.named_steps.keys())
    preprocessor = pipeline.named_steps[step_names[0]]
    estimator    = pipeline.named_steps[step_names[-1]]

    importances = estimator.feature_importances_

    # Get transformed feature names (e.g. "num__Age", "cat__Sex_M")
    try:
        encoded_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        encoded_names = ["feature_" + str(i) for i in range(len(importances))]

    n = min(len(importances), len(encoded_names))

    # Build a raw importance series
    raw_df = pd.DataFrame({
        "encoded":    encoded_names[:n],
        "importance": importances[:n]
    })

    # Aggregate back to original feature names by stripping prefixes
    # e.g. "num__Age" -> "Age",  "cat__Sex_M" -> "Sex"
    def original_name(encoded):
        # Remove transformer prefix (num__ or cat__)
        if "__" in encoded:
            rest = encoded.split("__", 1)[1]
        else:
            rest = encoded
        # For OHE columns like "Sex_M" or "ChestPainType_ATA",
        # match against known original feature names
        for feat in ORIGINAL_FEATURES:
            if rest == feat or rest.startswith(feat + "_"):
                return feat
        return rest

    raw_df["original"] = raw_df["encoded"].apply(original_name)

    # Sum importances that belong to the same original feature
    agg_df = (
        raw_df.groupby("original", sort=False)["importance"]
        .sum()
        .reset_index()
        .rename(columns={"original": "Feature", "importance": "Importance"})
        .sort_values("Importance", ascending=True)
    )

    return agg_df, raw_df


# ==============================================================
# CLINICAL REFERENCE RANGES
# ==============================================================

CLINICAL_REFS = {
    "RestingBP":   {"normal": (90, 120),  "unit": "mmHg",  "note": "Normal: 90-120 mmHg | Stage 1 HTN: 130-139"},
    "Cholesterol": {"normal": (125, 200), "unit": "mg/dL", "note": "Optimal: <200 | Borderline: 200-239 | High: >=240"},
    "Oldpeak":     {"normal": (0.0, 1.0), "unit": "mm",    "note": "Normal: 0.0-1.0 mm | Significant: >2.0 mm"},
}


def range_warning(field, value):
    ref = CLINICAL_REFS.get(field)
    if ref:
        lo, hi = ref["normal"]
        if value < lo or value > hi:
            return (
                "Value " + str(value) + " " + ref["unit"] +
                " is outside the normal range (" + ref["note"] + ")"
            )
    return None

# ==============================================================
# HEADER
# ==============================================================

st.title("Heart Disease Prediction System")
st.markdown("Predict the risk of heart disease using an AI-powered Machine Learning model.")
st.markdown(
    '<div class="model-badge">Model: XGBoost Classifier &nbsp;|&nbsp; '
    'Pipeline: ColumnTransformer → XGBClassifier &nbsp;|&nbsp; '
    'Objective: binary:logistic</div>',
    unsafe_allow_html=True
)
st.divider()

# ==============================================================
# TABS
# ==============================================================

tab1, tab2, tab3 = st.tabs(["Prediction", "Patient History", "Clinical Guide"])

# --------------------------------------------------------------
# TAB 1 - PREDICTION
# --------------------------------------------------------------

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Patient Demographics & Vitals</div>',
            unsafe_allow_html=True
        )

        age = st.slider("Age", min_value=18, max_value=100, value=40)

        sex = st.selectbox("Sex", ["M", "F"])

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"],
            help="ATA: Atypical Angina | NAP: Non-Anginal Pain | ASY: Asymptomatic | TA: Typical Angina"
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=80, max_value=250, value=120
        )
        st.markdown(
            '<div class="clinical-note">Normal: 90-120 mmHg</div>',
            unsafe_allow_html=True
        )
        w = range_warning("RestingBP", resting_bp)
        if w:
            st.markdown('<div class="warn-box">⚠️ ' + w + '</div>', unsafe_allow_html=True)

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0, max_value=700, value=200
        )
        st.markdown(
            '<div class="clinical-note">Optimal: &lt;200 | Borderline: 200-239 | High: &ge;240</div>',
            unsafe_allow_html=True
        )
        w = range_warning("Cholesterol", cholesterol)
        if w:
            st.markdown('<div class="warn-box">⚠️ ' + w + '</div>', unsafe_allow_html=True)

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No (≤ 120 mg/dL)"
        )

    with col2:
        st.markdown(
            '<div class="section-header">ECG & Stress Test Results</div>',
            unsafe_allow_html=True
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            help="ST: ST-T wave abnormality | LVH: Left Ventricular Hypertrophy"
        )

        max_hr = st.slider(
            "Maximum Heart Rate", min_value=60, max_value=220, value=150
        )
        target_hr = 220 - age
        st.markdown(
            '<div class="clinical-note">Age-predicted max HR: ~' + str(target_hr) + ' bpm (220 - ' + str(age) + ')</div>',
            unsafe_allow_html=True
        )

        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No"
        )

        oldpeak = st.slider(
            "Oldpeak (ST depression in mm)",
            min_value=0.0, max_value=6.5, value=1.0, step=0.1
        )
        st.markdown(
            '<div class="clinical-note">Normal: 0.0-1.0 mm | Significant: &gt;2.0 mm</div>',
            unsafe_allow_html=True
        )
        w = range_warning("Oldpeak", oldpeak)
        if w:
            st.markdown('<div class="warn-box">⚠️ ' + w + '</div>', unsafe_allow_html=True)

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
            help="Up: lower risk | Flat: intermediate | Down: highest risk"
        )

    st.divider()

    if st.button("Predict Heart Disease Risk", use_container_width=True, type="primary"):

        input_data = pd.DataFrame({
            "Age":            [age],
            "Sex":            [sex],
            "ChestPainType":  [chest_pain],
            "RestingBP":      [resting_bp],
            "Cholesterol":    [cholesterol],
            "FastingBS":      [fasting_bs],
            "RestingECG":     [resting_ecg],
            "MaxHR":          [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak":        [oldpeak],
            "ST_Slope":       [st_slope]
        })

        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk_score  = round(probability * 100, 2)

        if risk_score < 30:
            risk_level = "Low"
            risk_class = "risk-low"
            risk_icon  = "✅"
        elif risk_score < 70:
            risk_level = "Moderate"
            risk_class = "risk-moderate"
            risk_icon  = "⚠️"
        else:
            risk_level = "High"
            risk_class = "risk-high"
            risk_icon  = "🚨"

        # -- METRICS --
        st.subheader("Prediction Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score",  str(risk_score) + "%")
        m2.metric("Risk Level",  risk_level)
        m3.metric("Prediction",  "Heart Disease" if prediction == 1 else "No Heart Disease")

        # -- COLOR-CODED RISK CARD --
        st.markdown(
            '<div class="' + risk_class + '">' +
            risk_icon + " " + risk_level + " Risk of Heart Disease — " + str(risk_score) + "%" +
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # -- GAUGE CHART --
        if risk_level == "Low":
            gauge_color = "#52b788"
        elif risk_level == "Moderate":
            gauge_color = "#ffd166"
        else:
            gauge_color = "#ff6b6b"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            title={"text": "Heart Disease Risk (%)"},
            delta={
                "reference": 50,
                "increasing": {"color": "#ff6b6b"},
                "decreasing": {"color": "#52b788"}
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": gauge_color},
                "steps": [
                    {"range": [0,  30],  "color": "#1a472a"},
                    {"range": [30, 70],  "color": "#3d2200"},
                    {"range": [70, 100], "color": "#6b1a1a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": risk_score
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ------------------------------------------------------
        # FEATURE IMPORTANCE (XGBoost Pipeline - aggregated)
        # ------------------------------------------------------
        st.subheader("Feature Importance & Explainability")

        try:
            agg_df, raw_df = get_feature_importance_df(model)

            # -- Aggregated chart (by original feature) --
            fig_agg = px.bar(
                agg_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#52b788", "#ffd166", "#ff6b6b"],
                title="Feature Importances — Aggregated to Original Features (XGBoost)",
                labels={"Importance": "Importance Score"}
            )
            fig_agg.update_layout(coloraxis_showscale=False, height=420)
            st.plotly_chart(fig_agg, use_container_width=True)

            # -- Detailed chart (encoded features) in expander --
            with st.expander("Show detailed encoded feature importances"):
                raw_sorted = raw_df.sort_values("importance", ascending=True)
                fig_raw = px.bar(
                    raw_sorted,
                    x="importance",
                    y="encoded",
                    orientation="h",
                    color="importance",
                    color_continuous_scale=["#52b788", "#ffd166", "#ff6b6b"],
                    title="Feature Importances — All Encoded Features",
                    labels={"importance": "Importance Score", "encoded": "Encoded Feature"}
                )
                fig_raw.update_layout(coloraxis_showscale=False, height=500)
                st.plotly_chart(fig_raw, use_container_width=True)

        except Exception as ex:
            st.warning("Could not extract feature importances: " + str(ex))

        # -- PATIENT SUMMARY TABLE --
        st.subheader("Patient Summary")
        summary = pd.DataFrame({
            "Feature": [
                "Age", "Sex", "Chest Pain Type", "Resting BP (mmHg)",
                "Cholesterol (mg/dL)", "Fasting Blood Sugar",
                "Resting ECG", "Max Heart Rate", "Exercise Angina",
                "Oldpeak (mm)", "ST Slope"
            ],
            "Value": [
                age, sex, chest_pain, resting_bp, cholesterol,
                "Yes" if fasting_bs == 1 else "No",
                resting_ecg, max_hr,
                "Yes" if exercise_angina == "Y" else "No",
                oldpeak, st_slope
            ],
            "Normal Range": [
                "18-100 yrs", "--", "--",
                "90-120 mmHg", "<200 mg/dL",
                "<=120 mg/dL", "Normal",
                str(target_hr) + " bpm target", "No",
                "0.0-1.0 mm", "Up"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # -- SAVE TO HISTORY --
        st.session_state.history.append({
            "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Age":            age,
            "Sex":            sex,
            "Chest Pain":     chest_pain,
            "Resting BP":     resting_bp,
            "Cholesterol":    cholesterol,
            "Max HR":         max_hr,
            "Oldpeak":        oldpeak,
            "Risk Score (%)": risk_score,
            "Risk Level":     risk_level,
            "Prediction":     "Heart Disease" if prediction == 1 else "No Heart Disease"
        })

        # -- DOWNLOAD REPORT --
        report = pd.DataFrame({
            "Date":            [datetime.now()],
            "Age":             [age],
            "Sex":             [sex],
            "ChestPainType":   [chest_pain],
            "RestingBP":       [resting_bp],
            "Cholesterol":     [cholesterol],
            "FastingBS":       [fasting_bs],
            "RestingECG":      [resting_ecg],
            "MaxHR":           [max_hr],
            "ExerciseAngina":  [exercise_angina],
            "Oldpeak":         [oldpeak],
            "ST_Slope":        [st_slope],
            "Prediction":      ["High Risk" if prediction == 1 else "Low Risk"],
            "Risk_Percentage": [risk_score],
            "Risk_Level":      [risk_level]
        })
        st.download_button(
            label="Download Report (CSV)",
            data=report.to_csv(index=False),
            file_name="heart_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv",
            mime="text/csv"
        )

# --------------------------------------------------------------
# TAB 2 - PATIENT HISTORY
# --------------------------------------------------------------

with tab2:
    st.subheader("Patient Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet. Run a prediction in the Prediction tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        color_map = {"Low": "#52b788", "Moderate": "#ffd166", "High": "#ff6b6b"}
        marker_colors = hist_df["Risk Level"].map(color_map).fillna("#a0aec0").tolist()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=list(range(1, len(hist_df) + 1)),
            y=hist_df["Risk Score (%)"],
            mode="lines+markers",
            marker=dict(
                color=marker_colors,
                size=13,
                line=dict(color="white", width=1.5)
            ),
            line=dict(color="#a0aec0", width=2),
            text=hist_df["Risk Level"],
            customdata=hist_df["Timestamp"],
            hovertemplate=(
                "<b>Run %{x}</b><br>"
                "Risk Score: %{y}%<br>"
                "Level: %{text}<br>"
                "Time: %{customdata}<extra></extra>"
            )
        ))
        fig_hist.add_hline(
            y=30, line_dash="dash", line_color="#52b788",
            annotation_text="Low threshold (30%)",
            annotation_position="top left"
        )
        fig_hist.add_hline(
            y=70, line_dash="dash", line_color="#ff6b6b",
            annotation_text="High threshold (70%)",
            annotation_position="top left"
        )
        fig_hist.update_layout(
            title="Risk Score Trend Across Predictions",
            xaxis_title="Prediction Run",
            yaxis_title="Risk Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        dist = hist_df["Risk Level"].value_counts().reset_index()
        dist.columns = ["Risk Level", "Count"]
        fig_dist = px.pie(
            dist,
            names="Risk Level",
            values="Count",
            color="Risk Level",
            color_discrete_map=color_map,
            title="Risk Level Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with c2:
            st.download_button(
                label="Download Full History (CSV)",
                data=hist_df.to_csv(index=False),
                file_name="heart_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

# --------------------------------------------------------------
# TAB 3 - CLINICAL GUIDE
# --------------------------------------------------------------

with tab3:
    st.subheader("Clinical Reference Guide")

    st.markdown("""
### Input Feature Reference

| Feature | Normal Range | Clinical Significance |
|---|---|---|
| **Resting BP** | 90-120 mmHg | >140 mmHg = Stage 2 Hypertension |
| **Cholesterol** | <200 mg/dL | >240 mg/dL = High cardiovascular risk |
| **Fasting Blood Sugar** | <=120 mg/dL | >120 mg/dL linked to diabetes risk |
| **Max Heart Rate** | ~220 minus Age | Lower max HR may indicate poor cardiac fitness |
| **Oldpeak** | 0.0-1.0 mm | >2.0 mm = Significant ST depression |

---

### Chest Pain Types
- **ATA** (Atypical Angina) - Chest pain not fully typical of cardiac angina
- **NAP** (Non-Anginal Pain) - Unrelated to heart; often musculoskeletal
- **ASY** (Asymptomatic) - No chest pain; paradoxically associated with higher risk
- **TA** (Typical Angina) - Classic pressure/squeezing during exertion

---

### ST Slope
| Slope | Risk |
|---|---|
| **Up** (Upsloping) | Generally normal / lower risk |
| **Flat** | Intermediate concern |
| **Down** (Downsloping) | Highest association with disease |

---

### Risk Level Interpretation
| Level | Score | Recommendation |
|---|---|---|
| Low | < 30% | Maintain healthy lifestyle; routine check-ups |
| Moderate | 30-70% | Consult a physician; lifestyle modifications |
| High | > 70% | Immediate medical evaluation strongly recommended |

---

### About the Model
- **Type**: XGBoost Classifier (XGBClassifier)
- **Pipeline**: ColumnTransformer (StandardScaler + OneHotEncoder) -> XGBClassifier
- **Objective**: binary:logistic
- **Numerical features** (StandardScaler): Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
- **Categorical features** (OneHotEncoder): Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

> **Disclaimer**: This tool is for educational and research purposes only.
> It does not replace professional medical diagnosis or advice.
    """)

# ==============================================================
# FOOTER
# ==============================================================

st.divider()
st.caption(
    "Developed by Kr. Roshan | Heart Disease Prediction System | "
    "XGBoost Pipeline | Enhanced with explainability, patient history & clinical references"
)
