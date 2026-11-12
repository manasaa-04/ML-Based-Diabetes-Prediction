"""ML-Based Diabetes Prediction: diabetes and hospital-associated UTI assessment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


st.set_page_config(
    page_title="ML-Based Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DISCLAIMER = (
    "This application is an educational project, not a medical device. "
    "Its predictions are not diagnoses. Consult a qualified healthcare professional."
)

DIABETES_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income"
]


@st.cache_resource
def load_models():
    diabetes_model = joblib.load(ROOT / "models" / "diabetes_model.pkl")
    uti_model = joblib.load(ROOT / "models" / "uti_model.pkl")
    diabetes_data = pd.read_csv(ROOT / "diabetes_cleaned.csv")
    uti_data = pd.read_csv(ROOT / "uti_female.csv")
    return diabetes_model, uti_model, diabetes_data, uti_data


try:
    diabetes_model, uti_model, diabetes_data, uti_data = load_models()
except Exception as error:
    st.error(f"Could not load the models or datasets: {error}")
    st.stop()


# -------------------- App styling --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(190, 210, 255, 0.34), transparent 30%),
            radial-gradient(circle at 92% 18%, rgba(190, 225, 230, 0.34), transparent 28%),
            linear-gradient(135deg, #e9eefb 0%, #edf3f8 48%, #e5f1f1 100%);
        color: #17243a;
    }
    [data-testid="stHeader"] { background: rgba(233,238,251,0.88); }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #172b4d; font-family: 'Manrope', sans-serif; }
    .hero {
        background: linear-gradient(120deg, #102b50 0%, #174b70 58%, #267b83 100%);
        color: white; padding: 2rem 2.2rem; border-radius: 22px;
        margin-bottom: 1.4rem; box-shadow: 0 12px 32px rgba(24,55,90,.12);
    }
    .hero h1 { color: white !important; margin: 0 0 .35rem 0; font-size: 2.15rem; }
    .hero p { color: #e4eff8; margin: 0; font-size: 1.02rem; }
    .eyebrow { color: #c6e9e9; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .72rem; }
    .step-pill {
        display: inline-block; padding: .42rem .8rem; border-radius: 999px;
        background: #e7f3f4; color: #176b70; font-weight: 700; font-size: .82rem;
        margin: .2rem 0 1rem 0;
    }
    .choice-card {
        background: white; border: 1px solid #e3eaf2; border-radius: 20px;
        padding: 1.35rem; min-height: 190px; box-shadow: 0 8px 24px rgba(28,53,87,.05);
    }
    .choice-icon { font-size: 2.8rem; margin-bottom: .45rem; }
    .choice-title { font-family: 'Manrope',sans-serif; font-size: 1.25rem; font-weight: 800; color: #183452; }
    .choice-copy { color: #607086; font-size: .92rem; }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 12px; min-height: 3rem; font-weight: 700;
        border: 1px solid #d8e3ee; transition: all .15s ease;
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: #176f78; border-color: #176f78; color: white;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        border-color: #176f78; color: #176f78;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background: #125b63; color: white;
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e5ebf2; padding: 1rem;
        border-radius: 15px; box-shadow: 0 5px 18px rgba(28,53,87,.04);
    }
    [data-testid="stForm"] {
        background: white; border: 1px solid #e5ebf2; border-radius: 18px;
        padding: 1.1rem 1.2rem;
    }
    [data-testid="stExpander"] {
        background: white; border: 1px solid #e5ebf2; border-radius: 14px;
    }
    .footer { color: #718096; font-size: .82rem; text-align: center; padding-top: 1.5rem; }
    .muted { color: #62748a; }
</style>
""", unsafe_allow_html=True)


# -------------------- Session state --------------------
DEFAULTS = {
    "page": "gender",
    "gender": None,
    "assessment": None,
    "diabetes_result": None,
    "uti_result": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page):
    st.session_state.page = page


def reset_assessment():
    st.session_state.diabetes_result = None
    st.session_state.uti_result = None


def render_header(step, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Clinical technology • Academic project</div>
            <h1>🩺 ML-Based Diabetes Prediction</h1>
            <p>{subtitle}</p>
        </div>
        <div class="step-pill">STEP {step} OF 4</div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(step / 4)


def render_footer():
    st.markdown(
        '<div class="footer">ML-Based Diabetes Prediction · Academic project · '
        'Not intended for medical diagnosis</div>',
        unsafe_allow_html=True,
    )


def show_model_information():
    with st.expander("Model performance information"):
        st.subheader("Diabetes model")
        st.write(
            "Trained using the CDC Diabetes Health Indicators dataset, "
            "which contains male and female survey records."
        )
        st.write(
            "Held-out test ROC-AUC: approximately 0.811. "
            "The model's default classification threshold is 0.50."
        )
        st.subheader("Hospital-associated UTI model")
        st.write(
            "Trained on female hospital admission records from the Dryad dataset. "
            "Held-out test ROC-AUC: approximately 0.835."
        )
        st.warning(
            "The UTI model has low precision at its current threshold. "
            "Many positive predictions are false positives. Further validation "
            "and threshold selection are required."
        )

    with st.expander("Fairness evaluation"):
        show_fairness()

    with st.expander("Probability calibration"):
        st.subheader("Diabetes model calibration")
        col1, col2 = st.columns(2)
        col1.metric("Brier score before calibration", "0.1847")
        col2.metric("Brier score after calibration", "0.1075", delta="-0.0771")
        st.write(
            "Probability calibration improves the agreement between predicted "
            "probabilities and observed outcomes."
        )
        reliability_path = ROOT / "reports" / "diabetes_reliability.png"
        if reliability_path.exists():
            st.image(
                reliability_path,
                caption="Reliability curve: before and after sigmoid calibration",
                use_container_width=True,
            )
        else:
            st.info("The calibration reliability plot was not found in the reports folder.")
        st.caption(
            "Evaluation was performed on the held-out test dataset. Calibration "
            "improved the Brier score, but the model is not perfectly calibrated."
        )


def diabetes_input_form(sex_code):
    st.header("Diabetes risk assessment")
    st.caption(
        "Enter health information using the categories from the CDC survey dataset."
    )

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            high_bp = st.selectbox(
                "High blood pressure", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            high_chol = st.selectbox(
                "High cholesterol", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            chol_check = st.selectbox(
                "Cholesterol checked in last 5 years", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            bmi = st.number_input("BMI", 12.0, 70.0, 28.0, 0.1)
            smoker = st.selectbox(
                "Smoked at least 100 cigarettes in lifetime", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            stroke = st.selectbox(
                "History of stroke", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            heart = st.selectbox(
                "Coronary heart disease or heart attack", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            activity = st.selectbox(
                "Physical activity outside work", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            fruits = st.selectbox(
                "Consume fruit at least once daily", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            veggies = st.selectbox(
                "Consume vegetables at least once daily", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        with col2:
            alcohol = st.selectbox(
                "Heavy alcohol consumption", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            healthcare = st.selectbox(
                "Have healthcare coverage", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            cost = st.selectbox(
                "Could not see a doctor because of cost", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            general_health = st.selectbox(
                "General health",
                [1, 2, 3, 4, 5],
                format_func=lambda x: {
                    1: "Excellent", 2: "Very good", 3: "Good",
                    4: "Fair", 5: "Poor"
                }[x]
            )
            mental_health = st.slider(
                "Days of poor mental health in past 30 days", 0, 30, 0
            )
            physical_health = st.slider(
                "Days of poor physical health in past 30 days", 0, 30, 0
            )
            walking = st.selectbox(
                "Serious difficulty walking or climbing stairs", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            age = st.selectbox(
                "Age category",
                list(range(1, 14)),
                format_func=lambda x: {
                    1: "18–24", 2: "25–29", 3: "30–34",
                    4: "35–39", 5: "40–44", 6: "45–49",
                    7: "50–54", 8: "55–59", 9: "60–64",
                    10: "65–69", 11: "70–74", 12: "75–79",
                    13: "80 or older"
                }[x]
            )
            education = st.selectbox(
                "Education level",
                list(range(1, 7)),
                format_func=lambda x: {
                    1: "No schooling / kindergarten",
                    2: "Grades 1–8",
                    3: "Grades 9–11",
                    4: "Grade 12 or GED",
                    5: "College 1–3 years",
                    6: "College 4 years or more"
                }[x]
            )
            income = st.selectbox(
                "Annual household income category (USD)",
                list(range(1, 9)),
                format_func=lambda x: {
                    1: "Below $10,000",
                    2: "$10,000–14,999",
                    3: "$15,000–19,999",
                    4: "$20,000–24,999",
                    5: "$25,000–34,999",
                    6: "$35,000–49,999",
                    7: "$50,000–74,999",
                    8: "$75,000 or more"
                }[x]
            )

        submitted = st.form_submit_button(
            "Predict diabetes risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        values = {
            "HighBP": high_bp,
            "HighChol": high_chol,
            "CholCheck": chol_check,
            "BMI": bmi,
            "Smoker": smoker,
            "Stroke": stroke,
            "HeartDiseaseorAttack": heart,
            "PhysActivity": activity,
            "Fruits": fruits,
            "Veggies": veggies,
            "HvyAlcoholConsump": alcohol,
            "AnyHealthcare": healthcare,
            "NoDocbcCost": cost,
            "GenHlth": general_health,
            "MentHlth": mental_health,
            "PhysHlth": physical_health,
            "DiffWalk": walking,
            "Sex": sex_code,
            "Age": age,
            "Education": education,
            "Income": income
        }
        return pd.DataFrame([values], columns=DIABETES_FEATURES)

    return None


# --------------------------------------------------
# Diabetes result and SHAP
# --------------------------------------------------

@st.cache_resource
def make_diabetes_explainer():
    background = diabetes_data.sample(
        min(50, len(diabetes_data)),
        random_state=42
    )[DIABETES_FEATURES]

    def predict_function(data):
        frame = pd.DataFrame(data, columns=DIABETES_FEATURES)
        return diabetes_model.predict_proba(frame)[:, 1]

    return shap.Explainer(
        predict_function,
        background,
        feature_names=DIABETES_FEATURES
    )



def show_diabetes_result(X):
    probability = float(diabetes_model.predict_proba(X)[0, 1])

    # Distance from the decision threshold
    confidence_indicator = min(abs(probability - 0.5) * 2, 1.0)

    st.subheader("Diabetes model result")

    col1, col2 = st.columns(2)

    col1.metric(
        "Calibrated positive-class probability",
        f"{probability:.1%}"
    )

    col2.metric(
        "Model classification at threshold 0.50",
        "Positive" if probability >= 0.5 else "Negative"
    )

    st.caption(
        "Positive means the CDC survey target: diabetes or prediabetes. "
        "This calibrated probability is an estimate based on survey data, "
        "not a clinically validated individual risk."
    )

    # Uncertainty flagging
    st.subheader("Prediction uncertainty")

    st.write(
        "This indicator shows how far the prediction is from the "
        "0.50 decision threshold. It is not a measure of clinical certainty."
    )

    st.progress(confidence_indicator)

    st.write(
        f"Distance-from-threshold indicator: "
        f"{confidence_indicator:.0%}"
    )

    if 0.40 <= probability <= 0.60:
        st.warning(
            "Uncertain prediction: the probability is close to the "
            "0.50 decision threshold. Do not rely on this classification alone."
        )
    else:
        st.success(
            "The prediction is farther from the decision threshold. "
            "This does not guarantee that the prediction is correct."
        )

    # SHAP explanation
    st.subheader("Why did the model make this prediction? (SHAP)")

    try:
        explainer = make_diabetes_explainer()
        explanation = explainer(X)

        fig = plt.figure()
        shap.plots.waterfall(explanation[0], show=False)
        st.pyplot(plt.gcf())
        plt.close(fig)

    except Exception as error:
        st.warning(
            f"SHAP explanation could not be generated: {error}"
        )

    # Health suggestions
    st.subheader("Health and lifestyle suggestions")

    st.write("- Prefer whole grains, vegetables and fibre-rich foods.")
    st.write("- Choose water or unsweetened drinks instead of sugary drinks.")
    st.write(
        "- Aim for regular physical activity, as appropriate for your health."
    )
    st.write(
        "- Discuss diabetes screening with a qualified healthcare professional."
    )

    if probability >= 0.5:
        st.warning(
            "Consider discussing blood glucose testing and HbA1c testing "
            "with a healthcare professional."
        )


# --------------------------------------------------
# UTI input form
# --------------------------------------------------

def uti_input_form():
    st.header("Hospital-associated UTI risk assessment")
    st.warning(
        "For hospital-staff demonstration only. "
        "This model was trained on Danish hospital admission records "
        "from 2017–2018. It is not a general community UTI predictor."
    )

    with st.form("uti_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Patient age", 0, 106, 50)
            admission = st.selectbox("Admission type", ["Acute", "Planned"])
            readmission = st.selectbox(
                "Readmission", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            hospital = st.selectbox(
                "Hospital category",
                sorted(uti_data["admission_hospital_text"].unique())
            )
            department = st.selectbox(
                "Department category",
                sorted(
                    uti_data["admitted_org_id_text"].unique(),
                    key=lambda x: int(x.split()[-1])
                )
            )

        with col2:
            copd = st.selectbox(
                "History of COPD", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            retention = st.selectbox(
                "History of urinary retention", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            neurological = st.selectbox(
                "History of neurological disease", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            hai = st.number_input("Previous hospital-associated UTIs", 0, 100, 0)
            cai = st.number_input("Previous community-associated UTIs", 0, 100, 0)
            catheter = st.number_input(
                "Previous indwelling urinary catheter count", 0, 100, 0
            )

        submitted = st.form_submit_button(
            "Assess hospital-associated UTI risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        values = {
            "age": age,
            "ADMISSION_TYPE": admission,
            "readmission": readmission,
            "admission_hospital_text": hospital,
            "admitted_org_id_text": department,
            "ICD10_COPD": copd,
            "ICD10_urinary_retention": retention,
            "ICD10_neurological_disease": neurological,
            "previous_HAI_UTI": hai,
            "previous_CAI_UTI": cai,
            "Previous_IUC": catheter
        }
        return pd.DataFrame([values])

    return None


def show_uti_result(X):
    probability = float(uti_model.predict_proba(X)[0, 1])

    st.subheader("Hospital-associated UTI model result")
    col1, col2 = st.columns(2)

    col1.metric("Predicted positive-class probability", f"{probability:.1%}")
    col2.metric(
        "Model classification at threshold 0.50",
        "Positive" if probability >= 0.5 else "Negative"
    )

    st.caption(
        "This estimate relates to the dataset's hospital-associated "
        "UTI outcome. It is not a diagnosis or a validated prediction "
        "for a different hospital or population."
    )

    if 0.40 <= probability <= 0.60:
        st.warning(
            "This prediction is close to the classification threshold."
        )


def show_fairness():
    st.header("Diabetes model fairness evaluation")

    X = diabetes_data[DIABETES_FEATURES]
    y = diabetes_data["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    predictions = diabetes_model.predict(X_test)
    results = []

    for code, label in [(0, "Female"), (1, "Male")]:
        mask = X_test["Sex"] == code  # Internal CDC dataset feature name
        if mask.sum() == 0:
            continue

        results.append({
            "Group": label,
            "Records": int(mask.sum()),
            "Accuracy": accuracy_score(y_test[mask], predictions[mask]),
            "Precision": precision_score(
                y_test[mask], predictions[mask], zero_division=0
            ),
            "Recall": recall_score(
                y_test[mask], predictions[mask], zero_division=0
            )
        })

    st.dataframe(
        pd.DataFrame(results).set_index("Group"),
        use_container_width=True
    )
    st.caption(
        "These metrics use the same held-out split as the original "
        "diabetes model. Differences are descriptive, not proof of "
        "fairness or unfairness."
    )




# -------------------- Four-step user journey --------------------
page = st.session_state.page
gender = st.session_state.gender
assessment = st.session_state.assessment

if page == "gender":
    render_header(1, "A simple, guided assessment experience.")
    st.warning(DISCLAIMER)
    st.markdown("## Select your Gender")
    st.markdown(
        '<p class="muted">Choose an option to continue to the relevant assessment.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="choice-card"><div class="choice-icon">♂️</div>'
            '<div class="choice-title">Male</div>'
            '<div class="choice-copy">Diabetes assessment</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Continue as Male  →", key="gender_male", type="primary", use_container_width=True):
            reset_assessment()
            st.session_state.gender = "Male"
            st.session_state.assessment = "Diabetes"
            go_to("assessment")
            st.rerun()
    with right:
        st.markdown(
            '<div class="choice-card"><div class="choice-icon">♀️</div>'
            '<div class="choice-title">Female</div>'
            '<div class="choice-copy">Diabetes, hospital UTI, or both assessments</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Continue as Female  →", key="gender_female", type="primary", use_container_width=True):
            reset_assessment()
            st.session_state.gender = "Female"
            st.session_state.assessment = None
            go_to("assessment")
            st.rerun()

elif page == "assessment":
    render_header(2, "Choose the assessment you want to complete.")
    st.markdown("## Choose your Assessment")
    if gender == "Male":
        st.info("The diabetes assessment is available for male users.")
        if st.button("Diabetes Assessment  →", type="primary", use_container_width=True):
            st.session_state.assessment = "Diabetes"
            go_to("details")
            st.rerun()
    else:
        st.markdown(
            '<p class="muted">Select one option. You can review your result on the next page.</p>',
            unsafe_allow_html=True,
        )
        choices = [
            ("🩺", "Diabetes", "Diabetes and prediabetes indicator assessment"),
            ("🏥", "Hospital UTI", "Hospital-associated UTI assessment for female patients"),
            ("📋", "Both", "Complete both assessments in one session"),
        ]
        cols = st.columns(3, gap="medium")
        for col, (symbol, name, description) in zip(cols, choices):
            with col:
                st.markdown(
                    f'<div class="choice-card"><div class="choice-icon">{symbol}</div>'
                    f'<div class="choice-title">{name}</div>'
                    f'<div class="choice-copy">{description}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Select {name}", key=f"assessment_{name}", type="primary", use_container_width=True):
                    st.session_state.assessment = name
                    reset_assessment()
                    go_to("details")
                    st.rerun()

    st.divider()
    if st.button("← Back to Gender", use_container_width=False):
        go_to("gender")
        st.rerun()

elif page == "details":
    render_header(3, "Enter the requested information below.")
    st.markdown("## Assessment Details")
    st.caption(f"Gender: **{gender}**  ·  Assessment: **{assessment}**")
    sex_code = 0 if gender == "Female" else 1

    if assessment in ("Diabetes", "Both"):
        diabetes_inputs = diabetes_input_form(sex_code)
        if diabetes_inputs is not None:
            st.session_state.diabetes_result = diabetes_inputs
            if assessment == "Diabetes":
                go_to("results")
                st.rerun()
            else:
                st.success("Diabetes information saved. Complete the hospital UTI form below.")

    if assessment in ("Hospital UTI", "Both"):
        st.divider()
        uti_inputs = uti_input_form()
        if uti_inputs is not None:
            st.session_state.uti_result = uti_inputs
            if assessment == "Hospital UTI":
                go_to("results")
                st.rerun()
            else:
                st.success("Hospital UTI information saved.")

    if assessment == "Both":
        st.divider()
        diabetes_done = st.session_state.diabetes_result is not None
        uti_done = st.session_state.uti_result is not None
        st.write(
            f"Completion: Diabetes {'✓' if diabetes_done else 'not yet completed'} · "
            f"Hospital UTI {'✓' if uti_done else 'not yet completed'}"
        )
        if st.button(
            "View Both Results  →",
            type="primary",
            use_container_width=True,
            disabled=not (diabetes_done and uti_done),
        ):
            go_to("results")
            st.rerun()

    st.divider()
    back_col, _ = st.columns([1, 3])
    with back_col:
        if st.button("← Back", use_container_width=True):
            go_to("assessment")
            st.rerun()

elif page == "results":
    render_header(4, "Your assessment results and model explanations.")
    st.markdown("## Results")
    if assessment in ("Diabetes", "Both") and st.session_state.diabetes_result is not None:
        show_diabetes_result(st.session_state.diabetes_result)

    if assessment == "Both":
        st.divider()

    if assessment in ("Hospital UTI", "Both") and st.session_state.uti_result is not None:
        show_uti_result(st.session_state.uti_result)

    st.warning(DISCLAIMER)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Details", use_container_width=True):
            go_to("details")
            st.rerun()
    with col2:
        if st.button("Start a New Assessment", type="primary", use_container_width=True):
            reset_assessment()
            st.session_state.gender = None
            st.session_state.assessment = None
            go_to("gender")
            st.rerun()

    show_model_information()

render_footer()
