"""
CampusPulse AI - Student Placement Intelligence Platform
=========================================================
Luminous Crystal Spatial UI Web Application for Machine Learning Placement Prediction,
Multi-Model Consensus, Institutional EDA, and Interactive Performance Benchmarks.
"""

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SPATIAL UI THEME (VisionOS Luminous Glass)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CampusPulse AI | Spatial Placement Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
        background-image: 
            radial-gradient(circle at 14% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 86% 14%, rgba(6, 182, 212, 0.07) 0%, transparent 42%),
            radial-gradient(circle at 50% 88%, rgba(168, 85, 247, 0.06) 0%, transparent 50%),
            radial-gradient(rgba(148, 163, 184, 0.15) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 32px 32px;
        background-attachment: fixed;
        color: #0f172a;
    }
    
    /* Luminous Glass Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        padding: 2.2rem 2.6rem;
        border-radius: 1.8rem;
        box-shadow: 0 20px 48px -10px rgba(15, 23, 42, 0.08), inset 0 1px 1px rgba(255, 255, 255, 1);
        margin-bottom: 2rem;
        border: 1px solid rgba(226, 232, 240, 0.9);
        backdrop-filter: blur(32px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #4f46e5, #0284c7, #9333ea);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(79, 70, 229, 0.08);
        color: #4f46e5;
        border: 1px solid rgba(79, 70, 229, 0.2);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        color: #0f172a;
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 0;
        max-width: 850px;
    }
    
    /* Luminous Spatial Cards */
    .metric-card, .leader-card, .driver-card, .glossary-card {
        background: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 1.4rem;
        padding: 1.4rem;
        box-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.05);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        text-align: left;
    }
    .metric-card:hover, .leader-card:hover, .driver-card:hover, .glossary-card:hover {
        transform: translateY(-3px);
        border-color: rgba(79, 70, 229, 0.3);
        box-shadow: 0 16px 36px -6px rgba(15, 23, 42, 0.09);
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 900;
        color: #4f46e5;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        margin-top: 4px;
        letter-spacing: 0.02em;
    }

    /* Outcome Badges */
    .status-badge-placed {
        background: linear-gradient(135deg, rgba(236, 253, 245, 0.95) 0%, rgba(209, 250, 229, 0.95) 100%);
        color: #065f46;
        padding: 1.3rem 2rem;
        border-radius: 1.4rem;
        font-size: 2.1rem;
        font-weight: 900;
        text-align: center;
        border: 1.5px solid rgba(5, 150, 105, 0.4);
        box-shadow: 0 16px 36px -10px rgba(5, 150, 105, 0.15);
        letter-spacing: 0.04em;
    }
    .status-badge-not-placed {
        background: linear-gradient(135deg, rgba(255, 241, 242, 0.95) 0%, rgba(254, 226, 226, 0.95) 100%);
        color: #9f1239;
        padding: 1.3rem 2rem;
        border-radius: 1.4rem;
        font-size: 2.1rem;
        font-weight: 900;
        text-align: center;
        border: 1.5px solid rgba(225, 29, 72, 0.4);
        box-shadow: 0 16px 36px -10px rgba(225, 29, 72, 0.15);
        letter-spacing: 0.04em;
    }
    
    /* Recommendations Box */
    .rec-box {
        background: #ffffff;
        border-left: 4px solid #4f46e5;
        border-right: 1px solid rgba(226, 232, 240, 0.9);
        border-top: 1px solid rgba(226, 232, 240, 0.9);
        border-bottom: 1px solid rgba(226, 232, 240, 0.9);
        padding: 1.2rem;
        border-radius: 0 1.2rem 1.2rem 0;
        margin-top: 0.8rem;
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.6;
        box-shadow: 0 6px 18px -4px rgba(15, 23, 42, 0.05);
    }

    /* Floating Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(226, 232, 240, 0.9);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: rgba(241, 245, 249, 0.7);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 12px 12px 0px 0px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ffffff;
        color: #0f172a;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #4f46e5 !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.12) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODEL & DATA LOADING WITH MULTI-PATH RESOLUTION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_scalers():
    """Locate and load trained ML models and feature scalers."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "CampusPredict-Integrated" / "backend" / "ml_models",
        base_dir / "backend" / "ml_models",
        base_dir / "ml_models",
        Path("CampusPredict-Integrated/backend/ml_models"),
        Path("backend/ml_models"),
    ]
    model_dir = None
    for candidate in candidates:
        if candidate.exists() and (candidate / "logistic_regression.pkl").exists():
            model_dir = candidate
            break

    if not model_dir:
        st.error("❌ Could not locate serialized models directory (backend/ml_models).")
        st.stop()

    model_files = {
        "logistic_regression": ("logistic_regression.pkl", "scaler.pkl"),
        "knn": ("knn.pkl", "scaler_knn.pkl"),
        "decision_tree": ("decision_tree.pkl", None),
        "svm_linear": ("svm_linear.pkl", "scaler_svm_linear.pkl"),
    }

    loaded_models = {}
    loaded_scalers = {}

    for name, (m_file, s_file) in model_files.items():
        loaded_models[name] = joblib.load(model_dir / m_file)
        if s_file:
            loaded_scalers[name] = joblib.load(model_dir / s_file)

    f_names = list(loaded_scalers["logistic_regression"].feature_names_in_)
    return loaded_models, loaded_scalers, f_names


@st.cache_data
def load_dataset():
    """Locate and load student placement dataset for EDA."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "CampusPredict-Integrated" / "ml_source" / "student_placement_prediction_dataset_2026.csv",
        base_dir / "CampusPredict-Integrated" / "student_placement_prediction_dataset_2026.csv",
        base_dir / "ml_source" / "student_placement_prediction_dataset_2026.csv",
        Path("CampusPredict-Integrated/ml_source/student_placement_prediction_dataset_2026.csv"),
        Path("student_placement_prediction_dataset_2026.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


models, scalers, feature_names = load_models_and_scalers()
raw_df = load_dataset()


# -----------------------------------------------------------------------------
# 3. PREPROCESSING & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
def preprocess_input(payload: dict) -> pd.DataFrame:
    """Transform raw student dictionary into 33-feature engineered format."""
    frame = pd.DataFrame([payload])

    # Domain feature engineering
    frame['academic_clearance_score'] = frame['cgpa'] * np.exp(-0.30 * frame['backlogs'])
    frame['has_zero_backlogs'] = (frame['backlogs'] == 0).astype(int)
    frame['technical_mastery'] = (
        0.40 * frame['coding_skill_score'] +
        0.25 * frame['aptitude_score'] +
        0.20 * frame['logical_reasoning_score'] +
        0.15 * frame['mock_interview_score']
    )
    frame['experience_index'] = (
        frame['internships_count'] * 3.5 +
        frame['projects_count'] * 1.5 +
        frame['hackathons_participated'] * 1.5 +
        frame['certifications_count'] * 0.8
    )
    frame['placement_readiness_score'] = (
        0.35 * (frame['academic_clearance_score'] * 10) +
        0.35 * frame['technical_mastery'] +
        0.20 * np.clip(frame['experience_index'] * 5, 0, 100) +
        0.10 * frame['attendance_percentage']
    )

    categorical = ["gender", "branch", "college_tier", "volunteer_experience"]
    frame = pd.get_dummies(frame, columns=categorical, drop_first=True)
    frame = frame.reindex(columns=feature_names, fill_value=0)
    return frame


def predict_student(model_name: str, payload: dict):
    """Run model inference for a single student profile."""
    X = preprocess_input(payload)
    model = models[model_name]

    if model_name == "decision_tree":
        transformed = X
    else:
        transformed = scalers[model_name].transform(X)

    pred = int(model.predict(transformed)[0])
    label = "Placed" if pred == 1 else "Not Placed"

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(transformed)[0, 1]) * 100.0
    else:
        prob = 100.0 if pred == 1 else 0.0

    prob = max(0.0, min(100.0, prob))
    return label, prob, X.iloc[0].to_dict()


# Session history initialization
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & PRESETS
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
    <div style="width:42px; height:42px; border-radius:12px; background:linear-gradient(135deg,#4f46e5,#0284c7); display:grid; place-items:center; color:#fff; font-size:20px;">🎓</div>
    <div>
        <strong style="font-size:1.15rem; color:#0f172a; display:block;">CampusPulse</strong>
        <span style="font-size:0.75rem; color:#64748b; font-weight:600;">Spatial Intelligence v2.4</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Active Classifier")

selected_model_key = st.sidebar.selectbox(
    "Select Model for Single Prediction:",
    options=["logistic_regression", "knn", "decision_tree", "svm_linear", "all_models"],
    format_func=lambda x: {
        "logistic_regression": "Logistic Regression (94.05% Acc)",
        "knn": "K-Nearest Neighbors (90.38% Acc)",
        "decision_tree": "Decision Tree (91.36% Acc)",
        "svm_linear": "Linear SVM (94.04% Acc)",
        "all_models": "⚡ Compare All 4 Models (Consensus)"
    }[x],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Candidate Presets")

preset_choice = st.sidebar.radio(
    "Autofill Candidate Profile:",
    options=["Custom Input", "Top Scholar", "Average Student", "At-Risk Candidate"],
    index=0
)

if preset_choice == "Top Scholar":
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 8.9, 0, 88.0, 85.0, 90.0, 85.0, 82.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 2, 4, 3, 3, 12, 350
    p_attend, p_extra, p_lead, p_sleep, p_study = 94.0, 80.0, 85.0, 7.5, 6.0
elif preset_choice == "Average Student":
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 7.4, 0, 68.0, 68.0, 70.0, 65.0, 70.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 1, 2, 1, 1, 4, 120
    p_attend, p_extra, p_lead, p_sleep, p_study = 84.0, 55.0, 50.0, 7.0, 4.0
elif preset_choice == "At-Risk Candidate":
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 5.6, 2, 38.0, 42.0, 40.0, 40.0, 45.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 0, 1, 0, 0, 1, 40
    p_attend, p_extra, p_lead, p_sleep, p_study = 65.0, 25.0, 25.0, 5.5, 1.5
else:
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 8.2, 0, 82.0, 76.0, 80.0, 75.0, 78.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 1, 3, 2, 2, 8, 320
    p_attend, p_extra, p_lead, p_sleep, p_study = 88.0, 70.0, 68.0, 7.0, 4.5


# -----------------------------------------------------------------------------
# 5. HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🎓 Spatial Placement Intelligence • 100k Benchmark</div>
    <div class="hero-title">CampusPulse Intelligence Platform</div>
    <div class="hero-subtitle">
        Predict student placement probabilities with calibrated Machine Learning models, domain-engineered metrics, real-time multi-model consensus, and institutional analytics trained across 100,000 alumni records.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. APPLICATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Placement Predictor",
    "⚖️ Multi-Model Consensus",
    "📊 Institutional EDA",
    "🧠 Model Performance",
    "📜 Session History"
])


# =============================================================================
# TAB 1: SINGLE PREDICTION
# =============================================================================
with tab1:
    st.subheader("📋 Enter Candidate Profile Attributes")
    
    with st.form(key="student_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📚 Academic Standing")
            cgpa = st.number_input("CGPA (out of 10)", min_value=4.5, max_value=10.0, value=float(p_cgpa), step=0.1)
            backlogs = st.number_input("Active Backlogs Count", min_value=0, max_value=6, value=int(p_backlogs), step=1)
            attendance_percentage = st.slider("Attendance Percentage (%)", min_value=50.0, max_value=100.0, value=float(p_attend), step=1.0)
            branch = st.selectbox("Academic Branch", options=["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"], index=0)
            college_tier = st.selectbox("College Tier", options=["Tier 1", "Tier 2", "Tier 3"], index=0)
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)

        with col2:
            st.markdown("#### 💻 Technical Competency")
            coding_skill_score = st.slider("Coding Assessment Score", min_value=20.0, max_value=100.0, value=float(p_coding), step=1.0)
            aptitude_score = st.slider("General Aptitude Score", min_value=20.0, max_value=100.0, value=float(p_apt), step=1.0)
            logical_reasoning_score = st.slider("Logical Reasoning Score", min_value=20.0, max_value=100.0, value=float(p_log), step=1.0)
            communication_skill_score = st.slider("Communication Score", min_value=20.0, max_value=100.0, value=float(p_comm), step=1.0)
            mock_interview_score = st.slider("Mock Interview Score", min_value=20.0, max_value=100.0, value=float(p_mock), step=1.0)
            github_repos = st.number_input("GitHub Repositories Count", min_value=0, max_value=100, value=int(p_git), step=1)

        with col3:
            st.markdown("#### 💼 Experience & Habits")
            internships_count = st.number_input("Completed Internships", min_value=0, max_value=10, value=int(p_intern), step=1)
            projects_count = st.number_input("Full-Stack Projects", min_value=0, max_value=20, value=int(p_proj), step=1)
            hackathons_participated = st.number_input("Hackathons Participated", min_value=0, max_value=20, value=int(p_hack), step=1)
            certifications_count = st.number_input("Technical Certifications", min_value=0, max_value=20, value=int(p_cert), step=1)
            linkedin_connections = st.number_input("LinkedIn Connections", min_value=0, max_value=5000, value=int(p_link), step=10)
            extracurricular_score = st.slider("Extracurricular Score", min_value=0.0, max_value=100.0, value=float(p_extra), step=1.0)
            leadership_score = st.slider("Leadership Score", min_value=0.0, max_value=100.0, value=float(p_lead), step=1.0)
            volunteer_experience = st.selectbox("Volunteer Experience", options=["Yes", "No"], index=0)
            study_hours_per_day = st.slider("Daily Study Hours", min_value=0.5, max_value=10.0, value=float(p_study), step=0.5)
            sleep_hours = st.slider("Average Sleep Hours", min_value=4.0, max_value=10.0, value=float(p_sleep), step=0.5)
            age = st.number_input("Age", min_value=18, max_value=28, value=21, step=1)

        submit_btn = st.form_submit_button("🔮 Predict Placement Likelihood", use_container_width=True)

    if submit_btn:
        payload = {
            "age": int(age), "gender": gender, "cgpa": float(cgpa), "branch": branch,
            "college_tier": college_tier, "internships_count": int(internships_count),
            "projects_count": int(projects_count), "certifications_count": int(certifications_count),
            "coding_skill_score": float(coding_skill_score), "aptitude_score": float(aptitude_score),
            "communication_skill_score": float(communication_skill_score),
            "logical_reasoning_score": float(logical_reasoning_score),
            "hackathons_participated": int(hackathons_participated),
            "github_repos": int(github_repos), "linkedin_connections": int(linkedin_connections),
            "mock_interview_score": float(mock_interview_score),
            "attendance_percentage": float(attendance_percentage),
            "backlogs": int(backlogs), "extracurricular_score": float(extracurricular_score),
            "leadership_score": float(leadership_score), "volunteer_experience": volunteer_experience,
            "sleep_hours": float(sleep_hours), "study_hours_per_day": float(study_hours_per_day)
        }

        eval_model = "logistic_regression" if selected_model_key == "all_models" else selected_model_key
        label, prob, X_processed = predict_student(eval_model, payload)

        st.session_state.history.append({
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": eval_model,
            "Prediction": label,
            "Probability (%)": round(prob, 2),
            "CGPA": cgpa,
            "Backlogs": backlogs,
            "Coding Score": coding_skill_score,
            "Branch": branch
        })

        st.markdown("---")
        st.subheader("🎯 Prediction Result & Spatial Analytics")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if label == "Placed":
                st.markdown('<div class="status-badge-placed">✅ PLACED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge-not-placed">❌ NOT PLACED</div>', unsafe_allow_html=True)

            # Luminous Probability Gauge with clean dark slate text
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={'suffix': '%', 'font': {'size': 38, 'color': '#0f172a', 'family': 'Outfit'}},
                title={'text': "Estimated Placement Probability", 'font': {'size': 16, 'color': '#475569', 'family': 'Outfit'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'color': '#475569'}},
                    'bar': {'color': "#4f46e5" if prob >= 50 else "#e11d48"},
                    'bgcolor': "#ffffff",
                    'borderwidth': 1,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(225, 29, 72, 0.08)'},
                        {'range': [50, 75], 'color': 'rgba(2, 132, 199, 0.08)'},
                        {'range': [75, 100], 'color': 'rgba(5, 150, 105, 0.1)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### 📈 Domain-Engineered Indicator Scores")
            readiness = X_processed.get('placement_readiness_score', 0.0)
            tech_mastery = X_processed.get('technical_mastery', 0.0)
            exp_index = X_processed.get('experience_index', 0.0)
            acad_clear = X_processed.get('academic_clearance_score', 0.0)

            ind_col1, ind_col2 = st.columns(2)
            with ind_col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{readiness:.1f}/100</div><div class="metric-label">Composite Readiness</div></div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="metric-card"><div class="metric-value">{acad_clear:.2f}</div><div class="metric-label">Academic Clearance Score</div></div>', unsafe_allow_html=True)
            with ind_col2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{tech_mastery:.1f}/100</div><div class="metric-label">Technical Mastery</div></div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="metric-card"><div class="metric-value">{exp_index:.1f}</div><div class="metric-label">Experience Index</div></div>', unsafe_allow_html=True)

        # Strategic Action Items
        st.markdown("#### 💡 Strategic Recommendations for Upskilling")
        recs = []
        if backlogs > 0:
            recs.append(f"⚠️ **Clear Active Backlogs**: You currently have **{backlogs} active backlogs**. Enterprise campus drives enforce a 0-backlog cutoff. Clearing them immediately raises eligibility.")
        if coding_skill_score < 75:
            recs.append("💻 **Enhance Coding Competency**: Target a coding assessment score above 75+ with competitive programming and Data Structures practice.")
        if internships_count == 0:
            recs.append("💼 **Secure Practical Internship Experience**: Completing 1-2 industrial internships adds +3.5 weighted points per internship to your placement profile.")
        if mock_interview_score < 75:
            recs.append("🗣️ **Conduct Mock Technical Interviews**: Improve mock interview feedback score to 75+ to boost interview calibration.")
        if not recs:
            recs.append("🌟 **Outstanding Profile**: This candidate displays top-tier technical readiness, excellent academic standing, and competitive practical experience!")

        for r in recs:
            st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: MULTI-MODEL CONSENSUS
# =============================================================================
with tab2:
    st.subheader("⚖️ Multi-Model Consensus & Comparison")
    st.write("Simultaneously evaluate student qualifications across all 4 machine learning models.")

    c_col1, c_col2 = st.columns([1, 1])

    payload_comp = {
        "age": 21, "gender": "Male", "cgpa": float(p_cgpa), "branch": "CSE", "college_tier": "Tier 1",
        "internships_count": int(p_intern), "projects_count": int(p_proj), "certifications_count": int(p_cert),
        "coding_skill_score": float(p_coding), "aptitude_score": float(p_apt), "communication_skill_score": float(p_comm),
        "logical_reasoning_score": float(p_log), "hackathons_participated": int(p_hack), "github_repos": int(p_git),
        "linkedin_connections": int(p_link), "mock_interview_score": float(p_mock), "attendance_percentage": float(p_attend),
        "backlogs": int(p_backlogs), "extracurricular_score": float(p_extra), "leadership_score": float(p_lead),
        "volunteer_experience": "Yes", "sleep_hours": float(p_sleep), "study_hours_per_day": float(p_study)
    }

    comp_results = []
    for m_name in ["logistic_regression", "knn", "decision_tree", "svm_linear"]:
        lbl, pr, _ = predict_student(m_name, payload_comp)
        comp_results.append({
            "Model Key": m_name,
            "Model Name": {
                "logistic_regression": "Logistic Regression",
                "knn": "K-Nearest Neighbors",
                "decision_tree": "Decision Tree",
                "svm_linear": "Linear SVM"
            }[m_name],
            "Prediction": lbl,
            "Probability (%)": round(pr, 2)
        })

    comp_df = pd.DataFrame(comp_results)
    placed_votes = sum(comp_df["Prediction"] == "Placed")
    majority_label = "Placed" if placed_votes >= 2 else "Not Placed"

    with c_col1:
        st.markdown(f"### Consensus Verdict: **{majority_label.upper()}** ({placed_votes}/4 Models Agree)")
        st.dataframe(comp_df[["Model Name", "Prediction", "Probability (%)"]], use_container_width=True)

    with c_col2:
        fig_bar = px.bar(
            comp_df,
            x="Model Name",
            y="Probability (%)",
            color="Prediction",
            color_discrete_map={"Placed": "#059669", "Not Placed": "#e11d48"},
            text="Probability (%)",
            title="Estimated Probability Spectrum Across Classifiers"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(255,255,255,0.7)",
            plot_bgcolor="rgba(248,250,252,0.5)",
            font_color="#0f172a",
            yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# TAB 3: INSTITUTIONAL EDA
# =============================================================================
with tab3:
    st.subheader("📊 Institutional Data Analytics & Distributions")

    if raw_df is not None:
        st.write(f"Analyzed Dataset: **{raw_df.shape[0]:,} Student Records** across 26 placement indicators.")

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            fig_box1 = px.box(
                raw_df.sample(min(5000, len(raw_df)), random_state=42),
                x="placement_status",
                y="cgpa",
                color="placement_status",
                title="CGPA Distribution by Placement Outcome",
                color_discrete_map={"Placed": "#059669", "Not Placed": "#e11d48"}
            )
            fig_box1.update_layout(
                paper_bgcolor="rgba(255,255,255,0.7)",
                plot_bgcolor="rgba(248,250,252,0.5)",
                font_color="#0f172a"
            )
            st.plotly_chart(fig_box1, use_container_width=True)

        with eda_col2:
            fig_scat = px.scatter(
                raw_df.sample(min(3000, len(raw_df)), random_state=42),
                x="coding_skill_score",
                y="aptitude_score",
                color="placement_status",
                title="Coding Skill vs Aptitude Assessment",
                color_discrete_map={"Placed": "#059669", "Not Placed": "#e11d48"},
                opacity=0.6
            )
            fig_scat.update_layout(
                paper_bgcolor="rgba(255,255,255,0.7)",
                plot_bgcolor="rgba(248,250,252,0.5)",
                font_color="#0f172a"
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("#### 🏢 Placement Outcomes across Academic Branches & Tiers")
        eda_col3, eda_col4 = st.columns(2)

        with eda_col3:
            branch_df = raw_df.groupby(["branch", "placement_status"]).size().reset_index(name="count")
            fig_branch = px.bar(
                branch_df,
                x="branch",
                y="count",
                color="placement_status",
                title="Placement Distribution by Engineering Branch",
                barmode="group",
                color_discrete_map={"Placed": "#059669", "Not Placed": "#e11d48"}
            )
            fig_branch.update_layout(
                paper_bgcolor="rgba(255,255,255,0.7)",
                plot_bgcolor="rgba(248,250,252,0.5)",
                font_color="#0f172a"
            )
            st.plotly_chart(fig_branch, use_container_width=True)

        with eda_col4:
            tier_df = raw_df.groupby(["college_tier", "placement_status"]).size().reset_index(name="count")
            fig_tier = px.bar(
                tier_df,
                x="college_tier",
                y="count",
                color="placement_status",
                title="Placement Outcomes by College Tier",
                barmode="group",
                color_discrete_map={"Placed": "#059669", "Not Placed": "#e11d48"}
            )
            fig_tier.update_layout(
                paper_bgcolor="rgba(255,255,255,0.7)",
                plot_bgcolor="rgba(248,250,252,0.5)",
                font_color="#0f172a"
            )
            st.plotly_chart(fig_tier, use_container_width=True)
    else:
        st.info("Dataset file (`student_placement_prediction_dataset_2026.csv`) is loading.")


# =============================================================================
# TAB 4: INTUITIVE MODEL BENCHMARKS & PERFORMANCE (100k DATASET)
# =============================================================================
with tab4:
    st.subheader("🧠 Machine Learning Performance & Benchmarks (100,000 Students)")
    st.write("Validation metrics evaluated on a stratified 80/20 test split (20,000 test instances out of 100,000 students).")

    # 1. Leader Trophy Cards
    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    with l_col1:
        st.markdown("""
        <div class="leader-card">
            <span style="font-size:10px; font-weight:700; color:#b45309; background:rgba(217,119,6,0.1); padding:4px 8px; border-radius:999px;">TOP ACCURACY</span>
            <div class="metric-value" style="color:#0f172a; margin-top:8px;">94.05%</div>
            <strong style="color:#334155; font-size:14px;">Logistic Regression</strong>
            <p style="color:#64748b; font-size:12px; margin:4px 0 0;">Leading accuracy across 100,000 students with calibrated margins.</p>
        </div>
        """, unsafe_allow_html=True)
    with l_col2:
        st.markdown("""
        <div class="leader-card">
            <span style="font-size:10px; font-weight:700; color:#065f46; background:rgba(5,150,105,0.1); padding:4px 8px; border-radius:999px;">TOP RECALL</span>
            <div class="metric-value" style="color:#0f172a; margin-top:8px;">94.65%</div>
            <strong style="color:#334155; font-size:14px;">Logistic Regression</strong>
            <p style="color:#64748b; font-size:12px; margin:4px 0 0;">Catches 95 of 100 placed candidates with minimal false negative loss.</p>
        </div>
        """, unsafe_allow_html=True)
    with l_col3:
        st.markdown("""
        <div class="leader-card">
            <span style="font-size:10px; font-weight:700; color:#6d28d9; background:rgba(124,58,237,0.1); padding:4px 8px; border-radius:999px;">TOP ROC-AUC</span>
            <div class="metric-value" style="color:#0f172a; margin-top:8px;">98.95%</div>
            <strong style="color:#334155; font-size:14px;">Linear SVM & Logistic</strong>
            <p style="color:#64748b; font-size:12px; margin:4px 0 0;">Near-perfect distinction between placed and unplaced profiles.</p>
        </div>
        """, unsafe_allow_html=True)
    with l_col4:
        st.markdown("""
        <div class="leader-card">
            <span style="font-size:10px; font-weight:700; color:#0369a1; background:rgba(2,132,199,0.1); padding:4px 8px; border-radius:999px;">AUDITABLE RULES</span>
            <div class="metric-value" style="color:#0f172a; margin-top:8px;">91.36%</div>
            <strong style="color:#334155; font-size:14px;">Decision Tree</strong>
            <p style="color:#64748b; font-size:12px; margin:4px 0 0;">Transparent hierarchical rule splits with 92.14% balanced F1.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Load dynamic metrics
    metrics_file = Path(__file__).resolve().parent / "CampusPredict-Integrated" / "model_metrics.csv"
    if not metrics_file.exists():
        metrics_file = Path(__file__).resolve().parent / "model_metrics.csv"

    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file)
        if "Accuracy" in metrics_df.columns and "Accuracy (%)" not in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={
                "Accuracy": "Accuracy (%)",
                "Precision": "Precision (%)",
                "Recall": "Recall (%)",
                "F1": "F1 Score (%)",
                "ROC-AUC": "ROC-AUC (%)"
            })
    else:
        metrics_data = [
            {"Model": "Logistic Regression", "Accuracy (%)": 94.05, "Precision (%)": 94.53, "Recall (%)": 94.65, "F1 Score (%)": 94.59, "ROC-AUC (%)": 98.95},
            {"Model": "Linear SVM", "Accuracy (%)": 94.04, "Precision (%)": 94.54, "Recall (%)": 94.63, "F1 Score (%)": 94.58, "ROC-AUC (%)": 98.95},
            {"Model": "Decision Tree", "Accuracy (%)": 91.36, "Precision (%)": 92.15, "Recall (%)": 92.13, "F1 Score (%)": 92.14, "ROC-AUC (%)": 97.49},
            {"Model": "KNN", "Accuracy (%)": 90.38, "Precision (%)": 90.25, "Recall (%)": 92.50, "F1 Score (%)": 91.36, "ROC-AUC (%)": 97.11}
        ]
        metrics_df = pd.DataFrame(metrics_data)

    # 2. Interactive Comparison Chart with Metric Selector
    st.markdown("#### 📊 Interactive Metric Visualizer")
    metric_choice = st.radio(
        "Select Performance Metric to Compare:",
        options=["Accuracy (%)", "Recall (%)", "Precision (%)", "F1 Score (%)", "ROC-AUC (%)"],
        horizontal=True
    )

    fig_metrics = px.bar(
        metrics_df,
        x="Model",
        y=metric_choice,
        color="Model",
        text=metric_choice,
        title=f"100,000-Record Benchmark: {metric_choice} across Models",
        color_discrete_sequence=["#4f46e5", "#0284c7", "#059669", "#7c3aed"]
    )
    fig_metrics.update_layout(
        paper_bgcolor="rgba(255,255,255,0.7)",
        plot_bgcolor="rgba(248,250,252,0.5)",
        font_color="#0f172a",
        yaxis=dict(range=[80, 102])
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # 3. Domain Feature Drivers
    st.markdown("#### 🎯 Domain Feature Drivers & Relative Weights")
    fd_col1, fd_col2 = st.columns(2)
    with fd_col1:
        st.markdown("""
        <div class="driver-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>Technical Competency & Projects</strong>
                <span style="font-weight:700; color:#4f46e5; background:rgba(79,70,229,0.08); padding:3px 8px; border-radius:999px;">35% Weight</span>
            </div>
            <p style="color:#64748b; font-size:12px; margin-top:8px;">Coding skill assessments, aptitude testing, GitHub repositories, and full-stack projects provide the highest predictive signal.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        <div class="driver-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>Academic Clearance & Baseline</strong>
                <span style="font-weight:700; color:#0284c7; background:rgba(2,132,199,0.08); padding:3px 8px; border-radius:999px;">35% Weight</span>
            </div>
            <p style="color:#64748b; font-size:12px; margin-top:8px;">CGPA thresholds, active backlog clearance status, and regular attendance percentage act as primary qualifying gates.</p>
        </div>
        """, unsafe_allow_html=True)
    with fd_col2:
        st.markdown("""
        <div class="driver-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>Practical Experience & Hackathons</strong>
                <span style="font-weight:700; color:#059669; background:rgba(5,150,105,0.08); padding:3px 8px; border-radius:999px;">20% Weight</span>
            </div>
            <p style="color:#64748b; font-size:12px; margin-top:8px;">Prior industry internships and competitive hackathons differentiate candidates at the interview stage.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        <div class="driver-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>Interview Readiness & Discipline</strong>
                <span style="font-weight:700; color:#7c3aed; background:rgba(124,58,237,0.08); padding:3px 8px; border-radius:999px;">10% Weight</span>
            </div>
            <p style="color:#64748b; font-size:12px; margin-top:8px;">Mock interview feedback scores, professional communication, and consistent study habits provide fine-grained calibration.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 4. Plain-English Metric Glossary
    st.markdown("#### 📖 Plain-English Metric Glossary")
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        st.markdown("""
        <div class="glossary-card">
            <strong>Accuracy</strong>
            <p style="color:#475569; font-size:12px; margin:6px 0;">Total fraction of predictions that match true outcomes.</p>
            <small style="color:#059669; font-weight:700;">Logistic leads at 94.05%</small>
        </div>
        """, unsafe_allow_html=True)
    with g_col2:
        st.markdown("""
        <div class="glossary-card">
            <strong>Recall</strong>
            <p style="color:#475569; font-size:12px; margin:6px 0;">Out of all placed students, what fraction the AI identified.</p>
            <small style="color:#059669; font-weight:700;">Logistic leads at 94.65%</small>
        </div>
        """, unsafe_allow_html=True)
    with g_col3:
        st.markdown("""
        <div class="glossary-card">
            <strong>Precision</strong>
            <p style="color:#475569; font-size:12px; margin:6px 0;">When model says "Placed", how often is it truly correct.</p>
            <small style="color:#059669; font-weight:700;">Linear SVM leads at 94.54%</small>
        </div>
        """, unsafe_allow_html=True)
    with g_col4:
        st.markdown("""
        <div class="glossary-card">
            <strong>ROC-AUC</strong>
            <p style="color:#475569; font-size:12px; margin:6px 0;">Separation ability across all confidence levels (100% is flawless).</p>
            <small style="color:#059669; font-weight:700;">SVM & Logistic lead at 98.95%</small>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 📋 Complete Evaluation Data Table")
    st.dataframe(metrics_df, use_container_width=True)


# =============================================================================
# TAB 5: HISTORY & EXPORT
# =============================================================================
with tab5:
    st.subheader("📜 Session Prediction Log")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv_data = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Session History as CSV",
            data=csv_data,
            file_name="campuspulse_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No predictions recorded in this session yet. Run a prediction in Tab 1 to populate history.")
