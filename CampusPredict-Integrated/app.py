"""
CampusPredict - Student Placement Prediction Platform
======================================================
Modern Academic SaaS Theme Web Application for Machine Learning Placement Prediction,
Multi-Model Comparison, Exploratory Data Analysis, and Performance Metrics.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MODERN ACADEMIC SAAS THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CampusPredict AI | Academic Placement SaaS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Academic SaaS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #070a12;
        background-image: 
            radial-gradient(circle at 85% 10%, rgba(99, 102, 241, 0.14), transparent 45%),
            radial-gradient(circle at 15% 85%, rgba(14, 165, 233, 0.10), transparent 40%);
        color: #f8fafc;
    }
    
    /* Academic Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(31, 41, 61, 0.9) 0%, rgba(17, 24, 39, 0.95) 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 1.5rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366f1, #0ea5e9, #10b981);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 0;
        max-width: 850px;
    }
    
    /* SaaS Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.9), rgba(11, 15, 25, 0.95));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 1.25rem;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Prediction Outcome Badges */
    .status-badge-placed {
        background: linear-gradient(135deg, rgba(6, 95, 70, 0.8) 0%, rgba(4, 120, 87, 0.9) 100%);
        color: #ecfdf5;
        padding: 1.2rem 1.8rem;
        border-radius: 1.25rem;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        border: 1px solid #10b981;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
        letter-spacing: 0.05em;
    }
    .status-badge-not-placed {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.8) 0%, rgba(185, 28, 28, 0.9) 100%);
        color: #fef2f2;
        padding: 1.2rem 1.8rem;
        border-radius: 1.25rem;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        border: 1px solid #f43f5e;
        box-shadow: 0 0 25px rgba(244, 63, 94, 0.35);
        letter-spacing: 0.05em;
    }
    
    /* Recommendations Container */
    .rec-box {
        background: rgba(17, 24, 39, 0.85);
        border-left: 4px solid #6366f1;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.1rem;
        border-radius: 0 0.8rem 0.8rem 0;
        margin-top: 0.8rem;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px 12px 0px 0px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(67, 56, 202, 0.4)) !important;
        color: #ffffff !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODEL & DATA LOADING WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_scalers():
    """Locate and load trained ML models and feature scalers."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "backend" / "ml_models",
        base_dir / "CampusPredict-Integrated" / "backend" / "ml_models",
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

    models = {}
    scalers = {}

    for name, (m_file, s_file) in model_files.items():
        models[name] = joblib.load(model_dir / m_file)
        if s_file:
            scalers[name] = joblib.load(model_dir / s_file)

    feature_names = list(scalers["logistic_regression"].feature_names_in_)
    return models, scalers, feature_names


@st.cache_data
def load_dataset():
    """Locate and load student placement dataset for EDA."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "ml_source" / "student_placement_prediction_dataset_2026.csv",
        base_dir / "CampusPredict-Integrated" / "ml_source" / "student_placement_prediction_dataset_2026.csv",
        base_dir / "student_placement_prediction_dataset_2026.csv",
        Path("CampusPredict-Integrated/ml_source/student_placement_prediction_dataset_2026.csv"),
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
    """Preprocess single student input dictionary into model feature dataframe."""
    frame = pd.DataFrame([payload])

    # Domain-specific feature engineering
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
    """Run model inference for a single student payload."""
    X = preprocess_input(payload)
    model = models[model_name]

    if model_name == "decision_tree":
        X_trans = X
    else:
        X_trans = scalers[model_name].transform(X)

    pred = int(model.predict(X_trans)[0])
    label = "Placed" if pred == 1 else "Not Placed"

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_trans)[0, 1]) * 100.0
    else:
        prob = 100.0 if pred == 1 else 0.0

    prob = max(0.0, min(100.0, prob))
    return label, prob, X.iloc[0]


# Initialize Session State for Prediction History
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & PRESETS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric-folders/100/graduation-cap.png", width=65)
st.sidebar.title("CampusPredict AI")
st.sidebar.caption("Modern Academic SaaS Platform")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Architecture")

selected_model_key = st.sidebar.selectbox(
    "Active ML Classifier",
    options=["logistic_regression", "knn", "decision_tree", "svm_linear", "all_models"],
    format_func=lambda x: {
        "logistic_regression": "Logistic Regression (StandardScaler)",
        "knn": "K-Nearest Neighbors (RobustScaler)",
        "decision_tree": "Decision Tree (Pruned & Tuned)",
        "svm_linear": "Linear SVM (Platt Calibrated)",
        "all_models": "⚡ Compare All 4 Models (Consensus)"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Profile Presets")

preset_choice = st.sidebar.radio(
    "Autofill Candidate Profile:",
    options=["Custom Input", "Top Scholar", "Average Student", "At-Risk Candidate"],
    index=0
)

# Preset values
if preset_choice == "Top Scholar":
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 8.9, 0, 88.0, 85.0, 90.0, 85.0, 82.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 2, 4, 3, 3, 12, 350
    p_attend, p_extra, p_lead, p_sleep, p_study = 94.0, 80.0, 85.0, 7.5, 6.0
elif preset_choice == "Average Student":
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 7.2, 0, 65.0, 68.0, 70.0, 65.0, 70.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 1, 2, 1, 1, 4, 120
    p_attend, p_extra, p_lead, p_sleep, p_study = 82.0, 50.0, 45.0, 7.0, 4.0
elif preset_choice == "At-Risk Candidate":
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 5.8, 2, 42.0, 48.0, 45.0, 40.0, 50.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 0, 1, 0, 0, 1, 45
    p_attend, p_extra, p_lead, p_sleep, p_study = 68.0, 30.0, 25.0, 5.5, 2.0
else:
    p_cgpa, p_backlogs, p_coding, p_apt, p_log, p_mock, p_comm = 7.8, 0, 75.0, 72.0, 76.0, 70.0, 74.0
    p_intern, p_proj, p_hack, p_cert, p_git, p_link = 1, 3, 2, 2, 6, 200
    p_attend, p_extra, p_lead, p_sleep, p_study = 86.0, 60.0, 60.0, 7.0, 4.5


# -----------------------------------------------------------------------------
# 5. HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🎓 Institutional Academic SaaS Portal • v2.4</div>
    <div class="hero-title">CampusPredict Intelligence Platform</div>
    <div class="hero-subtitle">
        Predict student placement probabilities with high-precision calibrated Machine Learning models, domain-engineered metrics, real-time multi-model benchmarks, and institutional analytics.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Placement Predictor",
    "⚖️ Multi-Model Consensus",
    "📊 Institutional EDA",
    "🧠 Model Benchmarks",
    "📜 History & CSV Export"
])


# =============================================================================
# TAB 1: SINGLE PREDICTION
# =============================================================================
with tab1:
    st.subheader("📋 Enter Student Profile Attributes")
    
    with st.form(key="student_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📚 Academic Performance")
            cgpa = st.number_input("CGPA (out of 10)", min_value=4.5, max_value=10.0, value=float(p_cgpa), step=0.1)
            backlogs = st.number_input("Active Backlogs Count", min_value=0, max_value=6, value=int(p_backlogs), step=1)
            attendance_percentage = st.slider("Attendance Percentage (%)", min_value=50.0, max_value=100.0, value=float(p_attend), step=1.0)
            branch = st.selectbox("Academic Branch", options=["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"], index=0)
            college_tier = st.selectbox("College Tier", options=["Tier 1", "Tier 2", "Tier 3"], index=1)
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
            age = st.number_input("Age", min_value=18, max_value=24, value=21, step=1)

        with col2:
            st.markdown("#### 💻 Technical & Aptitude Competency")
            coding_skill_score = st.slider("Coding Skill Score", 20.0, 100.0, float(p_coding), 1.0)
            aptitude_score = st.slider("Aptitude Assessment Score", 20.0, 100.0, float(p_apt), 1.0)
            logical_reasoning_score = st.slider("Logical Reasoning Score", 20.0, 100.0, float(p_log), 1.0)
            mock_interview_score = st.slider("Mock Interview Score", 20.0, 100.0, float(p_mock), 1.0)
            communication_skill_score = st.slider("Communication Skill Score", 20.0, 100.0, float(p_comm), 1.0)

        with col3:
            st.markdown("#### 🛠️ Experience & Soft Skills")
            internships_count = st.number_input("Internships Count", 0, 5, int(p_intern))
            projects_count = st.number_input("Projects Completed", 0, 10, int(p_proj))
            hackathons_participated = st.number_input("Hackathons Participated", 0, 10, int(p_hack))
            certifications_count = st.number_input("Certifications Count", 0, 10, int(p_cert))
            github_repos = st.number_input("GitHub Repositories", 0, 50, int(p_git))
            linkedin_connections = st.number_input("LinkedIn Connections", 0, 500, int(p_link))
            extracurricular_score = st.slider("Extracurricular Score", 0.0, 100.0, float(p_extra))
            leadership_score = st.slider("Leadership Score", 0.0, 100.0, float(p_lead))
            volunteer_experience = st.selectbox("Volunteer Experience", options=["Yes", "No"], index=0)
            sleep_hours = st.slider("Daily Sleep Hours", 3.0, 10.0, float(p_sleep), 0.5)
            study_hours_per_day = st.slider("Daily Study Hours", 0.5, 10.0, float(p_study), 0.5)

        submit_button = st.form_submit_button(label="🚀 Execute Placement Prediction", use_container_width=True)

    if submit_button:
        payload = {
            "age": int(age),
            "gender": gender,
            "cgpa": float(cgpa),
            "branch": branch,
            "college_tier": college_tier,
            "internships_count": int(internships_count),
            "projects_count": int(projects_count),
            "certifications_count": int(certifications_count),
            "coding_skill_score": float(coding_skill_score),
            "aptitude_score": float(aptitude_score),
            "communication_skill_score": float(communication_skill_score),
            "logical_reasoning_score": float(logical_reasoning_score),
            "hackathons_participated": int(hackathons_participated),
            "github_repos": int(github_repos),
            "linkedin_connections": int(linkedin_connections),
            "mock_interview_score": float(mock_interview_score),
            "attendance_percentage": float(attendance_percentage),
            "backlogs": int(backlogs),
            "extracurricular_score": float(extracurricular_score),
            "leadership_score": float(leadership_score),
            "volunteer_experience": volunteer_experience,
            "sleep_hours": float(sleep_hours),
            "study_hours_per_day": float(study_hours_per_day)
        }

        eval_model = "logistic_regression" if selected_model_key == "all_models" else selected_model_key
        label, prob, X_processed = predict_student(eval_model, payload)

        # Log to history
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
        st.subheader("🎯 Prediction Result & Analytics")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if label == "Placed":
                st.markdown(f'<div class="status-badge-placed">✅ PLACED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge-not-placed">❌ NOT PLACED</div>', unsafe_allow_html=True)

            st.write("")
            # Gauge chart for probability
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={'suffix': '%', 'font': {'size': 36, 'color': '#ffffff'}},
                title={'text': "Estimated Placement Probability", 'font': {'size': 18, 'color': '#c7d2fe'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                    'bar': {'color': "#6366f1" if prob >= 50 else "#f43f5e"},
                    'bgcolor': "#111827",
                    'borderwidth': 2,
                    'bordercolor': "#374151",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(244, 63, 94, 0.2)'},
                        {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ],
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### 📈 Engineered Indicator Scores")
            
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

        # Actionable Insights & Recommendations
        st.markdown("#### 💡 Actionable Insights & Strategic Recommendations")
        recs = []
        if backlogs > 0:
            recs.append(f"⚠️ **Clear Active Backlogs**: You currently have **{backlogs} backlogs**. Clearing active backlogs significantly improves academic clearance score and eligibility.")
        if coding_skill_score < 70:
            recs.append("💻 **Enhance Coding Skills**: Target a coding assessment score above 70+ by practicing competitive programming and Data Structures & Algorithms.")
        if internships_count == 0:
            recs.append("💼 **Complete an Internship**: Industrial internships add +3.5 points per internship to your Experience Index.")
        if mock_interview_score < 70:
            recs.append("🗣️ **Mock Interview Preparation**: Practice technical and HR mock interviews to increase mock interview score to 75+.")
        if not recs:
            recs.append("🌟 **Outstanding Profile**: Your student profile demonstrates strong technical skills, academic performance, and practical experience!")

        for r in recs:
            st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: MULTI-MODEL CONSENSUS
# =============================================================================
with tab2:
    st.subheader("⚖️ Multi-Model Consensus & Comparison")
    st.write("Evaluate candidate attributes simultaneously across all 4 calibrated Machine Learning classifiers.")

    c_col1, c_col2 = st.columns([1, 1])

    payload_comp = {
        "age": 21, "gender": "Male", "cgpa": float(p_cgpa), "branch": "CSE", "college_tier": "Tier 2",
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
        st.markdown(f"### Consensus Outcome: **{majority_label.upper()}** ({placed_votes}/4 Models Agree)")
        st.dataframe(comp_df[["Model Name", "Prediction", "Probability (%)"]], use_container_width=True)

    with c_col2:
        fig_bar = px.bar(
            comp_df,
            x="Model Name",
            y="Probability (%)",
            color="Prediction",
            color_discrete_map={"Placed": "#10b981", "Not Placed": "#f43f5e"},
            text="Probability (%)",
            title="Estimated Probability Across Models"
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab3:
    st.subheader("📊 Institutional Data Analytics & Distributions")

    if raw_df is not None:
        st.write(f"Dataset Dimensions: **{raw_df.shape[0]} Instances, {raw_df.shape[1]} Predictor Columns**")

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            fig_box1 = px.box(
                raw_df,
                x="placement_status",
                y="cgpa",
                color="placement_status",
                title="CGPA Distribution by Placement Status",
                color_discrete_map={"Placed": "#10b981", "Not Placed": "#f43f5e"}
            )
            fig_box1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_box1, use_container_width=True)

        with eda_col2:
            fig_scat = px.scatter(
                raw_df,
                x="coding_skill_score",
                y="aptitude_score",
                color="placement_status",
                title="Coding Skill vs Aptitude Assessment",
                color_discrete_map={"Placed": "#10b981", "Not Placed": "#f43f5e"},
                opacity=0.7
            )
            fig_scat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("#### 🏢 Placement Rates across Academic Branches & Tiers")
        eda_col3, eda_col4 = st.columns(2)

        with eda_col3:
            branch_df = raw_df.groupby(["branch", "placement_status"]).size().reset_index(name="count")
            fig_branch = px.bar(
                branch_df,
                x="branch",
                y="count",
                color="placement_status",
                title="Placement Distribution by Academic Branch",
                barmode="group",
                color_discrete_map={"Placed": "#10b981", "Not Placed": "#f43f5e"}
            )
            fig_branch.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
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
                color_discrete_map={"Placed": "#10b981", "Not Placed": "#f43f5e"}
            )
            fig_tier.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_tier, use_container_width=True)

    else:
        st.info("Raw dataset file (`student_placement_prediction_dataset_2026.csv`) is not available for interactive EDA visualizations.")


# =============================================================================
# TAB 4: MODEL BENCHMARKS & METRICS
# =============================================================================
with tab4:
    st.subheader("🧠 Machine Learning Benchmark Evaluation")
    st.write("Validation metrics evaluated on a stratified 80/20 test split (600 test instances).")

    metrics_data = [
        {"Model": "Logistic Regression", "Accuracy (%)": 90.67, "Precision (%)": 90.96, "Recall (%)": 92.07, "F1 Score (%)": 91.52, "ROC-AUC (%)": 96.08, "Scaler / Calibration": "StandardScaler"},
        {"Model": "K-Nearest Neighbors (KNN)", "Accuracy (%)": 91.67, "Precision (%)": 88.83, "Recall (%)": 96.95, "F1 Score (%)": 92.71, "ROC-AUC (%)": 96.93, "Scaler / Calibration": "RobustScaler"},
        {"Model": "Decision Tree", "Accuracy (%)": 89.67, "Precision (%)": 89.35, "Recall (%)": 92.07, "F1 Score (%)": 90.69, "ROC-AUC (%)": 93.73, "Scaler / Calibration": "Cost-Complexity Pruned"},
        {"Model": "Linear SVM", "Accuracy (%)": 90.67, "Precision (%)": 91.46, "Recall (%)": 91.46, "F1 Score (%)": 91.46, "ROC-AUC (%)": 96.17, "Scaler / Calibration": "Platt Sigmoid Calibration"}
    ]

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    # Comparison plot
    fig_metrics = px.bar(
        metrics_df,
        x="Model",
        y=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC-AUC (%)"],
        barmode="group",
        title="Comprehensive Performance Comparison across Evaluation Metrics",
        color_discrete_sequence=["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#a855f7"]
    )
    fig_metrics.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
    st.plotly_chart(fig_metrics, use_container_width=True)


# =============================================================================
# TAB 5: HISTORY & EXPORT
# =============================================================================
with tab5:
    st.subheader("📜 Session Prediction History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv_data = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download History as CSV",
            data=csv_data,
            file_name="campuspredict_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No predictions recorded in current session yet. Run a prediction in the 'Placement Predictor' tab to populate history.")
