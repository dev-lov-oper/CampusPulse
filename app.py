"""
CampusPulse AI - Student Placement Intelligence Platform
=========================================================
Luminous Crystal Spatial UI Web Application for Machine Learning Placement Prediction,
Multi-Step Interactive Wizard, Consensus Matrix, and 100k Benchmark Telemetry.
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
# 1. PAGE CONFIGURATION & LUMINOUS CRYSTAL SPATIAL THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CampusPulse AI | Placement Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-void: #090d16;
        --bg-glass-card: rgba(17, 24, 39, 0.72);
        --bg-glass-elevated: rgba(26, 34, 52, 0.85);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-specular: rgba(255, 255, 255, 0.16);
        --border-glow-primary: rgba(99, 102, 241, 0.45);
        --accent-indigo: #6366f1;
        --accent-cyan: #06b6d4;
        --accent-purple: #a855f7;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
    }
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 15.5px !important;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    
    /* Cosmic Obsidian Mesh Background with Multi-Layer Aurora Orbs */
    .stApp {
        background-color: var(--bg-void) !important;
        background-image: 
            radial-gradient(circle at 12% 12%, rgba(99, 102, 241, 0.18) 0%, transparent 48%),
            radial-gradient(circle at 88% 16%, rgba(6, 182, 212, 0.14) 0%, transparent 44%),
            radial-gradient(circle at 50% 85%, rgba(168, 85, 247, 0.12) 0%, transparent 52%),
            radial-gradient(circle at 30% 60%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
            radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 28px 28px !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
    }

    /* Hide ugly Streamlit header anchor links */
    [data-testid="stHeaderActionElements"], .stHeading a, a[href^="#"] {
        display: none !important;
    }
    
    /* Sleek Obsidian Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 23, 0.95) !important;
        backdrop-filter: blur(32px) !important;
        -webkit-backdrop-filter: blur(32px) !important;
        border-right: 1px solid var(--border-glass) !important;
        box-shadow: 6px 0 35px rgba(0, 0, 0, 0.5) !important;
        min-width: 310px !important;
    }
    
    /* Ultra-Sleek Crystal Nav Dock */
    [data-testid="stSidebar"] [data-baseweb="radio"] input {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        background: transparent !important;
        padding: 10px 14px !important;
        border-radius: 12px !important;
        margin-bottom: 5px !important;
        border: 1px solid transparent !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        transform: translateX(4px) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] p {
        font-size: 0.94rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"],
    [data-testid="stSidebar"] div[data-checked="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.18) 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.55) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] p,
    [data-testid="stSidebar"] div[data-checked="true"] p {
        color: #ffffff !important;
        font-weight: 750 !important;
        text-shadow: 0 0 12px rgba(129, 140, 248, 0.6) !important;
    }

    /* Container Border Wrappers -> Deep Translucent Frosted Glass Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-glass-card) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 20px !important;
        box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.6), inset 0 1px 1px var(--border-glass-specular) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        padding: 24px 28px !important;
        margin-bottom: 22px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 22px 50px -10px rgba(0, 0, 0, 0.7), 0 0 30px rgba(99, 102, 241, 0.12), inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
    }

    /* High-Impact Typography & Gradient Headings */
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.025em !important;
        color: var(--text-primary) !important;
    }
    h1 {
        font-size: 2.45rem !important;
        font-weight: 850 !important;
        background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    h2 {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
    }
    h3 {
        font-size: 1.35rem !important;
        font-weight: 750 !important;
        color: #e2e8f0 !important;
    }
    h4 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #cbd5e1 !important;
    }
    p, span, div {
        color: var(--text-secondary);
    }
    
    /* Input Labels */
    .stNumberInput label, .stSelectbox label, .stSlider label, .stTextInput label, .stRadio label {
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    
    /* Input Elements */
    input, select, .stSelectbox div[data-baseweb="select"] {
        font-size: 1rem !important;
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }
    input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.45) !important;
    }
    
    /* Slider Custom Accent */
    div[data-baseweb="slider"] div[role="slider"] {
        background: #6366f1 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 14px rgba(99, 102, 241, 0.8) !important;
    }
    
    /* Luminous Glass Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(26, 34, 52, 0.85) 0%, rgba(17, 24, 39, 0.85) 100%);
        padding: 2.2rem 2.6rem;
        border-radius: 1.6rem;
        box-shadow: 0 18px 45px -10px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.15);
        margin-bottom: 2rem;
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(32px);
        -webkit-backdrop-filter: blur(32px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #4f46e5, #06b6d4, #a855f7);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.16);
        color: #a5b4fc;
        border: 1px solid rgba(129, 140, 248, 0.35);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.18);
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.08rem;
        line-height: 1.65;
        margin-bottom: 0;
        max-width: 900px;
    }
    
    /* Luminous Spatial Glass Cards */
    .metric-card, .leader-card, .driver-card, .glossary-card, .about-card {
        background: var(--bg-glass-card);
        border: 1px solid var(--border-glass);
        border-radius: 1.25rem;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 14px 34px -6px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        text-align: left;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover, .leader-card:hover, .driver-card:hover, .glossary-card:hover, .about-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 20px 45px -8px rgba(0, 0, 0, 0.7), 0 0 30px rgba(99, 102, 241, 0.18);
    }
    .metric-value {
        font-size: 2.25rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-top: 5px;
        letter-spacing: 0.02em;
    }

    /* Champion Glowing Card for Linear SVM 94.15% */
    .champion-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(99, 102, 241, 0.12) 100%) !important;
        border: 1.5px solid rgba(52, 211, 153, 0.5) !important;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.22), inset 0 1px 2px rgba(255, 255, 255, 0.25) !important;
    }
    .champion-card:hover {
        border-color: rgba(52, 211, 153, 0.75) !important;
        box-shadow: 0 0 45px rgba(16, 185, 129, 0.35), inset 0 1px 2px rgba(255, 255, 255, 0.35) !important;
    }

    /* Outcome Badges (Cyber Holographic Style) */
    .status-badge-placed {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(4, 120, 87, 0.65) 100%);
        color: #6ee7b7;
        padding: 1.4rem 2rem;
        border-radius: 1.25rem;
        font-size: 2.1rem;
        font-weight: 900;
        text-align: center;
        border: 1.5px solid rgba(52, 211, 153, 0.55);
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.3);
        letter-spacing: 0.04em;
        text-shadow: 0 0 14px rgba(110, 231, 183, 0.5);
    }
    .status-badge-not-placed {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.85) 0%, rgba(190, 18, 60, 0.65) 100%);
        color: #fda4af;
        padding: 1.4rem 2rem;
        border-radius: 1.25rem;
        font-size: 2.1rem;
        font-weight: 900;
        text-align: center;
        border: 1.5px solid rgba(251, 113, 133, 0.55);
        box-shadow: 0 0 35px rgba(244, 63, 94, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.3);
        letter-spacing: 0.04em;
        text-shadow: 0 0 14px rgba(253, 164, 175, 0.5);
    }
    
    /* Recommendations Box */
    .rec-box {
        background: rgba(30, 41, 59, 0.65);
        border-left: 4px solid var(--accent-indigo);
        border-right: 1px solid var(--border-glass);
        border-top: 1px solid var(--border-glass);
        border-bottom: 1px solid var(--border-glass);
        padding: 1.2rem 1.4rem;
        border-radius: 0 1rem 1rem 0;
        margin-top: 0.85rem;
        color: var(--text-secondary);
        font-size: 0.98rem;
        line-height: 1.65;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
    }

    /* Gradient Glowing Buttons */
    .stButton>button {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 22px -4px rgba(79, 70, 229, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px -4px rgba(79, 70, 229, 0.7), 0 0 25px rgba(124, 58, 237, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }
    .stButton>button:active {
        transform: translateY(0) !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(10, 14, 23, 0.8);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.16);
        border-radius: 999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.55);
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


# -----------------------------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1

if "candidate_data" not in st.session_state:
    st.session_state.candidate_data = {
        "age": 21,
        "gender": "Male",
        "cgpa": 8.2,
        "backlogs": 0,
        "attendance_percentage": 88.0,
        "branch": "CSE",
        "college_tier": "Tier 1",
        "coding_skill_score": 82.0,
        "aptitude_score": 76.0,
        "logical_reasoning_score": 80.0,
        "communication_skill_score": 78.0,
        "mock_interview_score": 75.0,
        "github_repos": 8,
        "internships_count": 1,
        "projects_count": 3,
        "hackathons_participated": 2,
        "certifications_count": 2,
        "linkedin_connections": 320,
        "extracurricular_score": 70.0,
        "leadership_score": 68.0,
        "volunteer_experience": "Yes",
        "study_hours_per_day": 4.5,
        "sleep_hours": 7.0
    }

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & CANDIDATE PRESETS
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:20px;">
    <div style="width:46px; height:46px; border-radius:14px; background:linear-gradient(135deg,#6366f1,#06b6d4); display:grid; place-items:center; color:#fff; font-size:24px; font-weight:800; box-shadow:0 8px 24px rgba(99,102,241,0.4), inset 0 1px 1px rgba(255,255,255,0.4);">🎓</div>
    <div>
        <strong style="font-size:1.3rem; color:#ffffff; display:block; letter-spacing:-0.02em; font-family:'Outfit',sans-serif;">CampusPulse</strong>
        <span style="font-size:0.75rem; color:#a5b4fc; font-weight:750; letter-spacing:0.1em; text-transform:uppercase;">PLACEMENT INTELLIGENCE</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:0.8rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;'>WORKSPACE</p>", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    options=[
        "🔮 Predict Placement",
        "📊 Dashboard",
        "⚖️ Model Consensus",
        "🧠 Model Performance",
        "📜 History",
        "🏢 Institutional Analytics",
        "ℹ️ About"
    ],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:0.8rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;'>ACTIVE CLASSIFIER</p>", unsafe_allow_html=True)

selected_model_key = st.sidebar.selectbox(
    "Classifier Model:",
    options=["logistic_regression", "svm_linear", "decision_tree", "knn"],
    format_func=lambda x: {
        "logistic_regression": "Logistic Regression (94.0%)",
        "svm_linear": "Linear SVM (94.15%)",
        "decision_tree": "Decision Tree (91.4%)",
        "knn": "K-NN Classifier (90.4%)"
    }[x],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:0.8rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;'>PROFILES PRESET</p>", unsafe_allow_html=True)

preset_choice = st.sidebar.radio(
    "Autofill Candidate Profile:",
    options=["Balanced Candidate", "Top Scholar (High Placement)", "Average Student", "At-Risk Candidate"],
    index=0
)

if st.sidebar.button("Load Profile to Wizard", use_container_width=True):
    if preset_choice == "Top Scholar (High Placement)":
        st.session_state.candidate_data.update({
            "cgpa": 8.9, "backlogs": 0, "coding_skill_score": 88.0, "aptitude_score": 85.0,
            "logical_reasoning_score": 90.0, "mock_interview_score": 85.0, "communication_skill_score": 82.0,
            "internships_count": 2, "projects_count": 4, "hackathons_participated": 3, "certifications_count": 3,
            "github_repos": 12, "linkedin_connections": 350, "attendance_percentage": 94.0,
            "extracurricular_score": 80.0, "leadership_score": 85.0, "sleep_hours": 7.5, "study_hours_per_day": 6.0
        })
    elif preset_choice == "Average Student":
        st.session_state.candidate_data.update({
            "cgpa": 7.4, "backlogs": 0, "coding_skill_score": 68.0, "aptitude_score": 68.0,
            "logical_reasoning_score": 70.0, "mock_interview_score": 65.0, "communication_skill_score": 70.0,
            "internships_count": 1, "projects_count": 2, "hackathons_participated": 1, "certifications_count": 1,
            "github_repos": 4, "linkedin_connections": 120, "attendance_percentage": 84.0,
            "extracurricular_score": 55.0, "leadership_score": 50.0, "sleep_hours": 7.0, "study_hours_per_day": 4.0
        })
    elif preset_choice == "At-Risk Candidate":
        st.session_state.candidate_data.update({
            "cgpa": 5.6, "backlogs": 2, "coding_skill_score": 38.0, "aptitude_score": 42.0,
            "logical_reasoning_score": 40.0, "mock_interview_score": 40.0, "communication_skill_score": 45.0,
            "internships_count": 0, "projects_count": 1, "hackathons_participated": 0, "certifications_count": 0,
            "github_repos": 1, "linkedin_connections": 40, "attendance_percentage": 65.0,
            "extracurricular_score": 25.0, "leadership_score": 25.0, "sleep_hours": 5.5, "study_hours_per_day": 1.5
        })
    else:
        st.session_state.candidate_data.update({
            "cgpa": 8.2, "backlogs": 0, "coding_skill_score": 82.0, "aptitude_score": 76.0,
            "logical_reasoning_score": 80.0, "mock_interview_score": 75.0, "communication_skill_score": 78.0,
            "internships_count": 1, "projects_count": 3, "hackathons_participated": 2, "certifications_count": 2,
            "github_repos": 8, "linkedin_connections": 320, "attendance_percentage": 88.0,
            "extracurricular_score": 70.0, "leadership_score": 68.0, "sleep_hours": 7.0, "study_hours_per_day": 4.5
        })
    st.session_state.last_result = None
    st.success("Candidate attributes loaded!")


# =============================================================================
# PAGE: 🔮 PREDICT PLACEMENT (SPATIAL 2-COLUMN WIZARD MATCHING REACT UI)
# =============================================================================
if nav_page == "🔮 Predict Placement":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <span class="hero-badge">PREDICTION WORKSPACE</span>
        <h1 style="margin: 6px 0 8px;">Predict Placement</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; line-height: 1.6;">Enter the 23 student attributes used by the supplied trained models.</p>
    </div>
    """, unsafe_allow_html=True)

    step_info = [
        ("Personal", "Personal Information"),
        ("Academic", "Academic Standing & Clearance"),
        ("Technical", "Technical Assessment & Mastery"),
        ("Experience", "Internships, Projects & Hackathons"),
        ("Professional", "Professional Footprint & Leadership"),
        ("Lifestyle", "Habits, Study & Lifestyle")
    ]

    current_step = st.session_state.wizard_step

    # Spatial 2-Column Wizard Layout: Left Step Navigation Dock, Right Card
    col_nav, col_body = st.columns([1, 2.8], gap="large")

    # Column 1: Step Navigation List (matches React UI sidebar dock)
    with col_nav:
        st.markdown("<p style='font-size:0.8rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;'>STEPS</p>", unsafe_allow_html=True)
        for idx, (short_name, _) in enumerate(step_info, start=1):
            is_active = (idx == current_step)
            btn_label = f"{'👉 ' if is_active else '     '}{idx}.  {short_name}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(btn_label, key=f"step_btn_{idx}", type=btn_type, use_container_width=True):
                st.session_state.wizard_step = idx
                st.rerun()

    # Column 2: The Main Card Container with Form Elements Inside It
    with col_body:
        with st.container(border=True):
            # Step Card Header
            h_left, h_right = st.columns([3, 1])
            with h_left:
                st.markdown(f"<span style='font-size:0.82rem; font-weight:800; color:#4f46e5; letter-spacing:0.12em; text-transform:uppercase;'>STEP {current_step}</span>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='margin:4px 0 12px; font-size:1.75rem; color:#f8fafc;'>{step_info[current_step - 1][1]}</h2>", unsafe_allow_html=True)
            with h_right:
                st.markdown(f"<div style='text-align:right; font-size:0.95rem; font-weight:700; color:#64748b; margin-top:12px;'>{current_step} / 6</div>", unsafe_allow_html=True)
                st.progress(current_step / 6.0)

            st.write("")

            # 2-Column Form Fields Inside Card Container
            c_left, c_right = st.columns(2)

            if current_step == 1:
                with c_left:
                    st.session_state.candidate_data["age"] = st.number_input(
                        "Age (Years)",
                        min_value=18, max_value=28,
                        value=int(st.session_state.candidate_data.get("age", 21)),
                        step=1
                    )
                with c_right:
                    g_options = ["Male", "Female"]
                    g_idx = g_options.index(st.session_state.candidate_data.get("gender", "Male")) if st.session_state.candidate_data.get("gender", "Male") in g_options else 0
                    st.session_state.candidate_data["gender"] = st.selectbox(
                        "Gender",
                        options=g_options,
                        index=g_idx
                    )

            elif current_step == 2:
                with c_left:
                    st.session_state.candidate_data["cgpa"] = st.number_input(
                        "Cumulative GPA (CGPA out of 10.0)",
                        min_value=4.5, max_value=10.0,
                        value=float(st.session_state.candidate_data.get("cgpa", 8.2)),
                        step=0.1
                    )
                    st.session_state.candidate_data["backlogs"] = st.number_input(
                        "Active Backlogs Count",
                        min_value=0, max_value=6,
                        value=int(st.session_state.candidate_data.get("backlogs", 0)),
                        step=1
                    )
                    st.session_state.candidate_data["attendance_percentage"] = st.slider(
                        "Attendance Percentage (%)",
                        min_value=50.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("attendance_percentage", 88.0)),
                        step=1.0
                    )
                with c_right:
                    b_opts = ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]
                    b_idx = b_opts.index(st.session_state.candidate_data.get("branch", "CSE")) if st.session_state.candidate_data.get("branch", "CSE") in b_opts else 0
                    st.session_state.candidate_data["branch"] = st.selectbox(
                        "Academic Branch",
                        options=b_opts,
                        index=b_idx
                    )
                    t_opts = ["Tier 1", "Tier 2", "Tier 3"]
                    t_idx = t_opts.index(st.session_state.candidate_data.get("college_tier", "Tier 1")) if st.session_state.candidate_data.get("college_tier", "Tier 1") in t_opts else 0
                    st.session_state.candidate_data["college_tier"] = st.selectbox(
                        "College Tier",
                        options=t_opts,
                        index=t_idx
                    )

            elif current_step == 3:
                with c_left:
                    st.session_state.candidate_data["coding_skill_score"] = st.slider(
                        "Coding Assessment Score (0-100)",
                        min_value=20.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("coding_skill_score", 82.0)),
                        step=1.0
                    )
                    st.session_state.candidate_data["aptitude_score"] = st.slider(
                        "General Aptitude Score (0-100)",
                        min_value=20.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("aptitude_score", 76.0)),
                        step=1.0
                    )
                with c_right:
                    st.session_state.candidate_data["logical_reasoning_score"] = st.slider(
                        "Logical Reasoning Score (0-100)",
                        min_value=20.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("logical_reasoning_score", 80.0)),
                        step=1.0
                    )
                    st.session_state.candidate_data["mock_interview_score"] = st.slider(
                        "Mock Interview Score (0-100)",
                        min_value=20.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("mock_interview_score", 75.0)),
                        step=1.0
                    )

            elif current_step == 4:
                with c_left:
                    st.session_state.candidate_data["internships_count"] = st.number_input(
                        "Completed Industrial Internships",
                        min_value=0, max_value=10,
                        value=int(st.session_state.candidate_data.get("internships_count", 1)),
                        step=1
                    )
                    st.session_state.candidate_data["projects_count"] = st.number_input(
                        "Full-Stack Software Projects",
                        min_value=0, max_value=20,
                        value=int(st.session_state.candidate_data.get("projects_count", 3)),
                        step=1
                    )
                with c_right:
                    st.session_state.candidate_data["hackathons_participated"] = st.number_input(
                        "Hackathons Participated",
                        min_value=0, max_value=20,
                        value=int(st.session_state.candidate_data.get("hackathons_participated", 2)),
                        step=1
                    )
                    st.session_state.candidate_data["certifications_count"] = st.number_input(
                        "Technical Certifications",
                        min_value=0, max_value=20,
                        value=int(st.session_state.candidate_data.get("certifications_count", 2)),
                        step=1
                    )

            elif current_step == 5:
                with c_left:
                    st.session_state.candidate_data["github_repos"] = st.number_input(
                        "Public GitHub Repositories",
                        min_value=0, max_value=100,
                        value=int(st.session_state.candidate_data.get("github_repos", 8)),
                        step=1
                    )
                    st.session_state.candidate_data["linkedin_connections"] = st.number_input(
                        "LinkedIn Connections",
                        min_value=0, max_value=5000,
                        value=int(st.session_state.candidate_data.get("linkedin_connections", 320)),
                        step=10
                    )
                with c_right:
                    st.session_state.candidate_data["communication_skill_score"] = st.slider(
                        "Verbal Communication Score (0-100)",
                        min_value=20.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("communication_skill_score", 78.0)),
                        step=1.0
                    )
                    st.session_state.candidate_data["leadership_score"] = st.slider(
                        "Leadership & Initiative Score (0-100)",
                        min_value=0.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("leadership_score", 68.0)),
                        step=1.0
                    )

            elif current_step == 6:
                with c_left:
                    st.session_state.candidate_data["study_hours_per_day"] = st.slider(
                        "Daily Focused Study Hours",
                        min_value=0.5, max_value=10.0,
                        value=float(st.session_state.candidate_data.get("study_hours_per_day", 4.5)),
                        step=0.5
                    )
                    st.session_state.candidate_data["sleep_hours"] = st.slider(
                        "Average Sleep Hours",
                        min_value=4.0, max_value=10.0,
                        value=float(st.session_state.candidate_data.get("sleep_hours", 7.0)),
                        step=0.5
                    )
                with c_right:
                    st.session_state.candidate_data["extracurricular_score"] = st.slider(
                        "Extracurricular Activity Score (0-100)",
                        min_value=0.0, max_value=100.0,
                        value=float(st.session_state.candidate_data.get("extracurricular_score", 70.0)),
                        step=1.0
                    )
                    v_opts = ["Yes", "No"]
                    v_idx = v_opts.index(st.session_state.candidate_data.get("volunteer_experience", "Yes")) if st.session_state.candidate_data.get("volunteer_experience", "Yes") in v_opts else 0
                    st.session_state.candidate_data["volunteer_experience"] = st.selectbox(
                        "Community / Volunteer Experience",
                        options=v_opts,
                        index=v_idx
                    )

            st.write("")
            st.markdown("---")

            # Navigation Controls Row Inside Card
            b_prev, b_spacer, b_next = st.columns([1, 1.8, 1.2])

            with b_prev:
                if current_step > 1:
                    if st.button("⬅️ Previous", use_container_width=True):
                        st.session_state.wizard_step -= 1
                        st.rerun()

            with b_next:
                if current_step < 6:
                    if st.button("Next ➡️", type="primary", use_container_width=True):
                        st.session_state.wizard_step += 1
                        st.rerun()
                else:
                    if st.button("🔮 Predict Placement", type="primary", use_container_width=True):
                        with st.spinner("Running calibrated model inference..."):
                            lbl, pr, processed_feat = predict_student(selected_model_key, st.session_state.candidate_data)
                            st.session_state.last_result = {
                                "label": lbl,
                                "prob": pr,
                                "features": processed_feat,
                                "model": selected_model_key
                            }
                            st.session_state.history.append({
                                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Model": selected_model_key,
                                "Prediction": lbl,
                                "Probability (%)": round(pr, 2),
                                "CGPA": st.session_state.candidate_data["cgpa"],
                                "Backlogs": st.session_state.candidate_data["backlogs"],
                                "Coding Score": st.session_state.candidate_data["coding_skill_score"],
                                "Branch": st.session_state.candidate_data["branch"]
                            })
                        st.rerun()

    # Outcome Verdict Display
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.write("")
        st.markdown("---")
        st.subheader("🎯 Placement Prediction Verdict")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if res["label"] == "Placed":
                st.markdown('<div class="status-badge-placed">✅ PLACED (PROBABLE)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge-not-placed">❌ NOT PLACED (AT-RISK)</div>', unsafe_allow_html=True)

            # Luminous Probability Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["prob"],
                number={'suffix': '%', 'font': {'size': 42, 'color': '#ffffff', 'family': 'Outfit'}},
                title={'text': f"Placement Likelihood ({res['model'].replace('_',' ').title()})", 'font': {'size': 17, 'color': '#334155', 'family': 'Outfit'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'color': '#94a3b8', 'size': 13}},
                    'bar': {'color': "#4f46e5" if res["prob"] >= 50 else "#e11d48"},
                    'bgcolor': "rgba(15, 23, 42, 0.6)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(225, 29, 72, 0.08)'},
                        {'range': [50, 75], 'color': 'rgba(2, 132, 199, 0.08)'},
                        {'range': [75, 100], 'color': 'rgba(5, 150, 105, 0.1)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                height=270,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### 📈 Domain-Engineered Indicator Scores")
            readiness = res["features"].get('placement_readiness_score', 0.0)
            tech_mastery = res["features"].get('technical_mastery', 0.0)
            exp_index = res["features"].get('experience_index', 0.0)
            acad_clear = res["features"].get('academic_clearance_score', 0.0)

            ind_c1, ind_c2 = st.columns(2)
            with ind_c1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{readiness:.1f}/100</div><div class="metric-label">Composite Readiness</div></div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="metric-card"><div class="metric-value">{acad_clear:.2f}</div><div class="metric-label">Academic Clearance</div></div>', unsafe_allow_html=True)
            with ind_c2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{tech_mastery:.1f}/100</div><div class="metric-label">Technical Mastery</div></div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="metric-card"><div class="metric-value">{exp_index:.1f}</div><div class="metric-label">Experience Index</div></div>', unsafe_allow_html=True)

        # Strategic Action Items
        st.markdown("#### 💡 Strategic Recommendations for Upskilling")
        recs = []
        c_data = st.session_state.candidate_data
        if c_data["backlogs"] > 0:
            recs.append(f"⚠️ **Clear Active Backlogs**: You currently have **{c_data['backlogs']} active backlogs**. Enterprise campus drives enforce a 0-backlog cutoff. Clearing them immediately raises eligibility.")
        if c_data["coding_skill_score"] < 75:
            recs.append("💻 **Enhance Coding Competency**: Target a coding assessment score above 75+ with competitive programming and Data Structures practice.")
        if c_data["internships_count"] == 0:
            recs.append("💼 **Secure Practical Internship Experience**: Completing 1-2 industrial internships adds +3.5 weighted points per internship to your placement profile.")
        if c_data["mock_interview_score"] < 75:
            recs.append("🗣️ **Conduct Mock Technical Interviews**: Improve mock interview feedback score to 75+ to boost interview calibration.")
        if not recs:
            recs.append("🌟 **Outstanding Profile**: This candidate displays top-tier technical readiness, excellent academic standing, and competitive practical experience!")

        for r in recs:
            st.markdown(f'<div class="rec-box">{r}</div>', unsafe_allow_html=True)

        if st.button("🔄 Reset / Predict Another Candidate", use_container_width=False):
            st.session_state.wizard_step = 1
            st.session_state.last_result = None
            st.rerun()


# =============================================================================
# PAGE: 📊 WORKSPACE DASHBOARD
# =============================================================================
elif nav_page == "📊 Dashboard":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="hero-badge">WORKSPACE TELEMETRY</span>
        <h1 style="margin: 6px 0 8px;">Workspace Dashboard</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; line-height: 1.6;">High-level placement analytics, calibrated machine learning telemetry, and model benchmarks.</p>
    </div>
    """, unsafe_allow_html=True)

    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">100,000</div>
            <div class="metric-label">Alumni Records Calibrated</div>
        </div>
        """, unsafe_allow_html=True)
    with st2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94.15%</div>
            <div class="metric-label">Peak Model Accuracy (Linear SVM)</div>
        </div>
        """, unsafe_allow_html=True)
    with st3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">98.95%</div>
            <div class="metric-label">Peak ROC-AUC Distinction</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("### 🏆 100,000-Dataset Accuracy Benchmark")
    acc_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy (%)": 94.05, "Recall (%)": 94.65, "ROC-AUC (%)": 98.95},
        {"Model": "Linear SVM", "Accuracy (%)": 94.15, "Recall (%)": 94.63, "ROC-AUC (%)": 98.95},
        {"Model": "Decision Tree", "Accuracy (%)": 91.36, "Recall (%)": 92.13, "ROC-AUC (%)": 97.49},
        {"Model": "K-Nearest Neighbors", "Accuracy (%)": 90.38, "Recall (%)": 92.50, "ROC-AUC (%)": 97.11}
    ])

    fig_acc = px.bar(
        acc_df,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        text="Accuracy (%)",
        color_discrete_sequence=["#4f46e5", "#0284c7", "#059669", "#7c3aed"]
    )
    fig_acc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        font_family="Outfit",
        font_size=14,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#cbd5e1")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#cbd5e1"), range=[85, 100])
    )
    st.plotly_chart(fig_acc, use_container_width=True)


# =============================================================================
# PAGE: ⚖️ MULTI-MODEL CONSENSUS
# =============================================================================
elif nav_page == "⚖️ Model Consensus":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="hero-badge">MULTI-MODEL MATRIX</span>
        <h1 style="margin: 6px 0 8px;">Consensus Matrix</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; line-height: 1.6;">Simultaneously evaluate student qualifications across all 4 machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)

    comp_results = []
    for m_name in ["logistic_regression", "svm_linear", "decision_tree", "knn"]:
        lbl, pr, _ = predict_student(m_name, st.session_state.candidate_data)
        comp_results.append({
            "Model Name": {
                "logistic_regression": "Logistic Regression",
                "svm_linear": "Linear SVM",
                "decision_tree": "Decision Tree",
                "knn": "K-Nearest Neighbors"
            }[m_name],
            "Prediction": lbl,
            "Probability (%)": round(pr, 2)
        })

    comp_df = pd.DataFrame(comp_results)
    placed_votes = sum(comp_df["Prediction"] == "Placed")
    majority_label = "Placed" if placed_votes >= 2 else "Not Placed"

    c_col1, c_col2 = st.columns([1, 1])
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            font_family="Outfit",
            font_size=14,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#cbd5e1")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#cbd5e1"), range=[0, 105])
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# PAGE: 🧠 MODEL PERFORMANCE & BENCHMARKS (100k DATASET)
# =============================================================================
elif nav_page == "🧠 Model Performance":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="hero-badge">CALIBRATED ML BENCHMARKS</span>
        <h1 style="margin: 6px 0 8px;">Model Performance & Telemetry</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; line-height: 1.6;">Validation metrics evaluated on a stratified 80/20 test split (20,000 test instances out of 100,000 students).</p>
    </div>
    """, unsafe_allow_html=True)

    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    with l_col1:
        st.markdown("""
        <div class="leader-card champion-card">
            <span style="font-size:11px; font-weight:800; color:#34d399; background:rgba(16,185,129,0.18); border:1px solid rgba(52,211,153,0.3); padding:4px 10px; border-radius:999px; letter-spacing:0.04em;">👑 TOP ACCURACY CHAMPION</span>
            <div class="metric-value" style="margin-top:10px;">94.15%</div>
            <strong style="color:#ffffff; font-size:15px; display:block; margin-top:4px;">Linear SVM</strong>
            <p style="color:#94a3b8; font-size:13px; margin:4px 0 0; line-height:1.5;">Leading accuracy across 100,000 students with regularized probabilistic margins.</p>
        </div>
        """, unsafe_allow_html=True)
    with l_col2:
        st.markdown("""
        <div class="leader-card">
            <span style="font-size:11px; font-weight:800; color:#38bdf8; background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.3); padding:4px 10px; border-radius:999px; letter-spacing:0.04em;">🎯 TOP RECALL</span>
            <div class="metric-value" style="margin-top:10px;">94.65%</div>
            <strong style="color:#ffffff; font-size:15px; display:block; margin-top:4px;">Logistic Regression</strong>
            <p style="color:#94a3b8; font-size:13px; margin:4px 0 0; line-height:1.5;">Catches 95 of 100 placed candidates with minimal false negative loss.</p>
        </div>
        """, unsafe_allow_html=True)
    with l_col3:
        st.markdown("""
        <div class="leader-card">
            <span style="font-size:11px; font-weight:800; color:#c084fc; background:rgba(192,132,252,0.15); border:1px solid rgba(192,132,252,0.3); padding:4px 10px; border-radius:999px; letter-spacing:0.04em;">⚡ TOP ROC-AUC</span>
            <div class="metric-value" style="margin-top:10px;">98.95%</div>
            <strong style="color:#ffffff; font-size:15px; display:block; margin-top:4px;">Linear SVM & Logistic</strong>
            <p style="color:#94a3b8; font-size:13px; margin:4px 0 0; line-height:1.5;">Near-perfect distinction between placed and unplaced profiles.</p>
        </div>
        """, unsafe_allow_html=True)
    with l_col4:
        st.markdown("""
        <div class="leader-card">
            <span style="font-size:11px; font-weight:800; color:#fbbf24; background:rgba(251,191,36,0.15); border:1px solid rgba(251,191,36,0.3); padding:4px 10px; border-radius:999px; letter-spacing:0.04em;">🔍 AUDITABLE RULES</span>
            <div class="metric-value" style="margin-top:10px;">91.36%</div>
            <strong style="color:#ffffff; font-size:15px; display:block; margin-top:4px;">Decision Tree</strong>
            <p style="color:#94a3b8; font-size:13px; margin:4px 0 0; line-height:1.5;">Transparent hierarchical rule splits with 92.14% balanced F1.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    metrics_data = [
        {"Model": "Logistic Regression", "Accuracy (%)": 94.05, "Precision (%)": 94.53, "Recall (%)": 94.65, "F1 Score (%)": 94.59, "ROC-AUC (%)": 98.95},
        {"Model": "Linear SVM", "Accuracy (%)": 94.15, "Precision (%)": 94.54, "Recall (%)": 94.63, "F1 Score (%)": 94.58, "ROC-AUC (%)": 98.95},
        {"Model": "Decision Tree", "Accuracy (%)": 91.36, "Precision (%)": 92.15, "Recall (%)": 92.13, "F1 Score (%)": 92.14, "ROC-AUC (%)": 97.49},
        {"Model": "KNN", "Accuracy (%)": 90.38, "Precision (%)": 90.25, "Recall (%)": 92.50, "F1 Score (%)": 91.36, "ROC-AUC (%)": 97.11}
    ]
    metrics_df = pd.DataFrame(metrics_data)

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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        font_family="Outfit",
        font_size=14,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#cbd5e1")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#cbd5e1"), range=[80, 102])
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Plain-English Glossary
    st.markdown("#### 📖 Plain-English Metric Glossary")
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        st.markdown("""
        <div class="glossary-card">
            <strong style="font-size:15px; color:#ffffff;">Accuracy</strong>
            <p style="color:#94a3b8; font-size:13px; margin:6px 0; line-height:1.5;">Total fraction of predictions that match true outcomes.</p>
            <small style="color:#059669; font-weight:700;">Linear SVM leads at 94.15%</small>
        </div>
        """, unsafe_allow_html=True)
    with g_col2:
        st.markdown("""
        <div class="glossary-card">
            <strong style="font-size:15px; color:#ffffff;">Recall</strong>
            <p style="color:#94a3b8; font-size:13px; margin:6px 0; line-height:1.5;">Out of all placed students, what fraction the AI identified.</p>
            <small style="color:#059669; font-weight:700;">Logistic leads at 94.65%</small>
        </div>
        """, unsafe_allow_html=True)
    with g_col3:
        st.markdown("""
        <div class="glossary-card">
            <strong style="font-size:15px; color:#ffffff;">Precision</strong>
            <p style="color:#94a3b8; font-size:13px; margin:6px 0; line-height:1.5;">When model says "Placed", how often is it truly correct.</p>
            <small style="color:#059669; font-weight:700;">Linear SVM leads at 94.54%</small>
        </div>
        """, unsafe_allow_html=True)
    with g_col4:
        st.markdown("""
        <div class="glossary-card">
            <strong style="font-size:15px; color:#ffffff;">ROC-AUC</strong>
            <p style="color:#94a3b8; font-size:13px; margin:6px 0; line-height:1.5;">Separation ability across all confidence levels (100% is flawless).</p>
            <small style="color:#059669; font-weight:700;">SVM & Logistic lead at 98.95%</small>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE: 📜 PLACEMENT HISTORY
# =============================================================================
elif nav_page == "📜 History":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="hero-badge">SESSION PREDICTIONS</span>
        <h1 style="margin: 6px 0 8px;">Placement Prediction History</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; line-height: 1.6;">Audit log of candidate profiles evaluated during this active session.</p>
    </div>
    """, unsafe_allow_html=True)

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
        st.info("No predictions recorded in this session yet. Run a prediction in the Predict Placement Step Wizard to populate history.")


# =============================================================================
# PAGE: 🏢 INSTITUTIONAL DATA ANALYTICS
# =============================================================================
elif nav_page == "🏢 Institutional Analytics":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="hero-badge">EXPLORATORY DATA ANALYSIS</span>
        <h1 style="margin: 6px 0 8px;">Institutional Placement Analytics</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; line-height: 1.6;">Empirical distribution of 100,000 alumni across academic branches and qualification gates.</p>
    </div>
    """, unsafe_allow_html=True)

    if raw_df is not None:
        st.write(f"Analyzed Dataset: **{raw_df.shape[0]:,} Student Records** across 26 placement indicators.")

        eda_c1, eda_c2 = st.columns(2)
        with eda_c1:
            fig_box1 = px.box(
                raw_df.sample(min(5000, len(raw_df)), random_state=42),
                x="placement_status",
                y="cgpa",
                color="placement_status",
                title="CGPA Distribution by Placement Outcome",
                color_discrete_map={"Placed": "#059669", "Not Placed": "#e11d48"}
            )
            fig_box1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc", font_family="Outfit", font_size=14, xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
            st.plotly_chart(fig_box1, use_container_width=True)

        with eda_c2:
            fig_scat = px.scatter(
                raw_df.sample(min(3000, len(raw_df)), random_state=42),
                x="coding_skill_score",
                y="aptitude_score",
                color="placement_status",
                title="Coding Skill vs Aptitude Assessment",
                color_discrete_map={"Placed": "#059669", "Not Placed": "#e11d48"},
                opacity=0.6
            )
            fig_scat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc", font_family="Outfit", font_size=14, xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
            st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.info("Loading student dataset...")


# =============================================================================
# PAGE: ℹ️ ABOUT PLATFORM
# =============================================================================
elif nav_page == "ℹ️ About":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="hero-badge">PLATFORM ARCHITECTURE</span>
        <h1 style="margin: 6px 0 8px;">About CampusPulse AI</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0; line-height: 1.6;">Enterprise-grade placement intelligence combining machine learning and recruitment science.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="padding: 2.2rem 2.6rem;">
        <h3 style="margin-top:0;">System Overview</h3>
        <p style="color:#cbd5e1; font-size:1.02rem; line-height:1.75;">
            <strong style="color:#ffffff;">CampusPulse</strong> is a machine learning student placement prediction system designed to evaluate student employability, identify at-risk candidates early, and recommend actionable upskilling pathways.
        </p>
        <p style="color:#cbd5e1; font-size:1.02rem; line-height:1.75; margin-bottom:0;">
            The model pipeline was trained on <strong style="color:#a5b4fc;">100,000 alumni records</strong> with stratified validation achieving <strong style="color:#34d399;">94.15% Accuracy</strong> and <strong style="color:#c084fc;">98.95% ROC-AUC</strong>, featuring multi-model consensus and real-time inference.
        </p>
    </div>
    """, unsafe_allow_html=True)
