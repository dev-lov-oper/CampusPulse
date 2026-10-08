from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models"

MODEL_FILES = {
    "logistic_regression": ("logistic_regression.pkl", "scaler.pkl"),
    "knn": ("knn.pkl", "scaler_knn.pkl"),
    "decision_tree": ("decision_tree.pkl", None),
    "svm_linear": ("svm_linear.pkl", "scaler_svm_linear.pkl"),
}

models = {}
scalers = {}

for name, (model_file, scaler_file) in MODEL_FILES.items():
    models[name] = joblib.load(MODEL_DIR / model_file)
    if scaler_file:
        scalers[name] = joblib.load(MODEL_DIR / scaler_file)

feature_names = list(scalers["logistic_regression"].feature_names_in_)


import numpy as np

def preprocess(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame([payload])
    
    # Practical domain-specific feature engineering:
    # 1. Academic clearance penalized by backlogs
    frame['academic_clearance_score'] = frame['cgpa'] * np.exp(-0.30 * frame['backlogs'])
    # 2. Strict eligibility milestone
    frame['has_zero_backlogs'] = (frame['backlogs'] == 0).astype(int)
    # 3. Weighted technical aptitude mastery
    frame['technical_mastery'] = (
        0.40 * frame['coding_skill_score'] +
        0.25 * frame['aptitude_score'] +
        0.20 * frame['logical_reasoning_score'] +
        0.15 * frame['mock_interview_score']
    )
    # 4. Industry readiness experience index
    frame['experience_index'] = (
        frame['internships_count'] * 3.5 +
        frame['projects_count'] * 1.5 +
        frame['hackathons_participated'] * 1.5 +
        frame['certifications_count'] * 0.8
    )
    # 5. Composite multi-dimensional placement readiness score
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


def predict_one(model_name: str, payload: dict):
    X = preprocess(payload)
    model = models[model_name]

    if model_name == "decision_tree":
        transformed = X
    else:
        transformed = scalers[model_name].transform(X)

    pred = int(model.predict(transformed)[0])
    label = "Placed" if pred == 1 else "Not Placed"

    # Every model used by the application must expose a probability.
    # The SVM model is calibrated during training, so it has predict_proba().
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"{model_name} does not provide probability output. "
            "Retrain the model with probability calibration."
        )

    score = float(model.predict_proba(transformed)[0, 1]) * 100.0
    score = max(0.0, min(100.0, score))
    return label, score, "probability"
