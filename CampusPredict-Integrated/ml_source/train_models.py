"""
CampusPredict - Machine Learning Training & Evaluation Pipeline
================================================================
This script implements the complete end-to-end ML training workflow for
Student Placement Prediction:
  1: Data Cleaning & Splitting
  2: Logistic Regression Training & Evaluation
  3: K-Nearest Neighbors (KNN with RobustScaler)
  4: Decision Tree (Cost-Complexity Pruning)
  5: Linear Support Vector Machine (Linear SVM with Platt Calibration)
  6: Evaluation Summary & Export to backend/ml_models
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def find_dataset() -> Path:
    """Locate the dataset file across potential working directories."""
    candidates = [
        Path("ml_source/student_placement_prediction_dataset_2026.csv"),
        Path("student_placement_prediction_dataset_2026.csv"),
        Path("../student_placement_prediction_dataset_2026.csv"),
        Path(__file__).resolve().parent / "student_placement_prediction_dataset_2026.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate student_placement_prediction_dataset_2026.csv")


def step_1_data_cleaning_and_splitting(data_path: Path):
    """
    1: Data Cleaning & Splitting
    ---------------------------
    - Load raw student placement dataset
    - Inspect data types, nulls, duplicates
    - Encode binary target (Placed: 1, Not Placed: 0)
    - Engineer domain-specific placement indicators
    - One-hot encode categorical predictors
    - Separate predictors X and ground-truth target y
    - Perform stratified 80/20 train-test split
    """
    print("=" * 70)
    print("STEP 1: DATA CLEANING & SPLITTING")
    print("=" * 70)

    # 1.1 Load dataset
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Initial Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1.2 Check Missing Values and Duplicates
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"Missing Values: {missing_count} | Duplicate Rows: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Shape after removing duplicates: {df.shape}")

    # Fill any numeric NaNs with column median (defensive check)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 1.3 Target Encoding
    target_col = "placement_status"
    print(f"\nTarget distribution before encoding:\n{df[target_col].value_counts()}")
    if df[target_col].dtype == object or df[target_col].dtype == "string":
        df[target_col] = df[target_col].map({"Not Placed": 0, "Placed": 1})
    print(f"Target distribution after encoding:\n{df[target_col].value_counts()}")

    # 1.4 Domain-Specific Feature Engineering
    print("\nEngineering domain-specific placement indicators...")
    # (1) Academic clearance attenuated by active backlogs
    df["academic_clearance_score"] = df["cgpa"] * np.exp(-0.30 * df["backlogs"])

    # (2) Strict zero backlogs eligibility flag
    df["has_zero_backlogs"] = (df["backlogs"] == 0).astype(int)

    # (3) Technical competency mastery (weighted campus evaluation test scores)
    df["technical_mastery"] = (
        0.40 * df["coding_skill_score"]
        + 0.25 * df["aptitude_score"]
        + 0.20 * df["logical_reasoning_score"]
        + 0.15 * df["mock_interview_score"]
    )

    # (4) Practical industry experience index
    df["experience_index"] = (
        df["internships_count"] * 3.5
        + df["projects_count"] * 1.5
        + df["hackathons_participated"] * 1.5
        + df["certifications_count"] * 0.8
    )

    # (5) Composite employability readiness score
    df["placement_readiness_score"] = (
        0.35 * (df["academic_clearance_score"] * 10)
        + 0.35 * df["technical_mastery"]
        + 0.20 * np.clip(df["experience_index"] * 5, 0, 100)
        + 0.10 * df["attendance_percentage"]
    )

    # 1.5 Exclude non-predictive identifiers and target leakage
    excluded_columns = ["student_id", "placement_status", "salary_package_lpa"]
    X = df.drop(columns=[col for col in excluded_columns if col in df.columns])
    y = df[target_col]

    # 1.6 One-Hot Encode Categorical Features
    categorical_columns = ["gender", "branch", "college_tier", "volunteer_experience"]
    available_cat = [c for c in categorical_columns if c in X.columns]
    X = pd.get_dummies(X, columns=available_cat, drop_first=True)
    print(f"Total features after encoding & engineering: {X.shape[1]}")

    # 1.7 Stratified 80/20 Train-Test Split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split shape: {X_train.shape} | Test split shape: {X_test.shape}")
    print(f"Train target mean: {y_train.mean():.4f} | Test target mean: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test, list(X.columns)


def evaluate_model(model_name: str, y_true, y_pred, y_proba):
    """Compute and format standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy  : {acc * 100:.2f}% ({acc:.4f})")
    print(f"Precision : {prec * 100:.2f}% ({prec:.4f})")
    print(f"Recall    : {rec * 100:.2f}% ({rec:.4f})")
    print(f"F1 Score  : {f1 * 100:.2f}% ({f1:.4f})")
    print(f"ROC-AUC   : {auc * 100:.2f}% ({auc:.4f})")

    return {
        "Model": model_name,
        "Accuracy": round(acc * 100, 3),
        "Precision": round(prec * 100, 3),
        "Recall": round(rec * 100, 3),
        "F1": round(f1 * 100, 3),
        "ROC-AUC": round(auc * 100, 3),
    }


def main():
    data_path = find_dataset()
    X_train, X_test, y_train, y_test, feature_names = step_1_data_cleaning_and_splitting(data_path)

    metrics_summary = []

    # ---------------------------------------------------------
    # MODEL 1: Logistic Regression
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING MODEL 1 - LOGISTIC REGRESSION")
    print("=" * 70)
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    X_test_lr = scaler_lr.transform(X_test)

    model_lr = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    model_lr.fit(X_train_lr, y_train)

    preds_lr = model_lr.predict(X_test_lr)
    probs_lr = model_lr.predict_proba(X_test_lr)[:, 1]
    metrics_summary.append(evaluate_model("Logistic Regression", y_test, preds_lr, probs_lr))

    # ---------------------------------------------------------
    # MODEL 2: K-Nearest Neighbors (KNN with RobustScaler)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING MODEL 2 - K-NEAREST NEIGHBORS (KNN)")
    print("=" * 70)
    scaler_knn = RobustScaler()
    X_train_knn = scaler_knn.fit_transform(X_train)
    X_test_knn = scaler_knn.transform(X_test)

    model_knn = KNeighborsClassifier(n_neighbors=13, weights="distance", p=1)
    model_knn.fit(X_train_knn, y_train)

    preds_knn = model_knn.predict(X_test_knn)
    probs_knn = model_knn.predict_proba(X_test_knn)[:, 1]
    metrics_summary.append(evaluate_model("KNN", y_test, preds_knn, probs_knn))

    # ---------------------------------------------------------
    # MODEL 3: Decision Tree (Pruned & Tuned)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING MODEL 3 - DECISION TREE (PRUNED)")
    print("=" * 70)
    # Decision trees are invariant to monotonic feature scaling
    model_dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=6,
        min_samples_leaf=15,
        ccp_alpha=0.005,
        random_state=42,
    )
    model_dt.fit(X_train, y_train)

    preds_dt = model_dt.predict(X_test)
    probs_dt = model_dt.predict_proba(X_test)[:, 1]
    metrics_summary.append(evaluate_model("Decision Tree", y_test, preds_dt, probs_dt))

    # ---------------------------------------------------------
    # MODEL 4: Linear Support Vector Machine (Linear SVM)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING MODEL 4 - LINEAR SVM (CALIBRATED)")
    print("=" * 70)
    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_train)
    X_test_svm = scaler_svm.transform(X_test)

    base_svm = LinearSVC(C=0.01, max_iter=4000, random_state=42)
    # Platt scaling calibration to guarantee probability outputs
    model_svm = CalibratedClassifierCV(estimator=base_svm, method="sigmoid", cv=5)
    model_svm.fit(X_train_svm, y_train)

    preds_svm = model_svm.predict(X_test_svm)
    probs_svm = model_svm.predict_proba(X_test_svm)[:, 1]
    metrics_summary.append(evaluate_model("Linear SVM", y_test, preds_svm, probs_svm))

    # ---------------------------------------------------------
    # SUMMARY EVALUATION TABLE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL 4-MODEL COMPARISON SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_string(index=False))

    # ---------------------------------------------------------
    # EXPORT MODELS TO backend/ml_models
    # ---------------------------------------------------------
    backend_dir = Path(__file__).resolve().parent.parent / "backend" / "ml_models"
    backend_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting serialized artifacts to: {backend_dir}")

    joblib.dump(model_lr, backend_dir / "logistic_regression.pkl")
    joblib.dump(scaler_lr, backend_dir / "scaler.pkl")

    joblib.dump(model_knn, backend_dir / "knn.pkl")
    joblib.dump(scaler_knn, backend_dir / "scaler_knn.pkl")

    joblib.dump(model_dt, backend_dir / "decision_tree.pkl")

    joblib.dump(model_svm, backend_dir / "svm_linear.pkl")
    joblib.dump(scaler_svm, backend_dir / "scaler_svm_linear.pkl")

    # Also save in current directory for local notebook testing
    local_dir = Path(__file__).resolve().parent / "artifacts"
    local_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_lr, local_dir / "logistic_regression.pkl")
    joblib.dump(scaler_lr, local_dir / "scaler.pkl")
    joblib.dump(model_knn, local_dir / "knn.pkl")
    joblib.dump(scaler_knn, local_dir / "scaler_knn.pkl")
    joblib.dump(model_dt, local_dir / "decision_tree.pkl")
    joblib.dump(model_svm, local_dir / "svm_linear.pkl")
    joblib.dump(scaler_svm, local_dir / "scaler_svm_linear.pkl")

    print("[SUCCESS] All 4 models and scalers successfully trained, validated, and saved!")


if __name__ == "__main__":
    main()
