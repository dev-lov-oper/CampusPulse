# CampusPulse — Student Placement Prediction System

A comprehensive machine learning-based platform that predicts student campus placement outcomes using academic records, technical proficiencies, internships, projects, hackathons, aptitude scores, communication skills, and engineered employability indices.

The repository includes both an interactive **Streamlit Application** and a production-ready **Full-Stack Web Platform (React + FastAPI + SQLite)**, evaluated across 4 classical machine learning classification algorithms on a dataset of **100,000 student records**.

---

## 📊 Benchmark Model Performance (100,000 Records)

All models are trained and evaluated on an 80/20 stratified split (`random_state=42`) using domain-specific feature engineering:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Champion Status |
| :--- | :---:| :---:| :---:| :---:| :---:| :---:|
| **Linear SVM (Calibrated)** | **94.15%** | **94.54%** | **94.63%** | **94.58%** | **98.95%** | 🏆 Champion |
| **Logistic Regression** | **94.05%** | **94.53%** | **94.65%** | **94.59%** | **98.95%** | 🏆 Co-Champion |
| **Decision Tree (Pruned)** | **91.36%** | **92.15%** | **92.13%** | **92.14%** | **97.49%** | High Interpretability |
| **KNN (Robust Distance)** | **90.38%** | **90.25%** | **92.50%** | **91.36%** | **97.11%** | Instance-Based |

*(Metrics sourced directly from [model_metrics.csv](file:///c:/Users/hp/Downloads/CampusPredict-Integrated%20(2)/model_metrics.csv) and reproducible via `CampusPredict-Integrated/ml_source/train_models.py`)*

---

## 🎯 Key Objectives & Capabilities

- **Accurate Placement Inference**: Predicts placement likelihood with calibrated confidence probabilities.
- **Model Comparison**: Real-time cross-comparison across all 4 machine learning models on identical inputs.
- **Explainable Metrics**: Complete confusion matrices, ROC-AUC curves, classification reports, and feature importance.
- **Domain Feature Engineering**: Incorporates academic clearance attenuation, technical competency mastery, and composite employability indices.
- **Dual Applications**:
  1. Lightweight, self-contained **Streamlit dashboard** (`app.py`).
  2. Enterprise **React + FastAPI platform** with JWT authentication, user history, and search capabilities.

---

## 🧠 Machine Learning Algorithms

1. **Logistic Regression (`C=0.1`, `max_iter=2000`)**
   - Baseline linear classifier providing interpretable probability-based predictions via sigmoid activation.
   - Scaled using `StandardScaler`.
2. **Support Vector Machine (Linear SVM with Platt Scaling)**
   - Utilizes `LinearSVC(C=0.01)` wrapped in `CalibratedClassifierCV(cv=5, method='sigmoid')` to output well-calibrated probabilities.
3. **Decision Tree Classifier (`max_depth=8`, `min_samples_leaf=20`, `criterion='entropy'`)**
   - Optimized with depth constraining and entropy splitting to balance expressiveness with generalization.
4. **K-Nearest Neighbors (`k=15`, `weights='distance'`, `RobustScaler`)**
   - Uses distance-weighted neighborhood voting with `RobustScaler` to neutralize outlier sensitivity.

*(Note: Ensemble methods like Random Forest are intentionally excluded to focus on classical single-model behavior).*

---

## 🔬 Domain-Specific Feature Engineering

The pipeline engineers 5 domain indicators that elevate raw feature signals:

1. **Academic Clearance Score**: `cgpa * exp(-0.30 * backlogs)` — Models non-linear academic risk from active backlogs.
2. **Strict Clearance Flag**: `(backlogs == 0)` — Binary threshold for strict corporate eligibility criteria.
3. **Technical Mastery Index**: Weighted combination of coding (`40%`), aptitude (`25%`), logical reasoning (`20%`), and mock interview scores (`15%`).
4. **Experience Index**: Weighted combination of internships (`3.5x`), projects (`1.5x`), hackathons (`1.5x`), and certifications (`0.8x`).
5. **Placement Readiness Score**: Composite index integrating academic clearance, technical mastery, practical experience, and attendance.

---

## 📂 Repository Structure

```text
CampusPulse/
├── app.py                                   # Interactive Streamlit Web Application
├── model_metrics.csv                        # Benchmark metrics for all 4 trained models
├── requirements.txt                         # Python dependencies for Streamlit & ML pipeline
├── README.md                                # Repository documentation
│
└── CampusPredict-Integrated/                # Full-Stack Web Platform (React + FastAPI)
    ├── src/                                 # React (Vite) frontend application
    │   ├── main.jsx                         # Application UI, dashboards & charts
    │   ├── styles.css                       # Responsive dark-theme styling
    │   └── services/api.js                  # Axios REST API client
    ├── backend/                             # FastAPI backend service
    │   ├── app/                             # API routes, auth, schemas & database
    │   ├── ml_models/                       # Exported .pkl model & scaler artifacts
    │   ├── requirements.txt                 # FastAPI backend dependencies
    │   └── .env.example                     # Backend environment configuration
    ├── ml_source/                           # Machine Learning source directory
    │   ├── student_placement_prediction_dataset_2026.csv  # 100,000 student records
    │   ├── train_models.py                  # End-to-end training & evaluation pipeline
    │   ├── artifacts/                       # Pre-trained models and matching scalers
    │   └── notebooks/                       # Individual model Jupyter Notebooks
    │       ├── 01_logistic_regression.ipynb
    │       ├── 02_knn_student_placement.ipynb
    │       ├── 03_decision_tree_student_placement.ipynb
    │       └── 04_svm_student_placement_updated.ipynb
    ├── PROJECT_REPORT.md                    # Detailed ML engineering report
    ├── package.json                         # Node frontend dependencies
    └── README.md                            # Integrated platform guide
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** & npm (for the React frontend)
- **Git**

---

### Option A: Run Streamlit Application (Quick Start)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dev-lov-oper/CampusPulse.git
   cd CampusPulse
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

### Option B: Run Integrated Full-Stack Platform (React + FastAPI)

#### 1. Start the FastAPI Backend

```bash
cd CampusPredict-Integrated/backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
- API Endpoint: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

#### 2. Start the React Frontend

Open a new terminal:
```bash
cd CampusPredict-Integrated
npm install
npm run dev
```
Open `http://localhost:5173` to access the full-stack portal.

---

### Option C: Retrain Models or Run Notebooks

To retrain all models and re-export artifacts:
```bash
cd CampusPredict-Integrated/ml_source
python train_models.py
```

The standalone notebooks are located in `CampusPredict-Integrated/ml_source/notebooks/`:
- `01_logistic_regression.ipynb`
- `02_knn_student_placement.ipynb`
- `03_decision_tree_student_placement.ipynb`
- `04_svm_student_placement_updated.ipynb`

---

## 🛡️ Responsible Use & Disclaimer

This software is developed for educational, analytical, and research purposes. Predictions reflect probabilistic patterns within historical training datasets and should not be used as the definitive or sole factor in real-world recruitment decisions.

---

## Take a quick look :
https://campuspulsegit-rvjdwyybmdkm9deb6dwrxe.streamlit.app/
