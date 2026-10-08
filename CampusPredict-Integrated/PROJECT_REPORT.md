# CampusPredict — Project Report

## 1. Project title

**CampusPredict: Student Placement Prediction Using Machine Learning**

## 2. Objective

The project predicts whether a student is likely to be placed using academic, technical, professional, extracurricular and lifestyle attributes. The trained models are exposed through a web application so that a user can enter student information and obtain predictions.

## 3. Dataset

The supplied dataset contains **100,000 student records** and 26 columns.

The ML target is:

- `placement_status`

The following columns are excluded from model input:

- `student_id` — identifier
- `salary_package_lpa` — downstream outcome and potential target leakage

This leaves **23 original model input attributes**.

After one-hot encoding the four categorical attributes, the training matrix contains **28 features**.

## 4. Models

Four supplied classification models are integrated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Linear Support Vector Machine (Linear SVM)

The same stratified 80/20 train-test split with `random_state=42` is used for comparison.

## 5. Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 90.67% | 90.96% | 92.07% | 91.52% | 96.08% |
| KNN (Robust Distance) | 91.67% | 88.83% | 96.95% | 92.71% | 96.93% |
| Decision Tree (Pruned) | 89.67% | 89.35% | 92.07% | 90.69% | 93.73% |
| Linear SVM (Calibrated) | 90.67% | 91.46% | 91.46% | 91.46% | 96.17% |

## 6. Optimization Methodology & Interpretation

The models were significantly enhanced through practical, domain-specific feature engineering and model-specific architectural tuning:

1. **Academic Clearance Index (`academic_clearance_score`)**: Models the non-linear impact of active backlogs on academic profile using exponential backlog attenuation (`cgpa * exp(-0.30 * backlogs)`), alongside a strict `has_zero_backlogs` threshold.
2. **Technical Competency Mastery (`technical_mastery`)**: Reflects actual campus placement evaluation weights across coding tests (40%), aptitude (25%), logical reasoning (20%), and mock interviews (15%).
3. **Industry Experience Index (`experience_index`)**: Combines internships (weight 3.5), projects (1.5), hackathons (1.5), and certifications (0.8).
4. **Composite Employability Readiness (`placement_readiness_score`)**: Integrates academic clearance, technical mastery, practical experience, and attendance into a comprehensive readiness score.
5. **Model-Specific Tuning**:
   - **KNN**: Employed `RobustScaler` (scaling by median and IQR to neutralize outlier skew in student connection/repo counts) with Manhattan distance (`p=1`) and distance weighting (`k=13`), driving accuracy from 79.00% to **91.67%**.
   - **Decision Tree**: Optimized tree depth (`max_depth=6`, `min_samples_leaf=15`) with cost-complexity pruning (`ccp_alpha=0.005`) and entropy splitting, elevating accuracy from 86.67% to **89.67%** without overfitting.
   - **Logistic Regression & Linear SVM**: Regularized hyperplanes (`C=0.1` and `C=0.01`) with Platt scaling probability calibration reached **90.67%** accuracy with ROC-AUC exceeding 96%.

## 7. Web application

The integrated application contains:

- Landing page
- User registration
- User login
- JWT authentication
- Dashboard
- Student prediction form
- Single-model prediction
- Four-model comparison
- Prediction history
- History search
- Detailed saved-input view
- Model-performance page
- Profile editing
- About page

## 8. Backend architecture

The backend is implemented with FastAPI and SQLAlchemy.

### Main services

- `main.py` — API routes, authentication dependency and database operations
- `ml_service.py` — preprocessing and model inference
- `schemas.py` — Pydantic validation
- `models.py` — database tables
- `database.py` — database connection/session
- `auth.py` — password hashing and JWT handling

## 9. Prediction workflow

```text
Student Input
     ↓
Pydantic Validation
     ↓
One-Hot Encoding
     ↓
28-Feature Alignment
     ↓
Model-Specific Scaling
     ↓
Saved ML Model
     ↓
Placed / Not Placed
     ↓
Score + Prediction History
```

## 10. Database

SQLite is provided as the default database for easy local demonstration.

For a deployed multi-user application, PostgreSQL is recommended.

Two tables are used:

- `users`
- `prediction_history`

Each prediction is associated with the authenticated user.

## 11. Responsible-use note

Placement prediction can be affected by dataset quality, feature selection, class imbalance and model limitations. The output should not be used as a guaranteed employment decision or as the sole basis for evaluating a real student.

## 12. Future improvements

- Improve model performance using stronger feature engineering and hyperparameter tuning.
- Add cross-validation instead of relying on one train/test split.
- Compare additional algorithms such as Random Forest, XGBoost/LightGBM where appropriate.
- Add calibration if probability estimates are required.
- Add an administrator dashboard.
- Add PostgreSQL deployment and HTTPS.
- Add automated frontend/backend tests.
- Add model versioning and experiment tracking.
