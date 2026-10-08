# CampusPredict — Integrated Student Placement Prediction System

CampusPredict is a React + FastAPI web application that uses four supplied machine-learning classification models to estimate student placement status.

## Included ML models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 90.67% | 90.96% | 92.07% | 91.52% | 96.08% |
| KNN (Robust Distance) | 91.67% | 88.83% | 96.95% | 92.71% | 96.93% |
| Decision Tree (Pruned) | 89.67% | 89.35% | 92.07% | 90.69% | 93.73% |
| Linear SVM (Calibrated) | 90.67% | 91.46% | 91.46% | 91.46% | 96.17% |

Metrics are from the held-out 20% test split (`random_state=42`, stratified). All models expose calibrated probabilities.

## What is included

- React/Vite frontend with responsive dark UI
- FastAPI backend
- JWT authentication
- Bcrypt password hashing
- SQLite by default; PostgreSQL configuration supported
- Four saved `.pkl` models and matching scalers
- 23 student input fields transformed into the trained 33-feature format with domain engineering
- Single-model prediction
- Compare-all prediction
- User-specific prediction history
- History search and detailed input view
- Model-performance page with finalized SVM metrics
- Swagger/OpenAPI backend documentation
- Original notebooks and dataset under `ml_source/`
- `PROJECT_REPORT.md` for submission/documentation

## Project structure

```text
CampusPredict-Integrated/
├── src/
│   ├── main.jsx
│   ├── styles.css
│   └── services/api.js
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── ml_service.py
│   │   ├── schemas.py
│   │   ├── models.py
│   │   ├── database.py
│   │   └── auth.py
│   ├── ml_models/
│   ├── requirements.txt
│   └── .env.example
├── ml_source/
│   ├── student_placement_prediction_dataset_2026.csv
│   └── notebooks/
├── .env.example
├── PROJECT_REPORT.md
├── index.html
└── package.json
```

## Run locally

### 1. Backend

Open a terminal:

```bash
cd backend
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and set a strong `JWT_SECRET`.

Start the API:

```bash
uvicorn app.main:app --reload
```

Backend:
`http://127.0.0.1:8000`

Swagger:
`http://127.0.0.1:8000/docs`

### 2. Frontend

Open a second terminal from the project root:

```bash
npm install
npm run dev
```

The default API URL is:

```text
http://127.0.0.1:8000/api
```

To change it, create `.env` in the project root:

```env
VITE_API_URL=http://127.0.0.1:8000/api
```

## Backend API

- `GET /api/health`
- `GET /api/models`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/predict`
- `POST /api/predict/compare`
- `GET /api/history`
- `GET /api/history/{id}`
- `PUT /api/profile`

## ML preprocessing

The backend reproduces the notebook preprocessing:

1. Start with the 23 user-facing student attributes.
2. One-hot encode `gender`, `branch`, `college_tier`, and `volunteer_experience` using `drop_first=True`.
3. Align columns to the saved 28-feature training order.
4. Scale Logistic Regression, KNN and Linear SVM using their matching saved scalers.
5. Do not scale Decision Tree.
6. Predict the binary target and convert it to `Placed` / `Not Placed`.

The target and leakage fields are never accepted from the web form:

- `student_id`
- `placement_status`
- `salary_package_lpa`

## Validation performed on the packaged version

- Python backend source files compile successfully.
- All four saved ML models load successfully.
- A sample 23-field input was successfully processed by all four models.
- The sample produced predictions without feature-order errors.
- The saved models reproduce the displayed test metrics using the supplied dataset and test split.
- React source integrity was checked after the UI changes.

A full Vite production build could not be executed in this environment because `npm install` timed out while fetching packages. Run `npm install` and `npm run build` on your local machine before final deployment.

## Security/deployment notes

For a real deployment:

- Replace the default JWT secret with a long random secret.
- Set `CORS_ORIGINS` to the exact deployed frontend origin(s).
- Use PostgreSQL instead of SQLite for multi-user deployment.
- Serve the frontend over HTTPS.
- Do not commit `.env` files or real credentials.
- The ML result should be described as a prediction/estimate, not a guaranteed placement decision.
