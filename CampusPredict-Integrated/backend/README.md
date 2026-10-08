# CampusPredict FastAPI Backend

## Run

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API: http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs

The backend loads the four supplied trained models and their matching scalers. It does not retrain them.

Default database is SQLite for easy local setup. Set `DATABASE_URL` to PostgreSQL for the project deployment.

## ML inference

The supplied notebooks use:
- remove `student_id`, `placement_status`, `salary_package_lpa`
- `pd.get_dummies(..., drop_first=True)` for categorical features
- align to the trained 28 feature columns
- scale Logistic Regression, KNN and Linear SVM with their supplied scalers
- do not scale Decision Tree
- Linear SVM returns a `model_score`, not a probability

## Endpoints

- POST `/api/auth/register`
- POST `/api/auth/login`
- GET `/api/auth/me`
- POST `/api/predict`
- POST `/api/predict/compare`
- GET `/api/history`
- GET `/api/history/{id}`
- PUT `/api/profile`
