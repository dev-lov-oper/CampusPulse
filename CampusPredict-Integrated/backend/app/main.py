from datetime import timezone
import os
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .database import Base, engine, get_db
from .models import User, PredictionHistory
from .schemas import RegisterRequest, LoginRequest, PredictRequest, StudentInput
from .auth import hash_password, verify_password, create_access_token, get_user_id
from .ml_service import predict_one, MODEL_FILES

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CampusPredict API", version="1.0.0")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        user_id = get_user_id(authorization[7:])
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.get("/api/health")
def health():
    return {"status": "ok", "models": list(MODEL_FILES.keys())}

@app.get("/api/models")
def model_performance():
    return [
        {"model_name": "logistic_regression", "label": "Logistic Regression", "accuracy": 90.667, "precision": 90.964, "recall": 92.073, "f1": 91.515, "roc_auc": 96.077},
        {"model_name": "knn", "label": "KNN", "accuracy": 91.667, "precision": 88.827, "recall": 96.951, "f1": 92.711, "roc_auc": 96.933},
        {"model_name": "decision_tree", "label": "Decision Tree", "accuracy": 89.667, "precision": 89.349, "recall": 92.073, "f1": 90.691, "roc_auc": 93.728},
        {"model_name": "svm_linear", "label": "Linear SVM", "accuracy": 90.667, "precision": 91.463, "recall": 91.463, "f1": 91.463, "roc_auc": 96.167, "c": 0.01, "score_type": "probability"},
    ]

@app.post("/api/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    email = str(body.email).lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(full_name=body.full_name.strip(), email=email, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": create_access_token(user.id), "token_type": "bearer",
            "user": {"id": user.id, "full_name": user.full_name, "email": user.email}}

@app.post("/api/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    email = str(body.email).lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": create_access_token(user.id), "token_type": "bearer",
            "user": {"id": user.id, "full_name": user.full_name, "email": user.email}}

@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "full_name": user.full_name, "email": user.email}

@app.post("/api/auth/logout")
def logout(user: User = Depends(current_user)):
    return {"message": "Logged out on client. Remove the bearer token."}

def save_prediction(db, user, model_name, payload, prediction, score, score_type):
    record = PredictionHistory(
        user_id=user.id,
        model_name=model_name,
        prediction=prediction,
        prediction_score=score,
        input_data=payload,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

@app.post("/api/predict")
def predict(body: PredictRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    payload = body.model_dump(exclude={"model_name"})
    try:
        prediction, score, score_type = predict_one(body.model_name, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")
    record = save_prediction(db, user, body.model_name, payload, prediction, score, score_type)
    return {
        "id": record.id,
        "model_name": body.model_name,
        "prediction": prediction,
        "score": round(score, 4) if score is not None else None,
        "score_type": score_type,
        "created_at": record.created_at.astimezone(timezone.utc).isoformat(),
    }

@app.post("/api/predict/compare")
def compare(body: StudentInput, user: User = Depends(current_user), db: Session = Depends(get_db)):
    payload = body.model_dump()
    results = []
    for model_name in MODEL_FILES:
        try:
            prediction, score, score_type = predict_one(model_name, payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{model_name} inference failed: {exc}")
        record = save_prediction(db, user, model_name, payload, prediction, score, score_type)
        results.append({
            "id": record.id,
            "model_name": model_name,
            "prediction": prediction,
            "score": round(score, 4) if score is not None else None,
            "score_type": score_type,
            "created_at": record.created_at.astimezone(timezone.utc).isoformat(),
        })
    placed = sum(x["prediction"] == "Placed" for x in results)
    return {
        "predictions": results,
        "majority_prediction": "Placed" if placed >= 2 else "Not Placed",
        "placed_votes": placed,
        "not_placed_votes": 4 - placed,
    }

@app.get("/api/history")
def history(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = (db.query(PredictionHistory)
              .filter(PredictionHistory.user_id == user.id)
              .order_by(PredictionHistory.created_at.desc())
              .all())
    items = []
    for r in rows:
        score = r.prediction_score
        # Older SVM history rows stored a raw decision_function value.
        # Recalculate those rows with the calibrated SVM so history always
        # displays a real probability percentage.
        if r.model_name == "svm_linear":
            try:
                _, score, _ = predict_one("svm_linear", r.input_data)
            except Exception:
                score = None
        items.append({
            "id": r.id, "model_name": r.model_name, "prediction": r.prediction,
            "score": score,
            "score_type": "probability",
            "input_data": r.input_data,
            "created_at": r.created_at.astimezone(timezone.utc).isoformat()
        })
    return items

@app.get("/api/history/{prediction_id}")
def history_item(prediction_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    r = db.query(PredictionHistory).filter(
        PredictionHistory.id == prediction_id,
        PredictionHistory.user_id == user.id
    ).first()
    if not r:
        raise HTTPException(status_code=404, detail="Prediction not found")
    score = r.prediction_score
    if r.model_name == "svm_linear":
        try:
            _, score, _ = predict_one("svm_linear", r.input_data)
        except Exception:
            score = None
    return {
        "id": r.id, "model_name": r.model_name, "prediction": r.prediction,
        "score": score,
        "score_type": "probability",
        "input_data": r.input_data,
        "created_at": r.created_at.astimezone(timezone.utc).isoformat()
    }

@app.put("/api/profile")
def update_profile(full_name: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.full_name = full_name.strip()
    db.commit()
    return {"id": user.id, "full_name": user.full_name, "email": user.email}
