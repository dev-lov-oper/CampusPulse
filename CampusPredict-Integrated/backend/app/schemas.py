from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict

ModelName = Literal["logistic_regression", "knn", "decision_tree", "svm_linear"]

class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

class StudentInput(BaseModel):
    age: int = Field(ge=18, le=24)
    gender: Literal["Male", "Female"]
    cgpa: float = Field(ge=4.5, le=10)
    branch: Literal["CSE", "Civil", "ECE", "EEE", "IT", "Mechanical"]
    college_tier: Literal["Tier 1", "Tier 2", "Tier 3"]
    internships_count: int = Field(ge=0)
    projects_count: int = Field(ge=0)
    certifications_count: int = Field(ge=0)
    coding_skill_score: float = Field(ge=20, le=100)
    aptitude_score: float = Field(ge=20, le=100)
    communication_skill_score: float = Field(ge=20, le=100)
    logical_reasoning_score: float = Field(ge=20, le=100)
    hackathons_participated: int = Field(ge=0)
    github_repos: int = Field(ge=0)
    linkedin_connections: int = Field(ge=0)
    mock_interview_score: float = Field(ge=20, le=100)
    attendance_percentage: float = Field(ge=50, le=100)
    backlogs: int = Field(ge=0, le=6)
    extracurricular_score: float = Field(ge=0, le=100)
    leadership_score: float = Field(ge=0, le=100)
    volunteer_experience: Literal["Yes", "No"]
    sleep_hours: float = Field(ge=3, le=10)
    study_hours_per_day: float = Field(ge=0.5, le=10)

class PredictRequest(StudentInput):
    model_name: ModelName

class PredictionResponse(BaseModel):
    id: Optional[int] = None
    model_name: str
    prediction: Literal["Placed", "Not Placed"]
    score: Optional[float] = None
    score_type: Literal["probability"]
    created_at: Optional[str] = None

class CompareResponse(BaseModel):
    predictions: list[PredictionResponse]
    majority_prediction: Literal["Placed", "Not Placed"]
    placed_votes: int
    not_placed_votes: int

class HistoryItem(BaseModel):
    id: int
    model_name: str
    prediction: str
    score: Optional[float]
    score_type: str
    input_data: dict
    created_at: str
    model_config = ConfigDict(from_attributes=True)
