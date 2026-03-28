from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.user_service import create_user, authenticate_user

router = APIRouter(tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "user"
    pet_name: str | None = None
    pet_type: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthUserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    pet_name: str | None = None
    pet_type: str | None = None

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    message: str
    user: AuthUserResponse


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = create_user(
            db=db,
            username=req.username,
            email=req.email,
            password=req.password,
            role=req.role,
            pet_name=req.pet_name,
            pet_type=req.pet_type,
        )
        return AuthResponse(message="user created", user=user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")

    return AuthResponse(message="login successful", user=user)
