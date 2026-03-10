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

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = create_user(
            db=db,
            username=req.username,
            email=req.email,
            password=req.password,
            role=req.role,
        )
        return {
            "message": "user created",
            "username": user.username,
            "role": user.role,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")

    return {
        "message": "login successful",
        "username": user.username,
        "role": user.role,
    }