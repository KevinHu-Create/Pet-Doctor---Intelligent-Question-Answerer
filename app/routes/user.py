from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.services.security import hash_password

router = APIRouter(prefix="/users", tags=["users"])


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    pet_name: Optional[str] = None
    pet_type: Optional[str] = None


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    pet_name: Optional[str] = None
    pet_type: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    pet_name: Optional[str] = None
    pet_type: Optional[str] = None

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="user not found")

    return user


@router.post("/", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing_user_by_email = db.query(User).filter(User.email == user.email).first()
    if existing_user_by_email:
        raise HTTPException(status_code=400, detail="email already exists")

    existing_user_by_username = db.query(User).filter(User.username == user.username).first()
    if existing_user_by_username:
        raise HTTPException(status_code=400, detail="username already exists")

    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=hash_password(user.password),
        pet_name=user.pet_name,
        pet_type=user.pet_type,
        role="user",
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: int, update: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="user not found")

    update_data = update.model_dump(exclude_unset=True)

    if "email" in update_data:
        existing_user = (
            db.query(User)
            .filter(User.email == update_data["email"], User.id != user_id)
            .first()
        )
        if existing_user:
            raise HTTPException(status_code=400, detail="email already exists")

    if "username" in update_data:
        existing_user = (
            db.query(User)
            .filter(User.username == update_data["username"], User.id != user_id)
            .first()
        )
        if existing_user:
            raise HTTPException(status_code=400, detail="username already exists")

    if "username" in update_data:
        user.username = update_data["username"]

    if "email" in update_data:
        user.email = update_data["email"]

    if "pet_name" in update_data:
        user.pet_name = update_data["pet_name"]

    if "pet_type" in update_data:
        user.pet_type = update_data["pet_type"]

    if "password" in update_data:
        user.password_hash = hash_password(update_data["password"])

    db.commit()
    db.refresh(user)

    return user

@router.delete("/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="user not found")

    db.delete(user)
    db.commit()

    return {"message": "user deleted successfully"}
