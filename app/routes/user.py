from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/users", tags=["users"])

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    pet_name: Optional[str] = None
    pet_type: Optional[str] = None


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    pet_name: Optional[str] = None
    pet_type: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    pet_name: Optional[str] = None
    pet_type: Optional[str] = None

# Temporary in-memory storage
fake_users_db: list[dict] = []
next_user_id = 1

# Routes
@router.get("/", response_model=list[UserResponse])
def list_users():
    return fake_users_db


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    for user in fake_users_db:
        if user["id"] == user_id:
            return user
    raise HTTPException(status_code=404, detail="user not found")


@router.post("/", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate):
    global next_user_id

    # simple duplicate email check
    for existing_user in fake_users_db:
        if existing_user["email"] == user.email:
            raise HTTPException(status_code=400, detail="email already exists")

    new_user = {
        "id": next_user_id,
        "username": user.username,
        "email": user.email,
        "pet_name": user.pet_name,
        "pet_type": user.pet_type,
    }
    fake_users_db.append(new_user)
    next_user_id += 1
    return new_user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: int, update: UserUpdate):
    for user in fake_users_db:
        if user["id"] == user_id:
            update_data = update.model_dump(exclude_unset=True)

            # optional duplicate email check
            if "email" in update_data:
                for other_user in fake_users_db:
                    if other_user["id"] != user_id and other_user["email"] == update_data["email"]:
                        raise HTTPException(status_code=400, detail="email already exists")

            user.update(update_data)
            return user

    raise HTTPException(status_code=404, detail="user not found")


@router.delete("/{user_id}")
def delete_user(user_id: int):
    for index, user in enumerate(fake_users_db):
        if user["id"] == user_id:
            deleted_user = fake_users_db.pop(index)
            return {"message": "user deleted successfully", "user": deleted_user}

    raise HTTPException(status_code=404, detail="user not found")
