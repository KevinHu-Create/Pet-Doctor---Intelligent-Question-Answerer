from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db.database import get_admin_db, get_db
from app.db.models import Admin, User
from app.services.user_service import authenticate_admin, authenticate_user, create_user

router = APIRouter(tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
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


def _build_auth_user(entity: User | Admin, role: str) -> AuthUserResponse:
    return AuthUserResponse(
        id=entity.id,
        username=entity.username,
        email=entity.email,
        role=role,
        pet_name=getattr(entity, "pet_name", None),
        pet_type=getattr(entity, "pet_type", None),
    )


def _save_session(request: Request, entity_id: int, role: str):
    request.session.clear()
    request.session["user_id"] = entity_id
    request.session["role"] = role


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    try:
        user = create_user(
            db=db,
            username=req.username,
            email=req.email,
            password=req.password,
            pet_name=req.pet_name,
            pet_type=req.pet_type,
        )
        _save_session(request, user.id, "user")
        return AuthResponse(message="user created", user=_build_auth_user(user, "user"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = authenticate_user(db, req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")

    _save_session(request, user.id, "user")
    return AuthResponse(message="login successful", user=_build_auth_user(user, "user"))


@router.post("/admin/login", response_model=AuthResponse)
def admin_login(
    req: LoginRequest,
    request: Request,
    admin_db: Session = Depends(get_admin_db),
):
    admin = authenticate_admin(admin_db, req.username, req.password)
    if not admin:
        raise HTTPException(status_code=401, detail="invalid admin credentials")

    _save_session(request, admin.id, "admin")
    return AuthResponse(
        message="admin login successful",
        user=_build_auth_user(admin, "admin"),
    )


@router.get("/me", response_model=AuthResponse)
def me(
    request: Request,
    db: Session = Depends(get_db),
    admin_db: Session = Depends(get_admin_db),
):
    role = request.session.get("role")
    entity_id = request.session.get("user_id")

    if not role or not entity_id:
        raise HTTPException(status_code=401, detail="authentication required")

    if role == "admin":
        admin = (
            admin_db.query(Admin)
            .filter(Admin.id == entity_id, Admin.is_active.is_(True))
            .first()
        )
        if not admin:
            request.session.clear()
            raise HTTPException(status_code=401, detail="session expired")
        return AuthResponse(
            message="authenticated",
            user=_build_auth_user(admin, "admin"),
        )

    user = db.query(User).filter(User.id == entity_id, User.is_active.is_(True)).first()
    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="session expired")

    return AuthResponse(message="authenticated", user=_build_auth_user(user, "user"))


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"message": "logout successful"}
