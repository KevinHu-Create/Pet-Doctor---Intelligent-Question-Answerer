from sqlalchemy.orm import Session
from app.db.models import User
from app.services.security import hash_password, verify_password


def create_user(db: Session, username: str, email: str, password: str, role="user"):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise ValueError("username already exists")

    existing_email = db.query(User).filter(User.email == email).first()
    if existing_email:
        raise ValueError("email already exists")

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()

    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user
