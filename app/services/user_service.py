from sqlalchemy.orm import Session
from app.db.models import Admin, User
from app.core.settings import settings
from app.services.security import hash_password, verify_password


def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    pet_name: str | None = None,
    pet_type: str | None = None,
):
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
        role="user",
        pet_name=pet_name,
        pet_type=pet_type,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str):
    user = (
        db.query(User)
        .filter(User.username == username, User.is_active.is_(True))
        .first()
    )

    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


def authenticate_admin(db: Session, username: str, password: str):
    admin = (
        db.query(Admin)
        .filter(Admin.username == username, Admin.is_active.is_(True))
        .first()
    )

    if not admin:
        return None

    if not verify_password(password, admin.password_hash):
        return None

    return admin


def seed_admin(db: Session):
    existing_admin = db.query(Admin).filter(Admin.username == settings.admin_seed_username).first()
    if existing_admin:
        if existing_admin.email == "admin@petdoctor.local":
            existing_admin.email = settings.admin_seed_email
            db.commit()
            db.refresh(existing_admin)
        return existing_admin

    existing_admin = db.query(Admin).filter(Admin.email == settings.admin_seed_email).first()
    if existing_admin:
        return existing_admin

    admin = Admin(
        username=settings.admin_seed_username,
        email=settings.admin_seed_email,
        password_hash=hash_password(settings.admin_seed_password),
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin
