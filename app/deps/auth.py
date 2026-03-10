from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.user_service import get_user_by_username

def get_current_user(
    x_username: str = Header(default=""),
    db: Session = Depends(get_db),
):
    if not x_username:
        raise HTTPException(status_code=401, detail="missing X-Username header")

    user = get_user_by_username(db, x_username)
    if not user:
        raise HTTPException(status_code=401, detail="invalid user")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="inactive user")

    return user

def require_role(required_role: str):
    def checker(user = Depends(get_current_user)):
        if user.role != required_role:
            raise HTTPException(status_code=403, detail="not enough permissions")
        return user
    return checker