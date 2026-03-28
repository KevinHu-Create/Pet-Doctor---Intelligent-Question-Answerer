from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db.database import get_admin_db, get_db
from app.db.models import Admin, User


def _session_user_id(request: Request) -> int | None:
    return request.session.get("user_id")


def _session_role(request: Request) -> str | None:
    return request.session.get("role")


def require_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    user_id = _session_user_id(request)
    role = _session_role(request)

    if not user_id or role != "user":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )

    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    if not user:
        request.session.clear()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="session expired",
        )

    return user


def require_current_admin(
    request: Request,
    admin_db: Session = Depends(get_admin_db),
) -> Admin:
    admin_id = _session_user_id(request)
    role = _session_role(request)

    if not admin_id or role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin access required",
        )

    admin = (
        admin_db.query(Admin)
        .filter(Admin.id == admin_id, Admin.is_active.is_(True))
        .first()
    )
    if not admin:
        request.session.clear()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="session expired",
        )

    return admin


def require_active_session(request: Request):
    role = _session_role(request)
    user_id = _session_user_id(request)
    if not role or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )
    return {"user_id": user_id, "role": role}


def require_account_owner_or_admin(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin_db: Session = Depends(get_admin_db),
) -> User | Admin:
    session = require_active_session(request)

    if session["role"] == "admin":
        admin = (
            admin_db.query(Admin)
            .filter(Admin.id == session["user_id"], Admin.is_active.is_(True))
            .first()
        )
        if not admin:
            request.session.clear()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="session expired",
            )
        return admin

    user = db.query(User).filter(User.id == session["user_id"], User.is_active.is_(True)).first()
    if not user:
        request.session.clear()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="session expired",
        )

    if user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="you can only access your own account",
        )
    return user
