"""
Lightweight session auth for the school dashboard.

- Session = signed cookie ("school_session") storing user_id.
- Sessions stored in-memory dict (single-process app); restart = logged out.
- Role checks via FastAPI dependencies:
    get_current_user        -> any logged-in user (else 401)
    get_current_user_optional -> user or None
    require_admin           -> admin only (else 403)
    require_admin_or_director -> admin or director
"""

import secrets
from fastapi import Request, HTTPException, Depends
from sqlalchemy.orm import Session
from models import User, get_session


# session_id -> user_id
_sessions = {}

COOKIE_NAME = "school_session"


def create_session(user_id: int) -> str:
    sid = secrets.token_urlsafe(32)
    _sessions[sid] = user_id
    return sid


def destroy_session(sid: str):
    _sessions.pop(sid, None)


def get_user_id_from_session(sid: str):
    return _sessions.get(sid)


def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_session),
):
    """Returns User or None. Use for pages that work for both anon + logged-in."""
    sid = request.cookies.get(COOKIE_NAME)
    if not sid:
        return None
    uid = _sessions.get(sid)
    if uid is None:
        return None
    user = db.query(User).filter(User.id == uid).first()
    return user


def get_current_user(
    request: Request,
    db: Session = Depends(get_session),
):
    """Returns User; raises 401 if not logged in."""
    user = get_current_user_optional(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Login required")
    return user


def require_admin(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def require_admin_or_director(user: User = Depends(get_current_user)):
    if user.role not in ("admin", "director"):
        raise HTTPException(status_code=403, detail="Admin or director only")
    return user


def can_access_class(user: User, class_id: int) -> bool:
    """
    Access rules:
      - admin    -> any class
      - director -> any class (read-only)
      - pedagog  -> only their assigned class
    """
    if user.role in ("admin", "director"):
        return True
    if user.role == "pedagog":
        return user.class_id == class_id
    return False


def can_modify(user: User) -> bool:
    """Director is read-only. Admin + pedagog can modify (pedagog within their class)."""
    return user.role in ("admin", "pedagog")


def user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "class_id": user.class_id,
        "class_name": user.school_class.name if user.school_class else None,
    }
