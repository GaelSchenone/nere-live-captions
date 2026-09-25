from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import COOKIE_NAME, TOKEN_TTL_SECONDS, create_token, get_current_user, revoke_token, verify_password
from ..db import get_db
from ..models import User

router = APIRouter(prefix="/api")


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/login")
async def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(user.password_hash, req.password):
        raise HTTPException(401, "email o password incorrectos")
    token = create_token(db, user)
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, samesite="lax", max_age=TOKEN_TTL_SECONDS
    )
    return {"ok": True, "email": user.email}


@router.post("/logout")
async def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        revoke_token(db, token)
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {"email": user.email}
