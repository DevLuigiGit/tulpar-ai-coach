"""Auth: this service's own short JWT for the web app; internal secret + act-as for the MCP server."""

from __future__ import annotations

import hmac
import time

import jwt
from fastapi import Depends, Header, HTTPException

from ..config import get_settings
from ..gateway import get_gateway
from ..gateway.base import User


def issue_token(user: User, hours: int = 12) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user.id, "role": user.role, "name": user.name, "iat": now, "exp": now + hours * 3600},
                      get_settings().jwt_secret, algorithm="HS256")


async def current_user(authorization: str | None = Header(default=None),
                       x_internal_secret: str | None = Header(default=None),
                       x_act_as: str | None = Header(default=None)) -> User:
    s, gw = get_settings(), get_gateway()
    if x_internal_secret is not None:
        if not hmac.compare_digest(x_internal_secret, s.internal_secret) or not x_act_as:
            raise HTTPException(401, "bad internal credentials")
        user = await gw.get_user(x_act_as)
    elif authorization and authorization.lower().startswith("bearer "):
        try:
            claims = jwt.decode(authorization[7:], s.jwt_secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            raise HTTPException(401, "invalid or expired token")
        user = await gw.get_user(claims["sub"])
    else:
        raise HTTPException(401, "not authenticated")
    if user is None:
        raise HTTPException(401, "unknown user")
    return user


async def client_user(user: User = Depends(current_user)) -> User:
    if user.role != "client":
        raise HTTPException(403, "only for clients")
    return user


async def trainer_user(user: User = Depends(current_user)) -> User:
    if user.role != "trainer":
        raise HTTPException(403, "only for trainers")
    return user
