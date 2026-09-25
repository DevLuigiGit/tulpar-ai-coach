"""In-memory token buckets for the public endpoints. The demo is public and every chat turn spends shared LLM quota,
so a single visitor must not be able to burn it. One process = one set of buckets, which fits one Railway replica."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Callable

from fastapi import Depends, HTTPException, Request

from ..config import get_settings
from ..gateway.base import User
from .auth import client_user, trainer_user


class TokenBucket:
    """`rate_per_min` tokens refill per minute up to `burst`; one request takes one token."""

    def __init__(self, rate_per_min: float, burst: int, max_keys: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        self.rate = rate_per_min / 60.0
        self.burst = float(burst)
        self.max_keys = max_keys
        self.clock = clock
        self._state: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def hit(self, key: str) -> float:
        """0.0 if allowed, otherwise seconds until the next token."""
        now = self.clock()
        tokens, last = self._state.pop(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        wait = 0.0
        if tokens >= 1.0:
            tokens -= 1.0
        else:
            wait = (1.0 - tokens) / self.rate if self.rate > 0 else math.inf
        self._state[key] = (tokens, now)
        while len(self._state) > self.max_keys:  # bounded memory under a flood of fresh keys
            self._state.popitem(last=False)
        return wait


_buckets: dict[str, TokenBucket] = {}


def bucket(name: str) -> TokenBucket:
    s = get_settings()
    if name not in _buckets:
        rate, burst = {"chat": (s.chat_rate_per_min, s.chat_burst),
                       "login": (s.login_rate_per_min, s.login_burst),
                       "trainer": (s.trainer_rate_per_min, s.trainer_burst)}[name]
        _buckets[name] = TokenBucket(rate, burst)
    return _buckets[name]


def reset() -> None:
    _buckets.clear()


def client_ip(request: Request) -> str:
    # The right-most X-Forwarded-For entry is the one our own proxy added; the left ones are client-controlled.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd and get_settings().trust_forwarded_for:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _check(name: str, key: str, message: str) -> None:
    if not get_settings().rate_limit_enabled:
        return
    wait = bucket(name).hit(key)
    if wait > 0:
        raise HTTPException(429, message, headers={"Retry-After": str(max(1, math.ceil(wait)))})


async def limit_login(request: Request) -> None:
    _check("login", client_ip(request), "Слишком много входов подряд. Подождите минуту и попробуйте снова.")


async def limit_chat(request: Request, user: User = Depends(client_user)) -> User:
    # The public demo shares one client account, so the key is (user, IP): one visitor cannot lock out the rest.
    _check("chat", f"{user.id}|{client_ip(request)}",
           "Слишком много сообщений подряд. Подождите немного и отправьте снова.")
    return user


async def limit_trainer(request: Request, user: User = Depends(trainer_user)) -> User:
    # The demo lets anyone log in as the trainer, and every draft or edit runs the program builder on shared LLM quota.
    _check("trainer", f"{user.id}|{client_ip(request)}",
           "Слишком много запросов на программы подряд. Подождите минуту и попробуйте снова.")
    return user
