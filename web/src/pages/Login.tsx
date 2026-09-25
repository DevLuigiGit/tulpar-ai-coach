import { useState } from "react";
import { errorText, login } from "../api";
import type { Role, User } from "../types";
import Spinner from "../components/Spinner";

interface LoginProps {
  onLoggedIn: (user: User) => void;
  /** Ошибка автоматического входа (Telegram Mini App), показывается сразу. */
  notice?: string | null;
}

/** Демо-вход: одна кнопка на роль, без пароля. Токен сохраняет api.login. */
export default function Login({ onLoggedIn, notice }: LoginProps) {
  const [busy, setBusy] = useState<Role | null>(null);
  const [error, setError] = useState<string | null>(notice ?? null);

  const go = async (role: Role) => {
    setBusy(role);
    setError(null);
    try {
      const res = await login(role);
      onLoggedIn(res.user);
    } catch (e) {
      setError(errorText(e));
      setBusy(null);
    }
  };

  return (
    <div className="login">
      <div className="login-card">
        <div className="login-brand">
          <span className="brand-mark" aria-hidden="true">T</span>
          <div className="stack tight">
            <h1 className="login-title">Tulpar AI Coach</h1>
            <p className="login-lead">
              Коуч считает КБЖУ по фото и голосу, отвечает со ссылками на источники, а правки
              программы только предлагает — применяет их тренер.
            </p>
          </div>
        </div>

        <div className="login-actions">
          <button className="btn btn-primary btn-lg login-choice" disabled={!!busy} onClick={() => go("client")}>
            <span className="login-choice-text">
              <span>Войти как клиент</span>
              <span className="login-choice-sub">Чат с коучем и программа</span>
            </span>
            {busy === "client" ? <Spinner size="sm" /> : <Arrow />}
          </button>
          <button className="btn btn-lg login-choice" disabled={!!busy} onClick={() => go("trainer")}>
            <span className="login-choice-text">
              <span>Войти как тренер</span>
              <span className="login-choice-sub muted">Очередь предложений и клиенты</span>
            </span>
            {busy === "trainer" ? <Spinner size="sm" /> : <Arrow />}
          </button>
        </div>

        {error && <div className="notice notice-error">{error}</div>}

        <p className="login-foot">Демо-режим: вход без пароля, данные тестовые.</p>
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
