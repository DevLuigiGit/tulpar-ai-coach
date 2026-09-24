import { useCallback, useEffect, useState } from "react";
import { ApiError, errorText, getToken, LOGOUT_EVENT, logout, me } from "./api";
import type { User } from "./types";
import Login from "./pages/Login";
import Spinner from "./components/Spinner";
import { ClientShell, TrainerShell } from "./Shells";

type AuthState =
  | { phase: "anon" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "ready"; user: User };

/**
 * Корень: без токена — экран входа; с токеном — GET /api/me и оболочка по роли.
 * На 401 api.ts стирает токен и шлёт событие tac:logout — возвращаемся ко входу.
 */
export default function App() {
  const [auth, setAuth] = useState<AuthState>(() => (getToken() ? { phase: "loading" } : { phase: "anon" }));

  const loadMe = useCallback(async () => {
    setAuth({ phase: "loading" });
    try {
      const user = await me();
      setAuth({ phase: "ready", user });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setAuth({ phase: "anon" });
      else setAuth({ phase: "error", message: errorText(e) });
    }
  }, []);

  useEffect(() => {
    if (getToken()) void loadMe();
  }, [loadMe]);

  useEffect(() => {
    const onLogout = () => setAuth({ phase: "anon" });
    window.addEventListener(LOGOUT_EVENT, onLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, onLogout);
  }, []);

  const signOut = useCallback(() => {
    window.location.hash = "";
    logout();
  }, []);

  switch (auth.phase) {
    case "anon":
      return (
        <Login
          onLoggedIn={(user) => {
            window.location.hash = "";
            setAuth({ phase: "ready", user });
          }}
        />
      );
    case "loading":
      return (
        <div className="loading-screen">
          <Spinner size="lg" label="Загружаем профиль" />
        </div>
      );
    case "error":
      return (
        <div className="loading-screen">
          <div className="stack center" style={{ maxWidth: 320, padding: 16 }}>
            <h3>Не удалось подключиться</h3>
            <p className="muted small">{auth.message}</p>
            <div className="row" style={{ justifyContent: "center" }}>
              <button className="btn btn-primary" onClick={() => void loadMe()}>
                Повторить
              </button>
              <button className="btn btn-ghost" onClick={signOut}>
                Выйти
              </button>
            </div>
          </div>
        </div>
      );
    case "ready":
      return auth.user.role === "trainer" ? (
        <TrainerShell user={auth.user} onLogout={signOut} />
      ) : (
        <ClientShell user={auth.user} onLogout={signOut} />
      );
  }
}
