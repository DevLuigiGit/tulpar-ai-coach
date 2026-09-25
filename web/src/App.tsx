import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, errorText, getToken, LOGOUT_EVENT, logout, me } from "./api";
import { closeTelegram, ensureTelegramSession, inTelegram, startTelegram } from "./telegram";
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
 * Внутри Telegram Mini App вход автоматический (telegram.ts): экран входа — только если он не удался.
 */
export default function App() {
  const [auth, setAuth] = useState<AuthState>(() =>
    getToken() || inTelegram() ? { phase: "loading" } : { phase: "anon" },
  );
  const [tgError, setTgError] = useState<string | null>(null);
  const tgRetried = useRef(false);

  const loadMe = useCallback(async () => {
    setAuth({ phase: "loading" });
    try {
      const user = await me();
      tgRetried.current = false;
      setAuth({ phase: "ready", user });
    } catch (e) {
      // In Telegram a 401 is handled by the tac:logout listener (re-auth via initData).
      if (e instanceof ApiError && e.status === 401) {
        if (!inTelegram()) setAuth({ phase: "anon" });
      } else setAuth({ phase: "error", message: errorText(e) });
    }
  }, []);

  const signInTelegram = useCallback(
    async (force: boolean) => {
      setAuth({ phase: "loading" });
      try {
        await ensureTelegramSession(force);
      } catch (e) {
        setTgError(errorText(e));
        setAuth({ phase: "anon" });
        return;
      }
      await loadMe();
    },
    [loadMe],
  );

  useEffect(() => {
    if (inTelegram()) {
      startTelegram();
      void signInTelegram(false);
    } else if (getToken()) void loadMe();
  }, [loadMe, signInTelegram]);

  useEffect(() => {
    const onLogout = () => {
      if (inTelegram() && !tgRetried.current) {
        tgRetried.current = true;
        void signInTelegram(true);
        return;
      }
      if (inTelegram()) setTgError("Сессия не принята сервером. Закройте приложение и откройте его снова из бота.");
      setAuth({ phase: "anon" });
    };
    window.addEventListener(LOGOUT_EVENT, onLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, onLogout);
  }, [signInTelegram]);

  const signOut = useCallback(() => {
    // Inside Telegram the account is the Telegram user: "Выйти" closes the Mini App.
    if (inTelegram()) return closeTelegram();
    window.location.hash = "";
    logout();
  }, []);

  switch (auth.phase) {
    case "anon":
      return (
        <Login
          notice={tgError}
          onLoggedIn={(user) => {
            window.location.hash = "";
            setAuth({ phase: "ready", user });
          }}
        />
      );
    case "loading":
      return (
        <div className="loading-screen">
          <Spinner size="lg" label={inTelegram() ? "Входим через Telegram" : "Загружаем профиль"} />
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
