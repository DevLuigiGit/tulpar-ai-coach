// Telegram Mini App: бот открывает это же веб-приложение, и оно входит само по WebApp.initData
// (POST /api/auth/telegram) — без экрана входа. Вне Telegram модуль ничего не делает.
import { ApiError, getToken, setToken } from "./api";
import type { LoginResponse } from "./types";
import "./telegram.css";

interface TelegramWebApp {
  initData: string;
  initDataUnsafe?: { user?: { id: number; first_name?: string } };
  version?: string;
  ready(): void;
  expand(): void;
  close(): void;
  isVersionAtLeast?(v: string): boolean;
  disableVerticalSwipes?(): void;
  setHeaderColor?(color: string): void;
  setBackgroundColor?(color: string): void;
}

declare global {
  interface Window {
    Telegram?: { WebApp?: TelegramWebApp };
  }
}

const TG_USER_KEY = "tac_tg_user";
const BG = "#0B0B0C";

/** WebApp, только если страница реально открыта из Telegram (initData не пустая). */
export function telegramApp(): TelegramWebApp | null {
  const app = window.Telegram?.WebApp;
  return app && app.initData ? app : null;
}

export const inTelegram = () => telegramApp() !== null;

let started = false;

/** Сообщаем Telegram, что приложение готово, разворачиваем на весь экран и красим рамку в цвет темы. */
export function startTelegram(): void {
  const app = telegramApp();
  if (!app || started) return;
  started = true;
  document.documentElement.classList.add("tg-webapp");
  try {
    app.ready();
    app.expand();
    const at = (v: string) => app.isVersionAtLeast?.(v) ?? false;
    if (at("6.1")) {
      app.setHeaderColor?.(BG);
      app.setBackgroundColor?.(BG);
    }
    // Иначе свайп вниз по ленте чата сворачивает приложение.
    if (at("7.7")) app.disableVerticalSwipes?.();
  } catch {
    /* старый клиент Telegram — работаем как обычная страница */
  }
}

export function closeTelegram(): void {
  telegramApp()?.close();
}

function storedUser(): string | null {
  try {
    return localStorage.getItem(TG_USER_KEY);
  } catch {
    return null;
  }
}

function storeUser(id: string | null): void {
  try {
    if (id) localStorage.setItem(TG_USER_KEY, id);
    else localStorage.removeItem(TG_USER_KEY);
  } catch {
    /* без сохранения войдём заново при следующем открытии */
  }
}

/** Токен есть, выдан этому же пользователю Telegram и не истекает в ближайшую минуту. */
function hasSessionFor(tgUserId: string): boolean {
  const token = getToken();
  if (!token || storedUser() !== tgUserId) return false;
  try {
    const payload = JSON.parse(atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    return typeof payload.exp === "number" && payload.exp * 1000 > Date.now() + 60_000;
  } catch {
    return false;
  }
}

/**
 * Внутри Telegram гарантирует сессию именно этого пользователя Telegram:
 * переиспользует сохранённый токен или входит заново по initData.
 * Бросает ApiError с понятным текстом, если вход не удался.
 */
export async function ensureTelegramSession(force = false): Promise<void> {
  const app = telegramApp();
  if (!app) return;
  const tgUserId = String(app.initDataUnsafe?.user?.id ?? "");
  if (!force && tgUserId && hasSessionFor(tgUserId)) return;

  let res: Response;
  try {
    res = await fetch("/api/auth/telegram", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ init_data: app.initData }),
    });
  } catch (e) {
    throw new ApiError(0, "Нет связи с сервером. Проверьте интернет и откройте приложение снова.", e);
  }
  if (!res.ok) {
    setToken(null);
    storeUser(null);
    throw new ApiError(res.status, telegramErrorText(res.status));
  }
  const data = (await res.json()) as LoginResponse;
  setToken(data.token);
  storeUser(tgUserId || null);
}

function telegramErrorText(status: number): string {
  if (status === 401) return "Telegram не подтвердил вход (данные устарели или повреждены). Закройте приложение и откройте его снова из бота.";
  if (status === 403) return "Этот аккаунт Telegram не привязан к пользователю клуба.";
  if (status === 404) return "Вход через Telegram не настроен на сервере.";
  if (status >= 500) return "Ошибка сервера при входе через Telegram, попробуйте позже.";
  return `Не удалось войти через Telegram (ошибка ${status}).`;
}
