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

/** Видимая часть ниже окна больше чем на пиксель — низ перекрыт клавиатурой iOS или её панелью (при внешней
 *  клавиатуре она ~50px). Масштаб в Mini App запрещён, других причин для разницы нет. */
const KEYBOARD_MIN_PX = 1;

/**
 * Сообщаем Telegram, что приложение готово, разворачиваем на весь экран и красим рамку в цвет темы.
 * Вызывается до первого рендера (main.tsx), чтобы раскладка Mini App не прыгала на первом кадре.
 */
export function startTelegram(): void {
  const app = telegramApp();
  if (!app || started) return;
  started = true;
  document.documentElement.classList.add("tg-webapp");
  lockZoom();
  followVisibleViewport();
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

/** Масштаб в Mini App не нужен: щипок или двойной тап увеличивают страницу, и вёрстка уезжает вбок.
 *  Вне Telegram meta не трогаем — там масштабирование остаётся. */
function lockZoom(): void {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="viewport"]');
  if (meta) meta.content = "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover";
}

/**
 * iOS: клавиатура не сжимает окно WebView (и viewportHeight Telegram не меняется) — сжимается только видимая
 * часть (visualViewport), а WebKit сдвигает весь документ вверх, чтобы показать поле ввода: шапка и вкладки
 * уезжают за экран, после закрытия клавиатуры страница может остаться сдвинутой. Поэтому при открытой
 * клавиатуре высота приложения — видимая часть (--tac-kb-height, класс tg-kb), а документ возвращаем в начало.
 * На Android окно сжимается само: visualViewport.height == innerHeight, класс не ставится.
 */
function followVisibleViewport(): void {
  const vv = window.visualViewport;
  if (!vv) return;
  const root = document.documentElement;
  let keyboard = false;
  const sync = () => {
    const open = window.innerHeight - vv.height > KEYBOARD_MIN_PX;
    root.classList.toggle("tg-kb", open);
    if (open) root.style.setProperty("--tac-kb-height", `${Math.floor(vv.height)}px`);
    else root.style.removeProperty("--tac-kb-height");
    if (window.scrollX || window.scrollY || vv.offsetTop) window.scrollTo(0, 0);
    // Приложение стало ниже — поле ввода (например, в карточке тренера) могло уйти под клавиатуру.
    if (open && !keyboard) window.requestAnimationFrame(revealFocused);
    keyboard = open;
  };
  vv.addEventListener("resize", sync);
  vv.addEventListener("scroll", sync);
  window.addEventListener("focusout", () => window.setTimeout(sync, 60));
  sync();
}

function revealFocused(): void {
  const el = document.activeElement;
  if (el instanceof HTMLElement && el.matches("input, textarea, select, [contenteditable]")) {
    el.scrollIntoView({ block: "nearest" });
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
