// Типизированные обёртки над REST API. Токен живёт в localStorage "tac_token".
// На 401 токен стирается и на window рассылается событие "tac:logout".
import type {
  ChatMessage,
  ChatReply,
  ClientContext,
  ClientSummary,
  ConfirmMealRequest,
  ConfirmMealResponse,
  DecisionAction,
  FeedbackRating,
  FeedbackResult,
  LoginResponse,
  Plan,
  Proposal,
  Role,
  TrainerFeedback,
  User,
} from "./types";

export const TOKEN_KEY = "tac_token";
export const LOGOUT_EVENT = "tac:logout";
/** Ответ коуча и решение тренера могут идти до 90 с — ждём 120 с. */
export const LONG_TIMEOUT_MS = 120_000;
const DEFAULT_TIMEOUT_MS = 30_000;

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* приватный режим — живём без сохранения */
  }
}

export function logout(): void {
  setToken(null);
  window.dispatchEvent(new Event(LOGOUT_EVENT));
}

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** Человекочитаемое сообщение для любой ошибки запроса. */
export function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Что-то пошло не так";
}

interface RequestOptions {
  method?: string;
  json?: unknown;
  form?: FormData;
  timeoutMs?: number;
  /** Вернуть тело как Blob (аудио), а не JSON. */
  blob?: boolean;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: opts.blob ? "*/*" : "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let body: BodyInit | undefined;
  if (opts.form) body = opts.form;
  else if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }

  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(path, {
      method: opts.method ?? (body ? "POST" : "GET"),
      headers,
      body,
      signal: ctrl.signal,
    });
  } catch (e) {
    if (ctrl.signal.aborted) throw new ApiError(0, "Сервер не ответил вовремя. Попробуйте ещё раз.");
    throw new ApiError(0, "Нет связи с сервером", e);
  } finally {
    window.clearTimeout(timer);
  }

  if (res.status === 401) {
    logout();
    throw new ApiError(401, "Сессия истекла, войдите снова");
  }
  if (opts.blob && res.ok) return (await res.blob()) as T;
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) {
    throw new ApiError(res.status, describeError(res.status, data), data);
  }
  return data as T;
}

function describeError(status: number, data: unknown): string {
  const detail =
    data && typeof data === "object" && "detail" in data ? (data as { detail: unknown }).detail : null;
  if (typeof detail === "string" && detail) return detail;
  if (status === 409) return "Состояние уже изменилось — обновите страницу";
  if (status === 404) return "Не найдено";
  if (status === 413) return "Файл слишком большой (до 8 МБ)";
  if (status === 422) return "Проверьте введённые данные";
  if (status >= 500) return "Ошибка сервера, попробуйте позже";
  return `Ошибка ${status}`;
}

// ---------- Авторизация ----------

export async function login(role: Role): Promise<LoginResponse> {
  const res = await request<LoginResponse>("/api/auth/demo-login", { json: { role } });
  setToken(res.token);
  return res;
}

export const me = () => request<User>("/api/me");

// ---------- Клиент ----------

export interface ChatInput {
  text?: string;
  photo?: File | Blob | null;
  audio?: File | Blob | null;
}

/** Собирает FormData для POST /api/chat из текста, фото и/или голосового. */
export function chatForm(input: ChatInput): FormData {
  const fd = new FormData();
  if (input.text) fd.append("text", input.text);
  if (input.photo) fd.append("photo", input.photo, fileName(input.photo, "photo.jpg"));
  if (input.audio) fd.append("audio", input.audio, fileName(input.audio, "voice.webm"));
  return fd;
}

function fileName(f: File | Blob, fallback: string): string {
  return f instanceof File && f.name ? f.name : fallback;
}

export const chat = (form: FormData) =>
  request<ChatReply>("/api/chat", { form, timeoutMs: LONG_TIMEOUT_MS });

/** Этап, на котором сейчас граф: «route» — разбор сообщения, «search» — поиск по базе знаний и т. д. */
export type ChatStage = "listen" | "route" | "search" | "answer" | "meal" | "photo" | "program";

const INTERRUPTED = { interrupted: true };

/** Связь оборвалась, когда сервер уже принял сообщение: ответ сохранится и появится в истории. */
export const isInterrupted = (e: unknown) => e instanceof ApiError && e.detail === INTERRUPTED;

/**
 * Тот же запрос, что chat(), через POST /api/chat/stream: пока граф работает, приходят этапы (onStage), потом
 * тот же ответ целиком. Сам текст не стримится — это замерили, выигрыша нет. Если потокового адреса нет (старый
 * сервер), отправляет обычный запрос.
 */
export async function chatStreamed(form: FormData, onStage: (stage: ChatStage) => void): Promise<ChatReply> {
  const headers: Record<string, string> = { Accept: "text/event-stream" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), LONG_TIMEOUT_MS);
  try {
    let res: Response;
    try {
      res = await fetch("/api/chat/stream", { method: "POST", headers, body: form, signal: ctrl.signal });
    } catch (e) {
      if (ctrl.signal.aborted) throw new ApiError(0, "Сервер не ответил вовремя. Попробуйте ещё раз.");
      throw new ApiError(0, "Нет связи с сервером", e);
    }
    // Нет такого адреса (старый сервер) — сообщение не принято, можно отправить обычным запросом.
    if (res.status === 404 || res.status === 405) return await chat(form);
    if (res.status === 401) {
      logout();
      throw new ApiError(401, "Сессия истекла, войдите снова");
    }
    if (!res.ok) {
      const text = await res.text();
      let data: unknown = text;
      try {
        data = JSON.parse(text);
      } catch {
        /* текст ошибки как есть */
      }
      throw new ApiError(res.status, describeError(res.status, data), data);
    }
    let buf = "";
    // Разбирает готовые события из буфера; возвращает ответ, когда пришёл «done».
    const drain = (): ChatReply | null => {
      let cut: number;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, cut).split("\n").find((l) => l.startsWith("data: "));
        buf = buf.slice(cut + 2);
        if (!line) continue;
        const ev = JSON.parse(line.slice(6)) as { type: string; stage?: ChatStage; reply?: ChatReply; detail?: string };
        if (ev.type === "stage" && ev.stage) onStage(ev.stage);
        else if (ev.type === "done" && ev.reply) return ev.reply;
        else if (ev.type === "error") throw new ApiError(500, ev.detail || "Ошибка сервера, попробуйте позже", ev);
      }
      return null;
    };
    try {
      if (!res.body) {
        // WebView без потокового чтения: тот же ответ целиком, когда запрос завершится (повторно не отправляем —
        // сервер сообщение уже принял).
        buf = await res.text();
        const reply = drain();
        if (reply) return reply;
      } else {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const reply = drain();
          if (reply) return reply;
        }
      }
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError(0, "Связь прервалась — ответ появится в чате после обновления", INTERRUPTED);
    }
    throw new ApiError(0, "Связь прервалась — ответ появится в чате после обновления", INTERRUPTED);
  } finally {
    window.clearTimeout(timer);
  }
}

/** Озвучка ответа коуча: MP3 (audio/mpeg). */
export const speech = (text: string) => request<Blob>("/api/tts", { json: { text }, blob: true });

export const history = (limit = 50) => request<ChatMessage[]>(`/api/chat/history?limit=${limit}`);

export const confirmMeal = (cardId: string, body: ConfirmMealRequest) =>
  request<ConfirmMealResponse>(`/api/meals/${encodeURIComponent(cardId)}/confirm`, { json: body });

export const rateMessage = (messageId: number, rating: FeedbackRating, comment?: string) =>
  request<FeedbackResult>(`/api/messages/${messageId}/feedback`, {
    json: comment ? { rating, comment } : { rating },
  });

export const myPlan = () => request<Plan | null>("/api/my/plan");

export const myProposals = () => request<Proposal[]>("/api/my/proposals");

// ---------- Тренер ----------

export const clients = () => request<ClientSummary[]>("/api/trainer/clients");

export const trainerFeedback = (rating?: FeedbackRating, limit = 50) =>
  request<TrainerFeedback>(`/api/trainer/feedback?limit=${limit}${rating ? `&rating=${rating}` : ""}`);

export const clientContext = (clientId: string) =>
  request<ClientContext>(`/api/trainer/clients/${encodeURIComponent(clientId)}/context`);

export const clientChat = (clientId: string, limit = 50) =>
  request<ChatMessage[]>(`/api/trainer/clients/${encodeURIComponent(clientId)}/chat?limit=${limit}`);

/** Тренер просит ИИ подготовить правку; вернётся Proposal в статусе drafting. */
export const requestChange = (clientId: string, text: string) =>
  request<Proposal>("/api/trainer/proposals", { json: { client_id: clientId, request: text } });

export const queue = (historyMode = false) =>
  request<Proposal[]>(`/api/queue?history=${historyMode ? "true" : "false"}`);

export const proposal = (id: string) => request<Proposal>(`/api/proposals/${encodeURIComponent(id)}`);

export const decide = (id: string, action: DecisionAction, comment?: string) =>
  request<Proposal>(`/api/proposals/${encodeURIComponent(id)}/decision`, {
    json: comment ? { action, comment } : { action },
    timeoutMs: LONG_TIMEOUT_MS,
  });

export const resolveEscalation = (id: string, reply?: string) =>
  request<Proposal>(`/api/escalations/${encodeURIComponent(id)}/resolve`, {
    json: reply ? { reply } : {},
  });

// ---------- Утилиты ----------

/** Опрашивает предложение, пока оно в drafting. Возвращает функцию остановки. */
export function pollProposal(
  id: string,
  onUpdate: (p: Proposal) => void,
  intervalMs = 3000,
): () => void {
  let stopped = false;
  let timer = 0;
  const tick = async () => {
    if (stopped) return;
    try {
      const p = await proposal(id);
      if (stopped) return;
      onUpdate(p);
      if (p.status !== "drafting") return;
    } catch {
      /* сеть моргнула — пробуем снова */
    }
    if (!stopped) timer = window.setTimeout(tick, intervalMs);
  };
  timer = window.setTimeout(tick, intervalMs);
  return () => {
    stopped = true;
    window.clearTimeout(timer);
  };
}
