// Разбор сообщений чата: какой payload у сообщения, вложения пользователя,
// какие карточки еды уже записаны. Чистые функции — без React.
import type { ChatMessage, ChatReply, MessagePayload, ReplyPayload, SystemPayload } from "../../types";

const REPLY_KINDS = new Set(["answer", "meal_card", "escalated", "proposal", "refusal", "info"]);

export function asReply(p: MessagePayload | null): ReplyPayload | null {
  if (!p || typeof p !== "object") return null;
  const kind = (p as { kind?: unknown }).kind;
  return typeof kind === "string" && REPLY_KINDS.has(kind) ? (p as ReplyPayload) : null;
}

export function asSystem(p: MessagePayload | null): SystemPayload["kind"] | null {
  if (!p || typeof p !== "object") return null;
  const kind = (p as { kind?: unknown }).kind;
  return kind === "meal_logged" || kind === "trainer_reply" ? kind : null;
}

/** Вложения сообщения пользователя: сервер кладёт {has_photo, has_audio} в payload. */
export function attachments(m: ChatMessage): { photo: boolean; audio: boolean } {
  const p = (m.payload ?? {}) as Record<string, unknown>;
  return { photo: p.has_photo === true, audio: p.has_audio === true };
}

/** Текст пользователя без служебных заглушек «[фото]» / «[голосовое]». */
export function userText(m: ChatMessage): string {
  const t = m.text.trim();
  return t === "[фото]" || t === "[голосовое]" ? "" : t;
}

/**
 * Карточки, которые уже записаны. Подтверждение добавляет в чат заметку meal_logged —
 * не обязательно сразу после карточки (между ними могут быть другие сообщения).
 * Если в заметке есть card_id — берём его; иначе относим заметку к последней ещё не
 * записанной карточке с позициями (пустую карточку записать нельзя). Если эвристика
 * промахнётся, сервер ответит already=true и карточка покажет «Уже записано».
 */
export function loggedCardIds(messages: ChatMessage[]): Set<string> {
  const out = new Set<string>();
  const open: string[] = [];
  for (const m of messages) {
    const card = asReply(m.payload)?.meal;
    if (card?.card_id && card.items.length > 0) open.push(card.card_id);
    if (asSystem(m.payload) !== "meal_logged") continue;
    const explicit = (m.payload as { card_id?: unknown }).card_id;
    const id = typeof explicit === "string" ? explicit : open[open.length - 1];
    if (!id) continue;
    out.add(id);
    const at = open.lastIndexOf(id);
    if (at >= 0) open.splice(at, 1);
  }
  return out;
}

/** Распознанный текст голосового: лежит в payload ответа сразу после сообщения. */
export function transcriptAfter(messages: ChatMessage[], index: number): string | null {
  const next = messages[index + 1];
  if (!next || next.role !== "assistant") return null;
  return asReply(next.payload)?.transcript ?? null;
}

/** Совпадают ли два снимка истории — чтобы не перерисовывать ленту зря. */
export function sameHistory(a: ChatMessage[], b: ChatMessage[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].id !== b[i].id || a[i].text !== b[i].text) return false;
  }
  return true;
}

let localSeq = 0;

/** Локальное сообщение, пока сервер не вернул историю с настоящими id. */
export function localMessage(
  role: ChatMessage["role"],
  text: string,
  payload: MessagePayload | Record<string, unknown> | null,
): ChatMessage {
  localSeq += 1;
  return {
    id: `local-${Date.now()}-${localSeq}`,
    client_id: "",
    role,
    text,
    payload: payload as MessagePayload | null,
    created_at: new Date().toISOString(),
  };
}

/** Ответ POST /api/chat в форме сообщения истории. С message_id сразу берём серверный id —
 *  оценку можно поставить, не дожидаясь обновления истории. */
export function replyToMessage(r: ChatReply): ChatMessage {
  const { reply, message_id, ...payload } = r;
  const m = localMessage("assistant", reply, payload);
  return typeof message_id === "number" ? { ...m, id: message_id } : m;
}

/** Оценивать можно ответы коуча, уже сохранённые на сервере (числовой id). */
export function canRate(m: ChatMessage): boolean {
  return m.role === "assistant" && typeof m.id === "number" && asReply(m.payload) !== null;
}

/** Сколько миллисекунд назад создано сообщение (время без зоны считаем UTC). */
export function ageMs(iso: string): number {
  const hasZone = /[zZ]|[+-]\d\d:?\d\d$/.test(iso);
  return Date.now() - Date.parse(hasZone ? iso : iso + "Z");
}

export const isLocal = (m: ChatMessage) => typeof m.id === "string" && m.id.startsWith("local-");

/** Ключ сообщения для словарей (превью фото и т. п.): id сервера — число. */
export const msgKey = (m: ChatMessage) => String(m.id);

export const MAX_FILE_BYTES = 8 * 1024 * 1024;

export function fileSizeLabel(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1).replace(".", ",")} МБ`;
}
