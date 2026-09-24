// Общие помощники тренерских экранов: порядок очереди, подписи, склонения.
import { useEffect, useRef } from "react";
import { ApiError } from "../../api";
import { dateShort } from "../../format";
import type { Proposal, ProposalStatus } from "../../types";

/** Статусы, после которых карточка больше не меняется. */
export const FINAL: ProposalStatus[] = ["applied", "rejected", "failed", "resolved"];
export const isFinal = (s: ProposalStatus) => FINAL.includes(s);

/** Ждёт решения тренера: черновик готов или вопрос открыт. */
export const needsDecision = (p: Proposal) => p.status === "pending" || p.status === "open";

function ts(iso: string): number {
  const hasZone = /[zZ]|[+-]\d\d:?\d\d$/.test(iso);
  const t = new Date(hasZone ? iso : iso + "Z").getTime();
  return Number.isNaN(t) ? 0 : t;
}

/** Ранг в очереди: ждущие решения и только что решённые — сверху, черновики в работе — ниже. */
function rank(p: Proposal): number {
  return p.status === "drafting" ? 1 : 0;
}

/** Очередь: сначала pending и открытые эскалации вперемешку по времени (новые выше), затем drafting. */
export function sortQueue(items: Proposal[]): Proposal[] {
  return [...items].sort((a, b) => rank(a) - rank(b) || ts(b.created_at) - ts(a.created_at));
}

export function sortByUpdated(items: Proposal[]): Proposal[] {
  return [...items].sort((a, b) => ts(b.updated_at) - ts(a.updated_at));
}

export const isConflict = (e: unknown) => e instanceof ApiError && e.status === 409;

/** Русское склонение: plural(3, ["клиент", "клиента", "клиентов"]). */
export function plural(n: number, forms: [string, string, string]): string {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return forms[0];
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return forms[1];
  return forms[2];
}

export const SOURCE_TEXT: Record<Proposal["source"], string> = {
  client: "запрос клиента",
  trainer: "запрос тренера",
};

export const OP_TEXT: Record<string, string> = {
  replace_exercise: "замена",
  set_volume: "объём",
  remove_exercise: "убрать",
  add_exercise: "добавить",
};

/** Заголовок причины эскалации по intent маршрутизатора. */
export function escalationHeadline(intent: string | null | undefined): string {
  if (intent === "escalate") return "Красный флаг: здоровье или безопасность";
  if (intent === "question") return "В базе знаний нет надёжного ответа";
  if (intent === "program_request") return "Просьба изменить программу";
  return "Нужен ответ тренера";
}

/** Технические причины маршрутизатора — по-русски; остальное как есть. */
export function reasonText(reason: string | null | undefined): string {
  if (!reason) return "";
  const r = reason.trim();
  if (/^red flag/i.test(r) || /hard red-flag/i.test(r))
    return "Сообщение касается боли, травмы, лекарств или другого медицинского вопроса — бот на такое не отвечает.";
  if (/prompt-injection/i.test(r)) return "Сообщение похоже на попытку обойти правила бота.";
  if (/^heuristic/i.test(r)) return "Модель была недоступна, сообщение разобрано по правилам.";
  return r;
}

/** Ссылка на последнее значение — для колбэков внутри таймеров и опросов. */
export function useLatest<T>(value: T) {
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  });
  return ref;
}

/** Дата «25 авг.» и для полной ISO-строки, и для голой даты YYYY-MM-DD (Safari не парсит «2026-08-25Z»). */
export function dayShort(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (m) {
    const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
    return d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
  }
  return dateShort(iso);
}
