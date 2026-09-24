// Общие форматтеры и русские подписи. Используют и клиентские, и тренерские экраны.
import type { MealItem, MealSlot, ProposalStatus, ReplyKind } from "./types";

export const STATUS_LABEL: Record<ProposalStatus, string> = {
  drafting: "Готовится",
  pending: "Ждёт решения",
  applied: "Применено",
  rejected: "Отклонено",
  failed: "Ошибка",
  open: "Нужен тренер",
  resolved: "Закрыто",
};

export const KIND_LABEL: Record<ReplyKind, string> = {
  answer: "Ответ",
  meal_card: "Приём пищи",
  escalated: "Передано тренеру",
  proposal: "Предложение",
  refusal: "Вне темы",
  info: "Справка",
};

export const MEAL_LABEL: Record<MealSlot, string> = {
  breakfast: "Завтрак",
  lunch: "Обед",
  dinner: "Ужин",
  snack: "Перекус",
};

/** Приём пищи по текущему времени — подставляется по умолчанию. */
export function mealByTime(d = new Date()): MealSlot {
  const h = d.getHours();
  if (h < 11) return "breakfast";
  if (h < 16) return "lunch";
  if (h < 21) return "dinner";
  return "snack";
}

export const SOURCE_LABEL: Record<string, string> = {
  exercises: "Каталог упражнений",
  nutrition: "Справочник питания",
  who2020: "ВОЗ, 2020",
};

const GOAL: Record<string, string> = {
  cut: "Сушка",
  keep: "Поддержание",
  gain: "Набор массы",
  lose_weight: "Снижение веса",
  weight_loss: "Снижение веса",
  fat_loss: "Снижение веса",
  gain_muscle: "Набор массы",
  muscle_gain: "Набор массы",
  maintain: "Поддержание",
  health: "Здоровье",
  strength: "Сила",
  endurance: "Выносливость",
};
const LEVEL: Record<string, string> = {
  beginner: "Новичок",
  inter: "Средний",
  adv: "Продвинутый",
  intermediate: "Средний",
  advanced: "Продвинутый",
};
const ACTIVITY: Record<string, string> = {
  sedentary: "Сидячая",
  light: "Низкая",
  moderate: "Средняя",
  high: "Высокая",
  very_high: "Очень высокая",
};
const SEX: Record<string, string> = { male: "Мужской", female: "Женский" };
const PLACE: Record<string, string> = {
  gym: "Зал",
  home: "Дом",
  outdoor: "Улица",
};

const EQUIPMENT: Record<string, string> = {
  barbell: "штанга",
  dumbbell: "гантели",
  bodyweight: "свой вес",
  machine: "тренажёр",
  cable: "блок",
  kettlebell: "гиря",
};

const pick = (dict: Record<string, string>, v: string | null | undefined) =>
  v ? dict[v] ?? v : "—";
export const goalLabel = (v?: string | null) => pick(GOAL, v);
export const levelLabel = (v?: string | null) => pick(LEVEL, v);
export const placeLabel = (v?: string | null) => pick(PLACE, v);
export const activityLabel = (v?: string | null) => pick(ACTIVITY, v);
export const equipmentLabel = (v?: string | null) => pick(EQUIPMENT, v);
export const sexLabel = (v?: string | null) => pick(SEX, v);

const WEEKDAYS_SHORT = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
const WEEKDAYS_FULL = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"];

/** День недели: число 0–6 (0 = понедельник) или 1–7, либо строка как есть. */
export function weekdayLabel(w: number | string | null | undefined, full = false): string {
  if (w === null || w === undefined || w === "") return "";
  const list = full ? WEEKDAYS_FULL : WEEKDAYS_SHORT;
  if (typeof w === "number") return list[((w % 7) + 7) % 7] ?? String(w);
  const n = Number(w);
  if (Number.isInteger(n)) return list[((n % 7) + 7) % 7] ?? w;
  return w;
}

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  return (parts[0]?.[0] ?? "?").toUpperCase() + (parts[1]?.[0] ?? "").toUpperCase();
}

// ---------- Числа ----------

const nf0 = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });

export const fmt0 = (n: number | null | undefined) => (n == null ? "—" : nf0.format(n));
export const fmt1 = (n: number | null | undefined) => (n == null ? "—" : nf1.format(n));

/** Округление «к чётному», как round() в Python: 322.5 → 322. Так ккал в карточке совпадают с текстом ответа. */
export function roundHalfEven(n: number): number {
  const r = Math.round(n);
  return Math.abs(n - Math.trunc(n)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Ккал целым числом с тем же округлением, что и на сервере. */
export const fmtKcal = (n: number | null | undefined) => (n == null ? "—" : nf0.format(roundHalfEven(n)));

/** КБЖУ порции: значения в MealItem даны на 100 г. */
export function portion(item: MealItem, grams = item.grams) {
  const k = grams / 100;
  return {
    kcal: item.kcal * k,
    protein: item.protein * k,
    fat: item.fat * k,
    carbs: item.carbs * k,
  };
}

// ---------- Даты ----------

function toDate(iso: string): Date {
  // Сервер может отдавать время без зоны — считаем его UTC.
  const hasZone = /[zZ]|[+-]\d\d:?\d\d$/.test(iso);
  return new Date(hasZone ? iso : iso + "Z");
}

export function timeShort(iso: string): string {
  return toDate(iso).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
}

export function dateShort(iso: string): string {
  return toDate(iso).toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
}

export function dateTime(iso: string): string {
  return `${dateShort(iso)}, ${timeShort(iso)}`;
}

/** «только что», «5 мин назад», «3 ч назад», иначе дата. */
export function relTime(iso: string): string {
  const diff = (Date.now() - toDate(iso).getTime()) / 1000;
  if (diff < 45) return "только что";
  if (diff < 3600) return `${Math.round(diff / 60)} мин назад`;
  if (diff < 86400) return `${Math.round(diff / 3600)} ч назад`;
  return dateTime(iso);
}
