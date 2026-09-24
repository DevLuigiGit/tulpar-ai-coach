// Разделитель дней в ленте: «Сегодня», «Вчера» или дата.
import { dateShort } from "../../format";

function toDate(iso: string): Date {
  const hasZone = /[zZ]|[+-]\d\d:?\d\d$/.test(iso);
  return new Date(hasZone ? iso : iso + "Z");
}

/** Ключ календарного дня в локальной зоне пользователя. */
export function dayKey(iso: string): string {
  const d = toDate(iso);
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
}

function dayLabel(iso: string): string {
  const d = toDate(iso);
  const today = new Date();
  const yesterday = new Date(today.getFullYear(), today.getMonth(), today.getDate() - 1);
  if (dayKey(iso) === dayKey(today.toISOString())) return "Сегодня";
  if (dayKey(iso) === dayKey(yesterday.toISOString())) return "Вчера";
  return d.getFullYear() === today.getFullYear() ? dateShort(iso) : d.toLocaleDateString("ru-RU");
}

export default function DaySeparator({ iso }: { iso: string }) {
  return (
    <div className="day-sep" role="separator">
      <span>{dayLabel(iso)}</span>
    </div>
  );
}
