// Мелкие части чата: индикатор «Коуч думает…» и пустое состояние с примерами.
import { useEffect, useState } from "react";
import type { ChatStage } from "../../api";
import { SparkIcon } from "./icons";

export const EXAMPLE_PROMPTS = [
  "Съел 200 г плова и чай",
  "Сколько минут активности в неделю рекомендует ВОЗ?",
  "Хочу добавить кардио в программу",
];

const STAGE_LABEL: Record<ChatStage, string> = {
  listen: "Слушаю голосовое…",
  route: "Разбираю сообщение…",
  search: "Ищу в базе знаний…",
  answer: "Пишу ответ по источникам…",
  meal: "Считаю калории…",
  photo: "Смотрю на фото…",
  program: "Передаю просьбу тренеру…",
};

/** Пока ждём ответ (до 90 с): точки, этап графа (если сервер его прислал), счётчик секунд и подсказка. */
export function TypingIndicator({ since, withMedia, stage }: { since: number; withMedia: boolean; stage?: ChatStage }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const sec = Math.max(0, Math.floor((now - since) / 1000));
  const hint =
    sec >= 45
      ? "Почти готово — сложные запросы занимают до полутора минут"
      : sec >= 12
        ? withMedia
          ? "Разбираю вложение и сверяюсь со справочником"
          : "Сверяюсь с источниками"
        : null;

  return (
    <div className="msg msg-assistant" aria-live="polite">
      <div className="bubble bubble-coach typing">
        <span className="typing-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span className="typing-label">{(stage && STAGE_LABEL[stage]) || "Коуч думает…"}</span>
        {sec >= 5 && <span className="typing-sec num">{sec} с</span>}
      </div>
      {hint && <div className="typing-hint">{hint}</div>}
    </div>
  );
}

export function ChatEmpty({ onPick, disabled }: { onPick: (text: string) => void; disabled: boolean }) {
  return (
    <div className="chat-empty">
      <div className="chat-empty-mark">
        <SparkIcon size={22} />
      </div>
      <div className="chat-empty-title">Ваш AI-коуч</div>
      <p className="chat-empty-text">
        Считаю КБЖУ по тексту, фото и голосу, отвечаю со ссылками на источники. Правки
        программы передаю тренеру — он решает.
      </p>
      <div className="chat-empty-label">Попробуйте спросить</div>
      <div className="chat-prompts">
        {EXAMPLE_PROMPTS.map((p) => (
          <button key={p} className="chat-prompt" disabled={disabled} onClick={() => onPick(p)}>
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}
