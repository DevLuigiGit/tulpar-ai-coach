// Мелкие части чата: ожидающий ответ (этап хода или растущий текст) и пустое состояние с примерами.
import { useEffect, useState } from "react";
import type { StreamStage } from "../../api";
import type { Sending } from "./useSendMessage";
import RichText from "./RichText";
import { SparkIcon } from "./icons";

export const EXAMPLE_PROMPTS = [
  "Съел 200 г плова и чай",
  "Сколько минут активности в неделю рекомендует ВОЗ?",
  "Хочу добавить кардио в программу",
];

export const STAGE_LABEL: Record<StreamStage, string> = {
  listen: "Распознаю голосовое…",
  route: "Разбираю сообщение…",
  search: "Ищу в источниках…",
  answer: "Пишу ответ…",
  meal: "Считаю КБЖУ…",
};

function useSeconds(since: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return Math.max(0, Math.floor((now - since) / 1000));
}

/** Ожидающий ответ: пока текста нет — индикатор с этапом, потом — пузырь, который растёт по предложениям. */
export function PendingReply({ sending }: { sending: Sending }) {
  if (sending.text) return <StreamingBubble text={sending.text} />;
  return <TypingIndicator since={sending.since} withMedia={sending.withMedia} stage={sending.stage} />;
}

function StreamingBubble({ text }: { text: string }) {
  return (
    <div className="msg msg-assistant" aria-live="polite" aria-busy="true">
      <div className="bubble bubble-coach is-streaming">
        <RichText text={text} />
      </div>
    </div>
  );
}

/** Пока ждём ответ (до 90 с): точки, этап хода (если сервер его прислал), счётчик секунд и подсказка. */
export function TypingIndicator({
  since,
  withMedia,
  stage = null,
}: {
  since: number;
  withMedia: boolean;
  stage?: StreamStage | null;
}) {
  const sec = useSeconds(since);
  const hint =
    sec >= 45
      ? "Почти готово — сложные запросы занимают до полутора минут"
      : sec >= 12 && !stage
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
