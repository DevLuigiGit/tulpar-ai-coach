// Переписка клиента с AI-коучем глазами тренера (только чтение).
import { useCallback, useEffect, useRef, useState } from "react";
import { clientChat, errorText } from "../../api";
import type { ChatMessage } from "../../types";
import { dateTime } from "../../format";
import Spinner from "../../components/Spinner";
import { Panel } from "./ClientStats";

const TAG: Record<string, [string, string]> = {
  escalated: ["передано тренеру", "is-red"],
  proposal: ["черновик программы", "is-accent"],
  meal_card: ["приём пищи", ""],
  refusal: ["вне темы", ""],
  info: ["справка", ""],
  meal_logged: ["записано в дневник", "is-green"],
  trainer_reply: ["ответ тренера", "is-accent"],
};

function tagOf(m: ChatMessage): [string, string] | null {
  const kind = m.payload && typeof m.payload === "object" ? (m.payload as { kind?: string }).kind : undefined;
  return kind ? TAG[kind] ?? null : null;
}

export default function ClientChat({ clientId, name }: { clientId: string; name: string }) {
  const [messages, setMessages] = useState<ChatMessage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setMessages(await clientChat(clientId, 40));
    } catch (e) {
      setError(errorText(e));
    }
  }, [clientId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const box = boxRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [messages]);

  return (
    <Panel
      title="Диалог с AI-коучем"
      aside={
        <button className="btn btn-ghost btn-sm" onClick={() => void load()}>
          Обновить
        </button>
      }
    >
      {error && <div className="notice notice-error">Не удалось загрузить переписку: {error}</div>}
      {!messages && !error && <Spinner label="Загружаем переписку" />}
      {messages && messages.length === 0 && (
        <p className="muted small">Переписки с коучем пока нет.</p>
      )}
      {messages && messages.length > 0 && (
        <div className="t-chat" ref={boxRef}>
          {messages.map((m) => {
            const tag = tagOf(m);
            const isTrainer = tag?.[0] === "ответ тренера";
            const who = m.role === "user" ? name : isTrainer ? "Вы" : "AI-коуч";
            return (
              <div key={m.id} className={`t-msg ${m.role === "user" ? "is-client" : "is-bot"} ${isTrainer ? "is-trainer" : ""}`}>
                <div className="t-msg-meta">
                  <span className="t-msg-who">{who}</span>
                  <span>{dateTime(m.created_at)}</span>
                  {tag && !isTrainer && <span className={`t-msg-tag ${tag[1]}`}>{tag[0]}</span>}
                  {m.feedback && (
                    <span className={`t-msg-tag ${m.feedback === "up" ? "is-green" : "is-red"}`}>
                      {m.feedback === "up" ? "оценка: полезно" : "оценка: неудачно"}
                    </span>
                  )}
                </div>
                <div className="t-msg-text">{m.text || <span className="faint">вложение</span>}</div>
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
