// Оценки ответов коуча от клиентов тренера: счётчики 👍/👎 и последние неудачные ответы
// с вопросом и комментарием — по ним видно, где коуч ошибается.
import { useCallback, useEffect, useState } from "react";
import { errorText, trainerFeedback } from "../../api";
import type { TrainerFeedback } from "../../types";
import { dateTime } from "../../format";
import { Panel } from "./ClientStats";

const SHOWN = 5;

export default function FeedbackPanel() {
  const [data, setData] = useState<TrainerFeedback | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setData(await trainerFeedback(undefined, 50));
    } catch (e) {
      setError(errorText(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const bad = (data?.items ?? []).filter((i) => i.rating === "down").slice(0, SHOWN);

  return (
    <Panel
      title="Оценки ответов коуча"
      aside={
        <button className="btn btn-ghost btn-sm" onClick={() => void load()}>
          Обновить
        </button>
      }
    >
      {error && <div className="notice notice-error">Не удалось загрузить оценки: {error}</div>}
      {data && (
        <div className="t-tiles t-fb-tiles">
          <div className="t-tile is-up">
            <div className="t-tile-value num">{data.counts.up}</div>
            <div className="t-tile-label">Полезно</div>
          </div>
          <div className="t-tile is-down">
            <div className="t-tile-value num">{data.counts.down}</div>
            <div className="t-tile-label">Неудачно</div>
          </div>
        </div>
      )}
      {data && data.counts.total === 0 && (
        <p className="muted small">Клиенты ещё не оценивали ответы. Кнопки оценки есть под ответами в чате и в боте.</p>
      )}
      {bad.length > 0 && (
        <div className="t-fb-list">
          {bad.map((i) => (
            <div key={i.id} className="t-fb-item">
              <div className="t-msg-meta">
                <span className="t-msg-who">{i.client_name}</span>
                <span>{dateTime(i.updated_at)}</span>
                <span className="t-msg-tag is-red">неудачно</span>
                {i.source === "telegram" && <span className="t-msg-tag">Telegram</span>}
              </div>
              {i.question && <div className="t-fb-q">«{i.question}»</div>}
              <div className="t-fb-a">{i.answer}</div>
              {i.comment && <div className="t-fb-c">Комментарий: {i.comment}</div>}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
