// Оценка ответа коуча: 👍/👎 и необязательный комментарий к 👎.
// Оценка уходит на сервер сразу; комментарий — вторым запросом, только если клиент его написал.
import { useState } from "react";
import { errorText, rateMessage } from "../../api";
import type { FeedbackRating } from "../../types";
import { ThumbDownIcon, ThumbUpIcon } from "./icons";

interface Props {
  messageId: number;
  initial: FeedbackRating | null;
}

const COMMENT_MAX = 500;

export default function AnswerFeedback({ messageId, initial }: Props) {
  const [rating, setRating] = useState<FeedbackRating | null>(initial);
  const [saving, setSaving] = useState(false);
  const [asking, setAsking] = useState(false);
  const [thanks, setThanks] = useState(false);
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);

  const rate = async (next: FeedbackRating) => {
    if (saving) return;
    if (next === rating) {
      // Повторный 👎 снова открывает поле комментария; повторный 👍 ничего не меняет.
      if (next === "down") setAsking(true);
      return;
    }
    const prev = rating;
    setRating(next);
    setError(null);
    setThanks(false);
    setSaving(true);
    try {
      await rateMessage(messageId, next);
      setAsking(next === "down");
      setThanks(next === "up");
    } catch (e) {
      setRating(prev);
      setError(errorText(e));
    } finally {
      setSaving(false);
    }
  };

  const sendComment = async () => {
    const text = comment.trim();
    if (!text) {
      setAsking(false);
      setThanks(true);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await rateMessage(messageId, "down", text);
      setAsking(false);
      setThanks(true);
      setComment("");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fb">
      <div className="fb-row">
        <button
          type="button"
          className={`fb-btn ${rating === "up" ? "is-up" : ""}`}
          aria-pressed={rating === "up"}
          aria-label="Полезный ответ"
          title="Полезный ответ"
          disabled={saving}
          onClick={() => void rate("up")}
        >
          <ThumbUpIcon size={15} />
        </button>
        <button
          type="button"
          className={`fb-btn ${rating === "down" ? "is-down" : ""}`}
          aria-pressed={rating === "down"}
          aria-label="Неудачный ответ"
          title="Неудачный ответ"
          disabled={saving}
          onClick={() => void rate("down")}
        >
          <ThumbDownIcon size={15} />
        </button>
        {thanks && !error && <span className="fb-note">Спасибо за оценку</span>}
        {error && (
          <span className="fb-note is-error" role="alert">
            {error}
          </span>
        )}
      </div>
      {asking && (
        <form
          className="fb-form"
          onSubmit={(e) => {
            e.preventDefault();
            void sendComment();
          }}
        >
          <input
            className="input input-sm fb-input"
            value={comment}
            maxLength={COMMENT_MAX}
            placeholder="Что не так с ответом? Необязательно"
            aria-label="Комментарий к оценке"
            onChange={(e) => setComment(e.target.value)}
            disabled={saving}
            autoFocus
          />
          <button className="btn btn-sm btn-outline" type="submit" disabled={saving}>
            Отправить
          </button>
          <button
            className="btn btn-sm btn-ghost"
            type="button"
            disabled={saving}
            onClick={() => {
              setAsking(false);
              setThanks(true);
            }}
          >
            Пропустить
          </button>
        </form>
      )}
    </div>
  );
}
