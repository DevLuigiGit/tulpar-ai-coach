// Решение тренера по черновику: принять, поправить с комментарием или отклонить.
import { useState } from "react";
import { decide, errorText, proposal as fetchProposal } from "../../api";
import type { DecisionAction, Proposal } from "../../types";
import Spinner from "../../components/Spinner";
import { isConflict } from "./util";

interface DecisionBarProps {
  proposal: Proposal;
  onChange: (p: Proposal) => void;
  hasErrors: boolean;
  /** Сообщение, которое должно пережить исчезновение панели (например, 409). */
  onNotice?: (text: string) => void;
}

const BUSY_TEXT: Record<DecisionAction, string> = {
  accept: "Применяем…",
  edit: "Отправляем…",
  reject: "Отклоняем…",
};

export default function DecisionBar({ proposal, onChange, hasErrors, onNotice }: DecisionBarProps) {
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState<DecisionAction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const trimmed = comment.trim();

  async function act(action: DecisionAction) {
    if (busy) return;
    if (action === "edit" && !trimmed) {
      setError("Напишите, что поправить, — AI переделает черновик по комментарию.");
      return;
    }
    setBusy(action);
    setError(null);
    setInfo(null);
    try {
      const next = await decide(proposal.id, action, trimmed || undefined);
      setComment("");
      onChange(next);
    } catch (e) {
      if (isConflict(e)) {
        const msg = "Это предложение уже решено — возможно, в Telegram или в другой вкладке. Показываем актуальное состояние.";
        setInfo(msg);
        onNotice?.(msg);
        try {
          onChange(await fetchProposal(proposal.id));
        } catch {
          /* карточку обновит очередной опрос очереди */
        }
      } else {
        setError(errorText(e));
      }
    } finally {
      setBusy(null);
    }
  }

  const label = (action: DecisionAction, text: string) =>
    busy === action ? (
      <>
        <Spinner size="sm" />
        {BUSY_TEXT[action]}
      </>
    ) : (
      text
    );

  return (
    <div className="t-decide">
      <textarea
        className="textarea t-decide-input"
        rows={2}
        value={comment}
        disabled={!!busy}
        aria-label="Комментарий для AI"
        placeholder="Комментарий для AI, если нужно поправить: например, «жим оставь, замени только присед»"
        onChange={(e) => setComment(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && trimmed) void act("edit");
        }}
      />
      {info && <div className="notice notice-warn">{info}</div>}
      {error && <div className="notice notice-error">{error}</div>}
      {hasErrors && !busy && (
        <div className="notice notice-error">
          Валидатор нашёл ошибки. Лучше попросить AI поправить черновик, чем принимать как есть.
        </div>
      )}
      <div className="t-decide-actions">
        <button className="btn btn-primary t-accept" disabled={!!busy} onClick={() => void act("accept")}>
          {label("accept", "Принять")}
        </button>
        <button
          className="btn btn-outline"
          disabled={!!busy || !trimmed}
          title={trimmed ? "Отправить комментарий AI и получить новый черновик" : "Сначала напишите комментарий"}
          onClick={() => void act("edit")}
        >
          {label("edit", "Поправить")}
        </button>
        <div className="grow" />
        <button className="btn btn-danger" disabled={!!busy} onClick={() => void act("reject")}>
          {label("reject", "Отклонить")}
        </button>
      </div>
      <p className="t-decide-hint">
        «Принять» сразу меняет программу клиента. «Поправить» отправит комментарий AI — новый черновик вернётся в
        очередь. Комментарий к «Отклонить» сохранится в истории.
      </p>
    </div>
  );
}
