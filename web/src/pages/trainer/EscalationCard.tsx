/**
 * EscalationCard — сообщение клиента, на которое AI намеренно не ответил сам.
 *
 * Props:
 *   proposal: Proposal                   — kind === "escalation"; draft: { reason, intent }.
 *   onChange: (p: Proposal) => void      — после resolveEscalation.
 *   onOpenClient?: (clientId: string) => void
 *   readOnly?: boolean                   — для «Истории»: показать ответ, без формы.
 *   onDismiss?: () => void               — «Скрыть» у закрытой карточки в очереди.
 *
 * «Ответить и закрыть» → POST /api/escalations/{id}/resolve {reply} (ответ уходит в чат клиента),
 * «Закрыть без ответа» → тот же запрос без reply.
 */
import { useState } from "react";
import { errorText, resolveEscalation } from "../../api";
import { isEscalationDraft, type Proposal } from "../../types";
import { relTime } from "../../format";
import Spinner from "../../components/Spinner";
import StatusChip from "../../components/StatusChip";
import { CardHead, Quote } from "./bits";
import { escalationHeadline, reasonText } from "./util";
import "./trainer.css";

export interface EscalationCardProps {
  proposal: Proposal;
  onChange: (p: Proposal) => void;
  onOpenClient?: (clientId: string) => void;
  readOnly?: boolean;
  onDismiss?: () => void;
}

export default function EscalationCard({ proposal: p, onChange, onOpenClient, readOnly = false, onDismiss }: EscalationCardProps) {
  const [reply, setReply] = useState("");
  const [busy, setBusy] = useState<"reply" | "close" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const draft = isEscalationDraft(p.draft) ? p.draft : null;
  const open = p.status === "open";
  const red = draft?.intent === "escalate";
  const reason = reasonText(draft?.reason);

  async function resolve(withReply: boolean) {
    if (busy) return;
    const text = reply.trim();
    if (withReply && !text) return;
    setBusy(withReply ? "reply" : "close");
    setError(null);
    try {
      onChange(await resolveEscalation(p.id, withReply ? text : undefined));
      setReply("");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <article className={`card t-card t-escalation ${open ? "is-open" : "is-closed"}`}>
      <CardHead
        name={p.client_name}
        clientId={p.client_id}
        onOpenClient={onOpenClient}
        tone={open ? "danger" : "default"}
        meta={<>Вопрос тренеру · {relTime(p.created_at)}</>}
        aside={<StatusChip status={p.status} kind={p.kind} />}
      />

      <Quote label="Сообщение клиента">{p.request}</Quote>

      <div className={`t-esc-reason ${red ? "is-red" : ""}`}>
        <div className="t-esc-reason-title">{escalationHeadline(draft?.intent)}</div>
        {reason && <div className="t-esc-reason-text">{reason}</div>}
        <div className="t-esc-reason-foot">AI не ответил сам и передал сообщение вам. Клиент получил безопасный ответ.</div>
      </div>

      {open && !readOnly && (
        <div className="t-decide">
          <textarea
            className="textarea"
            rows={3}
            value={reply}
            disabled={!!busy}
            aria-label="Ответ клиенту"
            placeholder="Ответ клиенту — он придёт в чат коуча от вашего имени"
            onChange={(e) => setReply(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void resolve(true);
            }}
          />
          {error && <div className="notice notice-error">{error}</div>}
          <div className="t-decide-actions">
            <button className="btn btn-primary" disabled={!!busy || !reply.trim()} onClick={() => void resolve(true)}>
              {busy === "reply" ? (
                <>
                  <Spinner size="sm" />
                  Отправляем…
                </>
              ) : (
                "Ответить и закрыть"
              )}
            </button>
            <div className="grow" />
            <button className="btn btn-ghost" disabled={!!busy} onClick={() => void resolve(false)}>
              {busy === "close" ? <Spinner size="sm" /> : null}
              Закрыть без ответа
            </button>
          </div>
        </div>
      )}

      {p.status === "resolved" && (
        <div className="t-outcome is-applied">
          <span className="t-outcome-mark" aria-hidden="true" />
          <div className="grow">
            <div className="t-outcome-title">{p.reply ? "Ответ отправлен клиенту" : "Закрыто без ответа"}</div>
            {p.reply && <div className="t-outcome-text">«{p.reply}»</div>}
          </div>
          {onDismiss && (
            <button className="btn btn-ghost btn-sm" onClick={onDismiss}>
              Скрыть
            </button>
          )}
        </div>
      )}
    </article>
  );
}
