// Компактная строка истории: статус, клиент, суть и комментарий решения; по клику — подробности.
import { useState } from "react";
import { isProgramDraft, type Proposal } from "../../types";
import { dateTime, initials } from "../../format";
import StatusChip from "../../components/StatusChip";
import ProposalCard from "./ProposalCard";
import EscalationCard from "./EscalationCard";
import { SOURCE_TEXT } from "./util";

interface HistoryItemProps {
  proposal: Proposal;
  onOpenClient: (clientId: string) => void;
}

const noop = () => {};

export default function HistoryItem({ proposal: p, onOpenClient }: HistoryItemProps) {
  const [open, setOpen] = useState(false);
  const program = p.kind === "program";
  // У failed причина уже в p.reply — вместо технического резюме показываем сам запрос.
  const summary = program && p.status !== "failed" && isProgramDraft(p.draft) ? p.draft.summary : null;
  const comment = p.decision?.comment;

  return (
    <article className={`t-hist is-${p.status} ${open ? "is-open" : ""}`}>
      <button type="button" className="t-hist-row" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="avatar t-hist-avatar" aria-hidden="true">
          {initials(p.client_name)}
        </span>
        <span className="t-hist-main">
          <span className="t-hist-top">
            <span className="t-hist-name">{p.client_name}</span>
            <span className="t-hist-kind">
              {program ? `Правка программы · ${SOURCE_TEXT[p.source]}` : "Вопрос тренеру"}
            </span>
          </span>
          <span className="t-hist-summary">{summary ?? `«${p.request}»`}</span>
          {program && comment && <span className="t-hist-note">Комментарий: «{comment}»</span>}
          {!program && p.reply && <span className="t-hist-note">Ответ: «{p.reply}»</span>}
          {p.status === "failed" && p.reply && program && <span className="t-hist-note is-failed">{p.reply}</span>}
        </span>
        <span className="t-hist-side">
          <StatusChip status={p.status} kind={p.kind} />
          <span className="xs faint nowrap">{dateTime(p.updated_at)}</span>
        </span>
        <span className="t-hist-caret" aria-hidden="true" />
      </button>
      {open && (
        <div className="t-hist-body">
          {program ? (
            <ProposalCard proposal={p} onChange={noop} onOpenClient={onOpenClient} readOnly />
          ) : (
            <EscalationCard proposal={p} onChange={noop} onOpenClient={onOpenClient} readOnly />
          )}
        </div>
      )}
    </article>
  );
}
