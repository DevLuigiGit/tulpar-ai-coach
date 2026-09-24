/**
 * ProposalCard — черновик правки программы, который AI подготовил для тренера.
 *
 * Props:
 *   proposal: Proposal                   — kind === "program".
 *   onChange: (p: Proposal) => void      — новая версия после решения или опроса drafting.
 *   onOpenClient?: (clientId: string) => void — ссылка на карточку клиента.
 *   readOnly?: boolean                   — без кнопок решения и без опроса (история).
 *   onDismiss?: () => void               — «Скрыть» у решённой карточки в очереди.
 *
 * Пока status === "drafting", карточка сама опрашивает GET /api/proposals/{id} раз в 3 с.
 */
import { useEffect, useState } from "react";
import { pollProposal } from "../../api";
import { isProgramDraft, type Proposal } from "../../types";
import { relTime } from "../../format";
import PlanView from "../../components/PlanView";
import StatusChip from "../../components/StatusChip";
import { CardHead, Quote, Section, SourceChip } from "./bits";
import { ChangesTable, SkillCheck } from "./review";
import { Drafting, Outcome } from "./ProposalParts";
import DecisionBar from "./DecisionBar";
import { plural, useLatest } from "./util";
import "./trainer.css";

export interface ProposalCardProps {
  proposal: Proposal;
  onChange: (p: Proposal) => void;
  onOpenClient?: (clientId: string) => void;
  readOnly?: boolean;
  onDismiss?: () => void;
}

export default function ProposalCard({ proposal: p, onChange, onOpenClient, readOnly = false, onDismiss }: ProposalCardProps) {
  const onChangeRef = useLatest(onChange);
  const [notice, setNotice] = useState<string | null>(null);
  const drafting = p.status === "drafting";

  useEffect(() => {
    if (readOnly || !drafting) return;
    return pollProposal(p.id, (next) => onChangeRef.current(next), 3000);
  }, [p.id, drafting, readOnly, onChangeRef]);

  const draft = isProgramDraft(p.draft) ? p.draft : null;
  const changes = p.changes ?? [];
  const hasErrors = (p.violations ?? []).some((v) => v.severity === "error");
  const pending = p.status === "pending";
  // Сбой без единой правки (модель недоступна): итог и причину покажет Outcome, пустой черновик не нужен.
  const emptyFailure = p.status === "failed" && changes.length === 0;

  return (
    <article className={`card t-card t-program is-${p.status}`}>
      <CardHead
        name={p.client_name}
        clientId={p.client_id}
        onOpenClient={onOpenClient}
        meta={<>Правка программы · {relTime(p.created_at)}</>}
        aside={
          <>
            <SourceChip source={p.source} />
            <StatusChip status={p.status} kind={p.kind} />
          </>
        }
      />

      <Quote label={p.source === "client" ? "Клиент написал" : "Запрос тренера"}>{p.request}</Quote>

      {drafting && !readOnly && (
        <Drafting redraft={p.decision?.action === "edit"} comment={p.decision?.comment} />
      )}

      {draft && (!drafting || readOnly) && !emptyFailure && (
        <>
          <Section label="Предложение AI">
            <p className="t-summary">{draft.summary}</p>
          </Section>

          <Section
            label="Изменения"
            aside={
              <span className="xs muted">
                {changes.length} {plural(changes.length, ["правка", "правки", "правок"])}
              </span>
            }
          >
            <ChangesTable changes={changes} violations={p.violations} />
          </Section>

          <SkillCheck violations={p.violations} />

          {draft.rationale && (
            <details className="t-details">
              <summary>Обоснование</summary>
              <p className="t-rationale">{draft.rationale}</p>
            </details>
          )}

          {pending && p.before && (
            <details className="t-details">
              <summary>Текущая программа клиента</summary>
              <PlanView plan={p.before} compact />
            </details>
          )}
        </>
      )}

      {notice && !pending && <div className="notice notice-warn">{notice}</div>}

      <Outcome proposal={p} readOnly={readOnly} onDismiss={onDismiss} />

      {pending && !readOnly && (
        <DecisionBar proposal={p} onChange={onChange} hasErrors={hasErrors} onNotice={setNotice} />
      )}
    </article>
  );
}
