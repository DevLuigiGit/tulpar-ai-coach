/**
 * QueuePage — вкладка «Очередь»: черновики правок программы и вопросы клиентов, ждущие тренера.
 *
 * Props (передаёт Shells, сигнатуру не менять):
 *   user: User
 *   onOpenClient: (clientId: string) => void
 *   onCountChange?: (n: number) => void — сколько ждут решения (pending + открытые эскалации).
 *
 * GET /api/queue?history=false раз в 15 с. Порядок: pending и эскалации вперемешку по времени,
 * затем drafting. Решённые в этой сессии карточки остаются на месте со статусом, пока их не скроют.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { errorText, queue } from "../../api";
import type { Proposal, User } from "../../types";
import { timeShort } from "../../format";
import Spinner from "../../components/Spinner";
import ProposalCard from "./ProposalCard";
import EscalationCard from "./EscalationCard";
import QueueEmpty from "./QueueEmpty";
import { isFinal, needsDecision, plural, sortQueue } from "./util";
import "./trainer.css";

export interface QueuePageProps {
  user: User;
  onOpenClient: (clientId: string) => void;
  onCountChange?: (n: number) => void;
}

const POLL_MS = 15_000;

export default function QueuePage({ onOpenClient, onCountChange }: QueuePageProps) {
  const [items, setItems] = useState<Proposal[] | null>(null);
  const [settled, setSettled] = useState<Record<string, Proposal>>({});
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const list = await queue(false);
      setItems(list);
      setError(null);
      setUpdatedAt(new Date().toISOString());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const id = window.setInterval(() => void load(), POLL_MS);
    return () => window.clearInterval(id);
  }, [load]);

  const onChange = useCallback((p: Proposal) => {
    if (isFinal(p.status)) setSettled((s) => ({ ...s, [p.id]: p }));
    setItems((list) => (list ? list.map((x) => (x.id === p.id ? p : x)) : list));
  }, []);

  const view = useMemo(() => {
    const byId = new Map<string, Proposal>();
    for (const p of items ?? []) byId.set(p.id, p);
    for (const p of Object.values(settled)) byId.set(p.id, p);
    return sortQueue([...byId.values()].filter((p) => !hidden.has(p.id)));
  }, [items, settled, hidden]);

  const waiting = view.filter(needsDecision).length;
  const drafting = view.filter((p) => p.status === "drafting").length;
  const escalations = view.filter((p) => p.kind === "escalation" && p.status === "open").length;

  useEffect(() => {
    if (items) onCountChange?.(waiting);
  }, [items, waiting, onCountChange]);

  const hide = (id: string) => setHidden((h) => new Set(h).add(id));

  return (
    <div className="t-page">
      <div className="t-queue-head">
        <div className="grow">
          <h2 className="page-title">Очередь решений</h2>
          <p className="page-sub">AI предлагает — вы решаете. Без вашего «Принять» программа клиента не меняется.</p>
        </div>
        <div className={`t-counter ${waiting ? "is-active" : ""}`} aria-live="polite">
          <span className="t-counter-num num">{items ? waiting : "—"}</span>
          <span className="t-counter-text">{plural(waiting, ["ждёт решения", "ждут решения", "ждут решения"])}</span>
        </div>
      </div>

      <div className="t-toolbar">
        <div className="row wrap">
          {escalations > 0 && (
            <span className="chip status-escalation chip-dot">
              {escalations} {plural(escalations, ["вопрос", "вопроса", "вопросов"])} от клиентов
            </span>
          )}
          {drafting > 0 && (
            <span className="chip status-drafting">
              <span className="spinner spinner-sm" aria-hidden="true" />
              AI готовит {drafting} {plural(drafting, ["черновик", "черновика", "черновиков"])}
            </span>
          )}
        </div>
        <div className="grow" />
        <span className="xs faint nowrap">{updatedAt ? `обновлено в ${timeShort(updatedAt)}` : ""}</span>
        <button className="btn btn-ghost btn-sm" onClick={() => void load()} disabled={loading}>
          {loading ? <Spinner size="sm" /> : null}
          Обновить
        </button>
      </div>

      {error && (
        <div className="notice notice-error t-gap">
          Не удалось обновить очередь: {error}. {items ? "Показываем последние данные." : ""}
        </div>
      )}

      {items === null && !error && <QueueSkeleton />}

      {items !== null && view.length === 0 && <QueueEmpty />}

      <div className="t-list">
        {view.map((p) =>
          p.kind === "escalation" ? (
            <EscalationCard
              key={p.id}
              proposal={p}
              onChange={onChange}
              onOpenClient={onOpenClient}
              onDismiss={isFinal(p.status) ? () => hide(p.id) : undefined}
            />
          ) : (
            <ProposalCard
              key={p.id}
              proposal={p}
              onChange={onChange}
              onOpenClient={onOpenClient}
              onDismiss={isFinal(p.status) ? () => hide(p.id) : undefined}
            />
          ),
        )}
      </div>
    </div>
  );
}

function QueueSkeleton() {
  return (
    <div className="t-list" aria-hidden="true">
      {[0, 1].map((i) => (
        <div key={i} className="card t-card t-skel-card">
          <span className="skeleton" style={{ width: 180, height: 16 }} />
          <span className="skeleton" style={{ width: "70%", height: 14 }} />
          <span className="skeleton" style={{ width: "100%", height: 64 }} />
        </div>
      ))}
    </div>
  );
}
