/**
 * HistoryPage — вкладка «История»: закрытые решения тренера.
 *
 * Props (передаёт Shells, сигнатуру не менять):
 *   user: User
 *   onOpenClient: (clientId: string) => void
 *
 * GET /api/queue?history=true → applied | rejected | failed | resolved, новые сверху, фильтр по статусу.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { errorText, queue } from "../../api";
import type { Proposal, ProposalStatus, User } from "../../types";
import Empty from "../../components/Empty";
import Spinner from "../../components/Spinner";
import HistoryItem from "./HistoryItem";
import { sortByUpdated } from "./util";
import "./trainer.css";

export interface HistoryPageProps {
  user: User;
  onOpenClient: (clientId: string) => void;
}

type Filter = "all" | ProposalStatus;

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "Все" },
  { id: "applied", label: "Применено" },
  { id: "rejected", label: "Отклонено" },
  { id: "resolved", label: "Вопросы закрыты" },
  { id: "failed", label: "Ошибки" },
];

export default function HistoryPage({ onOpenClient }: HistoryPageProps) {
  const [items, setItems] = useState<Proposal[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setItems(sortByUpdated(await queue(true)));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: items?.length ?? 0 };
    for (const p of items ?? []) c[p.status] = (c[p.status] ?? 0) + 1;
    return c;
  }, [items]);

  const shown = useMemo(
    () => (items ?? []).filter((p) => filter === "all" || p.status === filter),
    [items, filter],
  );

  return (
    <div className="t-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">История решений</h2>
          <p className="page-sub">Что AI предложил, что вы решили и с каким комментарием.</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={() => void load()} disabled={loading}>
          {loading ? <Spinner size="sm" /> : null}
          Обновить
        </button>
      </div>

      {items && items.length > 0 && (
        <div className="row wrap t-filters" role="group" aria-label="Фильтр по статусу">
          {FILTERS.filter((f) => f.id === "all" || counts[f.id]).map((f) => (
            <button
              key={f.id}
              type="button"
              className={`chip ${filter === f.id ? "is-active" : ""}`}
              onClick={() => setFilter(f.id)}
            >
              {f.label}
              <span className="t-filter-count num">{counts[f.id] ?? 0}</span>
            </button>
          ))}
        </div>
      )}

      {error && <div className="notice notice-error">Не удалось загрузить историю: {error}</div>}
      {!items && !error && <Spinner label="Загружаем историю" />}

      {items && items.length === 0 && (
        <Empty
          title="Решений пока нет"
          text="Когда вы примете, отклоните черновик или ответите на вопрос клиента, запись появится здесь."
        />
      )}
      {items && items.length > 0 && shown.length === 0 && <Empty title="Нет записей с таким статусом" />}

      {shown.length > 0 && (
        <div className="t-hist-list">
          {shown.map((p) => (
            <HistoryItem key={p.id} proposal={p} onOpenClient={onOpenClient} />
          ))}
        </div>
      )}
    </div>
  );
}
