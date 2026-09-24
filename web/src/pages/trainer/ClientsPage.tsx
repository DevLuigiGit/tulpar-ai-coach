/**
 * ClientsPage — вкладка «Клиенты» тренера.
 *
 * Props (передаёт Shells, сигнатуру не менять):
 *   user: User
 *   selectedClientId: string | null      — из адреса #/clients/<id>; null — список.
 *   onSelectClient: (id: string | null) => void
 *
 * GET /api/trainer/clients → строки (имя, цель/уровень/место, «ограничение», активная программа).
 * При selectedClientId рендерится <ClientDetail>.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { clients as fetchClients, errorText } from "../../api";
import type { ClientSummary, User } from "../../types";
import { goalLabel, initials, levelLabel, placeLabel } from "../../format";
import Empty from "../../components/Empty";
import ClientDetail from "./ClientDetail";
import { plural } from "./util";
import "./trainer.css";

export interface ClientsPageProps {
  user: User;
  selectedClientId: string | null;
  onSelectClient: (id: string | null) => void;
}

export default function ClientsPage({ selectedClientId, onSelectClient }: ClientsPageProps) {
  if (selectedClientId) {
    return <ClientDetail clientId={selectedClientId} onBack={() => onSelectClient(null)} />;
  }
  return <ClientList onSelect={onSelectClient} />;
}

function ClientList({ onSelect }: { onSelect: (id: string) => void }) {
  const [list, setList] = useState<ClientSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const load = useCallback(async () => {
    setError(null);
    try {
      setList(await fetchClients());
    } catch (e) {
      setError(errorText(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const all = list ?? [];
    return q ? all.filter((c) => c.name.toLowerCase().includes(q)) : all;
  }, [list, query]);

  const injured = (list ?? []).filter((c) => c.injury).length;

  return (
    <div className="t-page">
      <div className="page-head">
        <div>
          <h2 className="page-title">Клиенты</h2>
          <p className="page-sub">
            {list
              ? `${list.length} ${plural(list.length, ["клиент", "клиента", "клиентов"])}` +
                (injured ? ` · у ${injured} есть ограничения` : "")
              : "Загружаем список"}
          </p>
        </div>
        {list && list.length > 5 && (
          <input
            className="input input-sm t-search"
            type="search"
            placeholder="Поиск по имени"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        )}
      </div>

      {error && (
        <div className="notice notice-error row between">
          <span>Не удалось загрузить клиентов: {error}</span>
          <button className="btn btn-sm" onClick={() => void load()}>
            Повторить
          </button>
        </div>
      )}

      {!list && !error && (
        <div className="t-rows" aria-hidden="true">
          {[0, 1, 2].map((i) => (
            <div key={i} className="t-row is-skeleton">
              <span className="skeleton" style={{ width: 40, height: 40, borderRadius: "50%" }} />
              <span className="skeleton" style={{ width: "40%", height: 14 }} />
            </div>
          ))}
        </div>
      )}

      {list && list.length === 0 && (
        <Empty title="Пока нет клиентов" text="Когда к вам привяжут клиентов, они появятся здесь." />
      )}
      {list && list.length > 0 && shown.length === 0 && <Empty title="Никого не нашли" text="Попробуйте другое имя." />}

      {shown.length > 0 && (
        <div className="t-rows">
          {shown.map((c) => (
            <button key={c.id} type="button" className="t-row" onClick={() => onSelect(c.id)}>
              <span className="avatar t-row-avatar" aria-hidden="true">
                {initials(c.name)}
              </span>
              <span className="t-row-main">
                <span className="t-row-name">
                  {c.name}
                  {c.injury && <span className="chip t-injury">ограничение</span>}
                </span>
                <span className="t-row-plan truncate">
                  {c.active_plan_title ? c.active_plan_title : <span className="faint">нет активной программы</span>}
                </span>
              </span>
              <span className="t-row-chips">
                <span className="chip">{goalLabel(c.goal)}</span>
                <span className="chip">{levelLabel(c.level)}</span>
                <span className="chip">{placeLabel(c.place)}</span>
              </span>
              <span className="t-chevron" aria-hidden="true" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
