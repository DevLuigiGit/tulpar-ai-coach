/**
 * ClientDetail — карточка клиента для тренера.
 *
 * Props:
 *   clientId: string
 *   onBack: () => void                            — вернуться к списку клиентов.
 *   onProposalCreated?: (p: Proposal) => void     — после requestChange (статус drafting).
 *
 * GET /api/trainer/clients/{id}/context → профиль, заметка тренера, питание за 14 дней, тренировки,
 * вес, активная программа; GET /api/trainer/clients/{id}/chat → диалог с коучем.
 * Форма «Попросить AI изменить программу» → черновик в очереди (RequestChange).
 */
import { useCallback, useEffect, useState } from "react";
import { clientContext, errorText } from "../../api";
import type { ClientContext, Proposal } from "../../types";
import Empty from "../../components/Empty";
import PlanView from "../../components/PlanView";
import Spinner from "../../components/Spinner";
import { ClientHero, NoteCard } from "./ClientHero";
import { NutritionCard, Panel, SessionsCard, WeightsCard } from "./ClientStats";
import ClientChat from "./ClientChat";
import RequestChange from "./RequestChange";
import "./trainer.css";

export interface ClientDetailProps {
  clientId: string;
  onBack: () => void;
  onProposalCreated?: (p: Proposal) => void;
}

export default function ClientDetail({ clientId, onBack, onProposalCreated }: ClientDetailProps) {
  const [ctx, setCtx] = useState<ClientContext | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [planFlash, setPlanFlash] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      setCtx(await clientContext(clientId));
    } catch (e) {
      setError(errorText(e));
    }
  }, [clientId]);

  useEffect(() => {
    setCtx(null);
    void load();
  }, [load]);

  const onApplied = useCallback(() => {
    void load();
    setPlanFlash(true);
    window.setTimeout(() => setPlanFlash(false), 2400);
  }, [load]);

  return (
    <div className="t-page">
      <div>
        <button className="btn btn-ghost btn-sm t-back" onClick={onBack}>
          <span className="t-back-chevron" aria-hidden="true" />
          Все клиенты
        </button>
      </div>

      {error && (
        <div className="notice notice-error row between">
          <span>Не удалось загрузить карточку: {error}</span>
          <button className="btn btn-sm" onClick={() => void load()}>
            Повторить
          </button>
        </div>
      )}
      {!ctx && !error && <Spinner label="Загружаем карточку клиента" />}

      {ctx && (
        <>
          <ClientHero ctx={ctx} />
          <NoteCard note={ctx.trainer_note} />
          <div className="t-detail-grid">
            <div className="t-detail-main">
              <RequestChange
                clientId={ctx.client_id}
                hasPlan={!!ctx.active_plan}
                onCreated={onProposalCreated}
                onApplied={onApplied}
              />
              <Panel
                title="Активная программа"
                aside={planFlash ? <span className="chip status-applied chip-dot">обновлена</span> : undefined}
              >
                {ctx.active_plan ? (
                  <PlanView plan={ctx.active_plan} compact />
                ) : (
                  <Empty title="Нет активной программы" text="Назначьте клиенту программу — тогда AI сможет предлагать к ней правки." />
                )}
              </Panel>
              <ClientChat clientId={ctx.client_id} name={ctx.name} />
            </div>
            <aside className="t-detail-side">
              <NutritionCard n={ctx.nutrition_14d} />
              <SessionsCard sessions={ctx.recent_sessions} />
              <WeightsCard weights={ctx.weights} goal={ctx.profile.goal_weight_kg} />
            </aside>
          </div>
        </>
      )}
    </div>
  );
}
