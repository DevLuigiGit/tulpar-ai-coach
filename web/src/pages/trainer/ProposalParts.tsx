// Состояния карточки предложения: AI готовит черновик и итог решения.
import { useEffect, useState } from "react";
import type { Plan, PlanDay, Proposal } from "../../types";
import PlanView from "../../components/PlanView";
import Spinner from "../../components/Spinner";

function daySig(d: PlanDay | undefined): string {
  if (!d) return "";
  return d.exercises.map((e) => `${e.exercise_id}:${e.target_sets}:${e.target_reps}`).join("|");
}

/** Только дни, которые отличаются от before, — чтобы тренер сразу видел, что поменялось. */
function changedOnly(after: Plan, before: Plan | null): { plan: Plan; shown: number; total: number } {
  if (!before) return { plan: after, shown: after.days.length, total: after.days.length };
  const days = after.days.filter((d) => daySig(d) !== daySig(before.days.find((b) => b.day_index === d.day_index)));
  const list = days.length ? days : after.days;
  return { plan: { ...after, days: list }, shown: list.length, total: after.days.length };
}

/** Секунды с момента показа — чтобы на защите было видно, что работа идёт. */
function Elapsed() {
  const [start] = useState(() => Date.now());
  const [now, setNow] = useState(start);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const s = Math.max(0, Math.round((now - start) / 1000));
  return <span className="t-elapsed num">{s < 60 ? `${s} с` : `${Math.floor(s / 60)} мин ${s % 60} с`}</span>;
}

export function Drafting({ comment, redraft }: { comment?: string | null; redraft: boolean }) {
  return (
    <div className="t-drafting" aria-live="polite">
      <div className="t-drafting-head">
        <Spinner size="sm" />
        <span className="t-drafting-title">
          {redraft ? "AI переделывает черновик по вашему комментарию…" : "AI готовит черновик…"}
        </span>
        <Elapsed />
      </div>
      {redraft && comment && <p className="t-drafting-comment">«{comment}»</p>}
      <p className="t-drafting-sub">
        Подбирает упражнения из каталога с учётом ограничений клиента и проверяет результат валидатором Skill.
        Обычно 10–60 секунд, карточка обновится сама.
      </p>
      <div className="t-skeleton-rows" aria-hidden="true">
        <span className="skeleton" style={{ width: "72%" }} />
        <span className="skeleton" style={{ width: "88%" }} />
        <span className="skeleton" style={{ width: "54%" }} />
      </div>
    </div>
  );
}

interface OutcomeProps {
  proposal: Proposal;
  readOnly: boolean;
  onDismiss?: () => void;
}

/** Итог: применено (с программой до/после), отклонено или ошибка применения. */
export function Outcome({ proposal: p, readOnly, onDismiss }: OutcomeProps) {
  const comment = p.decision?.comment;
  const dismiss = onDismiss && (
    <button className="btn btn-ghost btn-sm" onClick={onDismiss}>
      Скрыть
    </button>
  );

  if (p.status === "applied") {
    return (
      <div className="stack tight">
        <div className="t-outcome is-applied">
          <span className="t-outcome-mark" aria-hidden="true" />
          <div className="grow">
            <div className="t-outcome-title">Применено</div>
            <div className="t-outcome-text">
              Программа клиента обновлена, клиенту ушло уведомление в чат.
              {comment && <> Комментарий: «{comment}»</>}
            </div>
          </div>
          {dismiss}
        </div>
        {p.after && (
          <details className="t-details" open={!readOnly}>
            <summary>Программа после изменений</summary>
            <div className="t-plan-legend xs muted">
              <span className="t-lg is-added">новое</span>
              <span className="t-lg is-changed">изменён объём</span>
              <span className="t-lg is-removed">убрано</span>
            </div>
            {(() => {
              const v = changedOnly(p.after, p.before);
              return (
                <>
                  {v.shown < v.total && (
                    <p className="xs muted t-plan-note">
                      {p.after.title}: показаны изменённые дни — {v.shown} из {v.total}
                    </p>
                  )}
                  <PlanView plan={v.plan} compareTo={p.before} compact showTitle={v.shown === v.total} />
                </>
              );
            })()}
          </details>
        )}
      </div>
    );
  }

  if (p.status === "rejected" || p.status === "failed") {
    const failed = p.status === "failed";
    return (
      <div className={`t-outcome ${failed ? "is-failed" : "is-rejected"}`}>
        <span className="t-outcome-mark" aria-hidden="true" />
        <div className="grow">
          <div className="t-outcome-title">{failed ? "Не удалось применить" : "Отклонено"}</div>
          <div className="t-outcome-text">
            {failed
              ? p.reply || "AI не смог подготовить корректный черновик. Программа клиента не менялась."
              : "Программа клиента осталась прежней."}
            {comment && <> Комментарий: «{comment}»</>}
          </div>
        </div>
        {dismiss}
      </div>
    );
  }

  return null;
}
