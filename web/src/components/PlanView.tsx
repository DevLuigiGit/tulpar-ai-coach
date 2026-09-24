import type { Plan, PlanDay, PlanExercise } from "../types";
import { equipmentLabel, weekdayLabel } from "../format";

export type ExerciseMark = "added" | "changed" | "removed";

interface PlanViewProps {
  plan: Plan;
  /** Плотный вид для карточек предложений и боковых панелей. */
  compact?: boolean;
  /** Показать заголовок программы над днями (по умолчанию да). */
  showTitle?: boolean;
  /**
   * Сравнить с другой версией (обычно proposal.before при показе proposal.after):
   * новые упражнения подсвечиваются зелёным, изменённые подходы/повторы — янтарным,
   * удалённые показываются зачёркнутыми.
   */
  compareTo?: Plan | null;
}

/** Программа тренировок: дни и упражнения с подходами × повторами. */
export default function PlanView({ plan, compact = false, showTitle = true, compareTo }: PlanViewProps) {
  const days = [...plan.days].sort((a, b) => a.day_index - b.day_index);
  return (
    <div className={`plan ${compact ? "is-compact" : ""}`}>
      {showTitle && (
        <div>
          <div className="plan-title">{plan.title}</div>
          {plan.meta?.subtitle && <div className="muted small">{plan.meta.subtitle}</div>}
        </div>
      )}
      <div className="plan-days">
        {days.map((day) => (
          <DayBlock key={day.id} day={day} before={findDay(compareTo, day)} compare={!!compareTo} />
        ))}
      </div>
    </div>
  );
}

function findDay(plan: Plan | null | undefined, day: PlanDay): PlanDay | undefined {
  if (!plan) return undefined;
  return plan.days.find((d) => d.day_index === day.day_index) ?? plan.days.find((d) => d.id === day.id);
}

function sameExercise(a: PlanExercise, b: PlanExercise) {
  return a.id === b.id || a.exercise_id === b.exercise_id;
}

function DayBlock({ day, before, compare }: { day: PlanDay; before?: PlanDay; compare: boolean }) {
  const rows: { ex: PlanExercise; mark: ExerciseMark | null }[] = day.exercises.map((ex) => {
    if (!compare) return { ex, mark: null };
    const prev = before?.exercises.find((b) => sameExercise(b, ex));
    if (!prev) return { ex, mark: "added" };
    const changed = prev.target_sets !== ex.target_sets || String(prev.target_reps) !== String(ex.target_reps);
    return { ex, mark: changed ? "changed" : null };
  });
  if (compare && before) {
    for (const prev of before.exercises) {
      if (!day.exercises.some((ex) => sameExercise(prev, ex))) rows.push({ ex: prev, mark: "removed" });
    }
  }
  const weekday = weekdayLabel(day.weekday);

  return (
    <section className="plan-day">
      <header className="plan-day-head">
        <span className="plan-day-index">{day.day_index + 1}</span>
        <span className="plan-day-title">{day.title}</span>
        {weekday && <span className="plan-day-weekday">{weekday}</span>}
      </header>
      {rows.length === 0 ? (
        <div className="plan-ex muted small">День отдыха</div>
      ) : (
        <ul className="plan-ex-list">
          {rows.map(({ ex, mark }) => (
            <li key={`${ex.id}-${mark ?? ""}`} className={`plan-ex ${mark ? `is-${mark}` : ""}`}>
              <div className="plan-ex-main">
                <div className="plan-ex-name truncate">{ex.exercise_name}</div>
                <div className="plan-ex-meta">
                  {[ex.muscle_group, ex.equipment ? equipmentLabel(ex.equipment) : null].filter(Boolean).join(" · ")}
                  {mark === "added" && " · новое"}
                  {mark === "removed" && " · убрано"}
                </div>
              </div>
              <div className="plan-ex-dose">
                {ex.target_sets} × {ex.target_reps}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
