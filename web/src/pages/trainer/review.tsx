// Разбор черновика для тренера: таблица «было → станет» и результат проверки Skill.
import type { Change, Violation } from "../../types";
import { OP_TEXT, plural } from "./util";

function worst(violations: Violation[] | null | undefined, index: number): "error" | "warning" | null {
  const mine = (violations ?? []).filter((v) => v.op_index === index);
  if (mine.some((v) => v.severity === "error")) return "error";
  if (mine.length) return "warning";
  return null;
}

/** «День · Было → Станет · Почему» по proposal.changes; строки с замечаниями валидатора подсвечены. */
export function ChangesTable({ changes, violations }: { changes: Change[]; violations?: Violation[] | null }) {
  if (!changes.length) {
    return <p className="muted small">Черновик не содержит изменений программы.</p>;
  }
  return (
    <div className="t-changes-wrap">
      <table className="t-changes">
        <thead>
          <tr>
            <th scope="col">День</th>
            <th scope="col">Было → Станет</th>
            <th scope="col">Почему</th>
          </tr>
        </thead>
        <tbody>
          {changes.map((c, i) => {
            const flag = worst(violations, i);
            const removing = c.op === "remove_exercise";
            return (
              <tr key={i} className={flag ? `has-${flag}` : ""}>
                <td className="t-ch-day">
                  <span className="t-ch-day-title">{c.day}</span>
                  <span className={`t-op is-${c.op}`}>{OP_TEXT[c.op] ?? c.op}</span>
                </td>
                <td className="t-ch-diff">
                  {c.was ? <span className="t-was">{c.was}</span> : <span className="t-was is-none">нет</span>}
                  <span className="t-arrow" aria-label="станет">→</span>
                  <span className={`t-becomes ${removing ? "is-remove" : ""}`}>{c.becomes}</span>
                </td>
                <td className="t-ch-why">{c.reason || <span className="faint">—</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Замечания валидатора Skill: ошибки красным, предупреждения янтарным. */
export function SkillCheck({ violations }: { violations: Violation[] | null | undefined }) {
  const list = [...(violations ?? [])].sort(
    (a, b) => (a.severity === "error" ? 0 : 1) - (b.severity === "error" ? 0 : 1),
  );
  const errors = list.filter((v) => v.severity === "error").length;
  const warnings = list.length - errors;
  const tone = errors ? "error" : warnings ? "warning" : "ok";
  const verdict = errors
    ? `${errors} ${plural(errors, ["ошибка", "ошибки", "ошибок"])}`
    : warnings
      ? `${warnings} ${plural(warnings, ["предупреждение", "предупреждения", "предупреждений"])}`
      : "нарушений нет";

  return (
    <section className={`t-skill is-${tone}`}>
      <div className="t-skill-head">
        <span className="t-skill-title">Проверка Skill</span>
        <span className="t-skill-sub">tulpar-program-builder · validate_plan</span>
        <span className={`t-skill-verdict is-${tone}`}>{verdict}</span>
      </div>
      {list.length > 0 && (
        <ul className="t-skill-list">
          {list.map((v, i) => (
            <li key={`${v.code}-${i}`} className={`t-viol is-${v.severity}`}>
              <span className="t-viol-sev">{v.severity === "error" ? "Ошибка" : "Внимание"}</span>
              <span className="t-viol-msg">
                {v.message}
                {v.op_index != null && <span className="faint"> · изменение {v.op_index + 1}</span>}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
