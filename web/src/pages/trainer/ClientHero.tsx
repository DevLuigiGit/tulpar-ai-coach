// Шапка карточки клиента: профиль и заметка тренера (травма подсвечена).
import type { ClientContext, TrainerNote } from "../../types";
import {
  activityLabel,
  fmt1,
  goalLabel,
  initials,
  levelLabel,
  placeLabel,
  sexLabel,
  weekdayLabel,
} from "../../format";

export function ClientHero({ ctx }: { ctx: ClientContext }) {
  const p = ctx.profile;
  const latest = [...ctx.weights].sort((a, b) => b.measured_at.localeCompare(a.measured_at))[0];
  const lastWeight = latest ? latest.weight_kg : null;
  const weight = lastWeight ?? p.weight_kg;
  const days = (p.training_days ?? []).map((d) => weekdayLabel(d)).join(", ");
  const facts: [string, string][] = [
    ["Пол", sexLabel(p.sex)],
    ["Возраст", p.age != null ? `${p.age}` : "—"],
    ["Рост", p.height_cm != null ? `${p.height_cm} см` : "—"],
    ["Вес", weight != null ? `${fmt1(weight)} кг` : "—"],
    ["Цель по весу", p.goal_weight_kg != null ? `${fmt1(p.goal_weight_kg)} кг` : "—"],
    ["Активность", activityLabel(p.activity)],
    ["Дни тренировок", days || "—"],
  ];

  return (
    <section className="card t-hero">
      <div className="t-hero-top">
        <span className="avatar avatar-lg t-hero-avatar" aria-hidden="true">
          {initials(ctx.name)}
        </span>
        <div className="grow">
          <h2 className="t-hero-name">{ctx.name}</h2>
          <div className="row wrap t-hero-chips">
            <span className="chip chip-accent">{goalLabel(p.goal)}</span>
            <span className="chip">{levelLabel(p.level)}</span>
            <span className="chip">{placeLabel(p.place)}</span>
            {ctx.trainer_note?.injury && <span className="chip t-injury">ограничение</span>}
          </div>
        </div>
      </div>
      <dl className="t-facts">
        {facts.map(([k, v]) => (
          <div key={k} className="t-fact">
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

/** Заметка тренера. С травмой — янтарно-красная плашка: AI учитывает её в черновиках. */
export function NoteCard({ note }: { note: TrainerNote | null }) {
  if (!note || !note.body?.trim()) {
    return (
      <section className="t-note">
        <div className="t-note-title">Заметка тренера</div>
        <p className="t-note-body muted">Заметки нет. Ограничения и травмы, записанные здесь, AI учитывает в черновиках.</p>
      </section>
    );
  }
  return (
    <section className={`t-note ${note.injury ? "is-injury" : ""}`}>
      <div className="t-note-title">
        {note.injury ? "Ограничение по здоровью" : "Заметка тренера"}
        {note.injury && <span className="t-note-tag">AI учитывает в черновиках</span>}
      </div>
      <p className="t-note-body">{note.body}</p>
    </section>
  );
}
