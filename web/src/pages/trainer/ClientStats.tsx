// Боковые панели карточки клиента: питание за 14 дней, тренировки, вес.
import type { ReactNode } from "react";
import type { Nutrition14d, RecentSession, WeightPoint } from "../../types";
import { fmt0, fmt1 } from "../../format";
import { dayShort } from "./util";

function Panel({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="card t-panel">
      <div className="t-panel-head">
        <h3 className="t-panel-title">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function NutritionCard({ n }: { n: Nutrition14d }) {
  // Окно сервера включает сегодняшний день, поэтому days_logged бывает 15 — ограничиваем 14.
  const days = Math.min(n.days_logged, 14);
  const pct = Math.round((days / 14) * 100);
  const tiles: [string, number | null | undefined, string][] = [
    ["Ккал", n.avg_kcal, ""],
    ["Белки", n.avg_protein, "г"],
    ["Жиры", n.avg_fat, "г"],
    ["Углеводы", n.avg_carbs, "г"],
  ];
  return (
    <Panel title="Питание, 14 дней" aside={<span className="xs muted">среднее в день</span>}>
      {n.days_logged === 0 ? (
        <div className="notice notice-warn">Клиент не вёл дневник питания последние две недели.</div>
      ) : (
        <div className="t-tiles">
          {tiles.map(([label, v, unit]) => (
            <div key={label} className="t-tile">
              <div className="t-tile-value num">
                {fmt0(v)}
                {unit && v != null && <span className="t-tile-unit"> {unit}</span>}
              </div>
              <div className="t-tile-label">{label}</div>
            </div>
          ))}
        </div>
      )}
      <div className="t-meter" aria-label={`Записи в дневнике: ${days} из 14 дней`}>
        <div className="t-meter-bar">
          <span style={{ width: `${pct}%` }} />
        </div>
        <span className="xs muted num">записи {days} из 14 дней</span>
      </div>
    </Panel>
  );
}

export function SessionsCard({ sessions }: { sessions: RecentSession[] }) {
  return (
    <Panel title="Последние тренировки">
      {sessions.length === 0 ? (
        <p className="muted small">Завершённых тренировок пока нет.</p>
      ) : (
        <ul className="t-mini-list">
          {[...sessions]
            .sort((a, b) => b.completed_at.localeCompare(a.completed_at))
            .slice(0, 6)
            .map((s, i) => (
            <li key={`${s.completed_at}-${i}`}>
              <span className="grow truncate">{s.day_title}</span>
              {s.avg_rpe != null && <span className={`chip t-rpe ${s.avg_rpe >= 9 ? "is-high" : ""}`}>RPE {fmt1(s.avg_rpe)}</span>}
              <span className="xs muted nowrap">{dayShort(s.completed_at)}</span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function Sparkline({ points }: { points: number[] }) {
  if (points.length < 2) return null;
  const w = 240;
  const h = 48;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const xy = points.map((v, i) => [(i / (points.length - 1)) * w, h - 6 - ((v - min) / span) * (h - 12)]);
  const d = xy.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const [lx, ly] = xy[xy.length - 1];
  return (
    <svg className="t-spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      <circle cx={lx} cy={ly} r="3" fill="var(--accent)" />
    </svg>
  );
}

export function WeightsCard({ weights: raw, goal }: { weights: WeightPoint[]; goal: number | null }) {
  const weights = [...raw].sort((a, b) => a.measured_at.localeCompare(b.measured_at));
  const pts = weights.map((w) => w.weight_kg);
  const delta = pts.length >= 2 ? pts[pts.length - 1] - pts[0] : null;
  return (
    <Panel
      title="Вес"
      aside={
        delta != null && (
          <span className="xs muted num">
            {delta > 0 ? "+" : ""}
            {fmt1(delta)} кг за период
          </span>
        )
      }
    >
      {weights.length === 0 ? (
        <p className="muted small">Замеров веса нет.</p>
      ) : (
        <>
          <Sparkline points={pts} />
          <ul className="t-mini-list">
            {[...weights].reverse().slice(0, 4).map((w, i) => (
              <li key={`${w.measured_at}-${i}`}>
                <span className="grow num">{fmt1(w.weight_kg)} кг</span>
                <span className="xs muted nowrap">{dayShort(w.measured_at)}</span>
              </li>
            ))}
          </ul>
          {goal != null && <p className="xs muted">Цель: {fmt1(goal)} кг</p>}
        </>
      )}
    </Panel>
  );
}

export { Panel };
