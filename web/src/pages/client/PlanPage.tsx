/**
 * PlanPage — вкладка «Программа» клиента.
 *
 * Props:
 *   user: User              — текущий клиент.
 *   onOpenChat: () => void  — вернуться во вкладку «Коуч» («Попросить изменить программу»).
 *
 * GET /api/my/plan → PlanView или пустое состояние; ниже «Запросы на изменение»
 * (GET /api/my/proposals) со статусом и резюме ИИ. Оба списка обновляются раз в 10 с.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { errorText, myPlan, myProposals } from "../../api";
import type { Plan, Proposal, User } from "../../types";
import { weekdayLabel } from "../../format";
import PlanView from "../../components/PlanView";
import Empty from "../../components/Empty";
import Spinner from "../../components/Spinner";
import ProposalItem from "./ProposalItem";
import { ArrowRightIcon, BookIcon } from "./icons";
import "./client.css";
import "./plan.css";

export interface PlanPageProps {
  user: User;
  onOpenChat: () => void;
}

const POLL_MS = 10_000;

export default function PlanPage({ onOpenChat }: PlanPageProps) {
  const [plan, setPlan] = useState<Plan | null | undefined>(undefined);
  const [proposals, setProposals] = useState<Proposal[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  const load = useCallback(async (initial: boolean) => {
    const [p, props] = await Promise.allSettled([myPlan(), myProposals()]);
    if (!alive.current) return;
    if (p.status === "fulfilled") setPlan(p.value ?? null);
    if (props.status === "fulfilled") setProposals(props.value);
    const failed = [p, props].find((r): r is PromiseRejectedResult => r.status === "rejected");
    if (failed) {
      if (initial) setError(errorText(failed.reason));
    } else setError(null);
  }, []);

  useEffect(() => {
    alive.current = true;
    void load(true);
    const id = window.setInterval(() => {
      if (!document.hidden) void load(false);
    }, POLL_MS);
    return () => {
      alive.current = false;
      window.clearInterval(id);
    };
  }, [load]);

  if (plan === undefined && proposals === null) {
    if (error) {
      return (
        <div className="notice notice-error plan-error">
          <span className="grow">Не удалось загрузить программу: {error}</span>
          <button className="btn btn-sm btn-outline" onClick={() => void load(true)}>
            Повторить
          </button>
        </div>
      );
    }
    return <Spinner label="Загружаем программу" />;
  }

  // Неудачные и отклонённые черновики, которые тренер запускал сам, клиенту не нужны.
  const visible = (proposals ?? []).filter(
    (p) => !(p.source === "trainer" && (p.status === "failed" || p.status === "rejected")),
  );
  const sorted = [...visible].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  const active = sorted.filter((p) => p.status === "drafting" || p.status === "pending").length;

  return (
    <div className="client-plan">
      {error && plan === undefined && <div className="notice notice-error">{error}</div>}

      {plan ? <PlanHeader plan={plan} /> : null}
      {plan ? (
        <PlanView plan={plan} showTitle={false} />
      ) : plan === null ? (
        <div className="card flat">
          <Empty
            mark={<BookIcon size={20} />}
            title="Программы пока нет"
            text="Тренер ещё не назначил программу. Расскажите коучу о цели и графике — он передаст запрос тренеру."
            action={
              <button className="btn btn-primary" onClick={onOpenChat}>
                Написать коучу
              </button>
            }
          />
        </div>
      ) : null}

      {plan && (
        <button className="plan-cta" onClick={onOpenChat}>
          <span className="plan-cta-text">
            <span className="plan-cta-title">Хотите что-то поменять?</span>
            <span className="plan-cta-sub">Напишите коучу — он подготовит вариант, решение примет тренер.</span>
          </span>
          <ArrowRightIcon size={18} />
        </button>
      )}

      <section className="plan-section">
        <div className="plan-section-head">
          <h3>Запросы на изменение</h3>
          {active > 0 && <span className="chip status-pending chip-dot">{active} в работе</span>}
        </div>
        {proposals === null ? (
          <div className="muted small">Не удалось загрузить запросы.</div>
        ) : sorted.length === 0 ? (
          <div className="plan-noprops">
            Запросов пока нет. Попросите коуча, например: «Хочу добавить кардио в программу».
          </div>
        ) : (
          <div className="stack tight">
            {sorted.map((p) => (
              <ProposalItem key={p.id} p={p} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function PlanHeader({ plan }: { plan: Plan }) {
  const days = [...plan.days].sort((a, b) => a.day_index - b.day_index);
  const exercises = days.reduce((n, d) => n + d.exercises.length, 0);
  const weekdays = days.map((d) => weekdayLabel(d.weekday)).filter(Boolean);
  return (
    <div className="plan-hero">
      <div className="label">Активная программа</div>
      <h2 className="plan-hero-title">{plan.title}</h2>
      {plan.meta?.subtitle && <div className="plan-hero-sub">{plan.meta.subtitle}</div>}
      <div className="plan-stats">
        <Stat value={days.length} label={plural(days.length, ["день", "дня", "дней"])} />
        <Stat value={exercises} label={plural(exercises, ["упражнение", "упражнения", "упражнений"])} />
        {weekdays.length > 0 && (
          <div className="plan-weekdays">
            {weekdays.map((w, i) => (
              <span key={`${w}-${i}`} className="plan-weekday">
                {w}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ value, label }: { value: number; label: string }) {
  return (
    <div className="plan-stat">
      <span className="plan-stat-value num">{value}</span>
      <span className="plan-stat-label">{label}</span>
    </div>
  );
}

function plural(n: number, [one, few, many]: [string, string, string]): string {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
