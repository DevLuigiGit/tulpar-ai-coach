// Запрос клиента на изменение программы: статус, текст запроса, краткое резюме ИИ,
// список правок и комментарий тренера.
import type { Proposal, ProposalStatus } from "../../types";
import { isProgramDraft } from "../../types";
import { relTime } from "../../format";
import StatusChip from "../../components/StatusChip";

const EXPLAIN: Partial<Record<ProposalStatus, string>> = {
  drafting: "ИИ готовит вариант правки — обычно это занимает до минуты.",
  pending: "Вариант готов и ждёт решения тренера.",
  applied: "Тренер одобрил — программа уже обновлена.",
  rejected: "Тренер отклонил этот вариант.",
  failed: "Не удалось подготовить правку. Попробуйте сформулировать запрос иначе.",
};

const OP_LABEL: Record<string, string> = {
  replace_exercise: "Замена",
  set_volume: "Объём",
  remove_exercise: "Убрать",
  add_exercise: "Добавить",
};

const MAX_CHANGES = 4;

export default function ProposalItem({ p }: { p: Proposal }) {
  // У failed резюме — техническое сообщение («модель недоступна»), клиенту его не показываем.
  const summary = isProgramDraft(p.draft) && p.status !== "failed" ? p.draft.summary : null;
  const changes = p.changes ?? [];
  const extra = changes.length - MAX_CHANGES;

  return (
    <article className={`card compact prop prop-${p.status}`}>
      <div className="prop-top">
        <StatusChip status={p.status} kind="program" />
        <span className="prop-src">{p.source === "trainer" ? "Инициатива тренера" : "Ваш запрос"}</span>
        <span className="grow" />
        <time className="prop-time">{relTime(p.updated_at || p.created_at)}</time>
      </div>

      <p className="prop-request">«{p.request}»</p>

      {summary && (
        <div className="prop-summary">
          <div className="prop-label">Что предлагает ИИ</div>
          <p>{summary}</p>
        </div>
      )}

      {changes.length > 0 && (
        <ul className="prop-changes">
          {changes.slice(0, MAX_CHANGES).map((c, i) => (
            <li key={i} className="prop-change">
              <span className="prop-change-op">{OP_LABEL[c.op] ?? c.op}</span>
              <span className="prop-change-body">
                <span className="prop-change-day">{c.day}: </span>
                {c.was && <span className="prop-change-was">{c.was}</span>}
                {c.was && <span className="prop-change-arrow"> → </span>}
                <span className="prop-change-new">{c.becomes}</span>
              </span>
            </li>
          ))}
          {extra > 0 && <li className="prop-change-more">и ещё {extra}</li>}
        </ul>
      )}

      {p.decision?.comment && (
        <div className="prop-comment">
          <div className="prop-label">Комментарий тренера</div>
          <p>{p.decision.comment}</p>
        </div>
      )}

      {EXPLAIN[p.status] && <div className="prop-explain">{EXPLAIN[p.status]}</div>}
    </article>
  );
}
