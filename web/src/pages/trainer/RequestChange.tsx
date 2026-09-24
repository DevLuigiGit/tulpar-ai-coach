// «Попросить AI изменить программу»: POST /api/trainer/proposals → drafting → pending.
// Карточка черновика показывается прямо здесь; её опрос делает сама ProposalCard.
import { useState } from "react";
import { errorText, requestChange } from "../../api";
import type { Proposal } from "../../types";
import { navigate } from "../../route";
import Spinner from "../../components/Spinner";
import ProposalCard from "./ProposalCard";
import { isFinal } from "./util";

interface RequestChangeProps {
  clientId: string;
  hasPlan: boolean;
  onCreated?: (p: Proposal) => void;
  /** Правка применена — родитель перезагружает контекст клиента. */
  onApplied?: () => void;
}

const SUGGESTIONS = [
  "Замени приседания на упражнение, щадящее колено",
  "Снизь объём на ноги на 20% на две недели",
  "Добавь в конец каждой тренировки упражнение на кор",
];

export default function RequestChange({ clientId, hasPlan, onCreated, onApplied }: RequestChangeProps) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const drafting = proposal?.status === "drafting";
  const canSend = !!text.trim() && !sending && !drafting;

  async function submit() {
    if (!canSend) return;
    setSending(true);
    setError(null);
    try {
      const p = await requestChange(clientId, text.trim());
      setProposal(p);
      setText("");
      onCreated?.(p);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setSending(false);
    }
  }

  function handleChange(p: Proposal) {
    setProposal(p);
    if (p.status === "applied") onApplied?.();
  }

  return (
    <div className="stack">
      <section className="card t-ask">
        <div className="t-ask-head">
          <h3 className="t-panel-title">Попросить AI изменить программу</h3>
          <p className="xs muted">
            AI подготовит черновик по Skill и проверит его валидатором. Программа клиента не изменится, пока вы не
            нажмёте «Принять».
          </p>
        </div>
        {!hasPlan && (
          <div className="notice notice-warn">У клиента нет активной программы — черновик правки подготовить не получится.</div>
        )}
        <textarea
          className="textarea"
          rows={3}
          value={text}
          disabled={sending}
          aria-label="Что изменить в программе"
          placeholder="Например: колено беспокоит — замени приседания и выпады на щадящие упражнения"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void submit();
          }}
        />
        <div className="t-suggest">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="chip" onClick={() => setText(s)} disabled={sending}>
              {s}
            </button>
          ))}
        </div>
        {error && <div className="notice notice-error">{error}</div>}
        <div className="row">
          <button className="btn btn-primary" disabled={!canSend} onClick={() => void submit()}>
            {sending ? (
              <>
                <Spinner size="sm" />
                Отправляем…
              </>
            ) : (
              "Подготовить черновик"
            )}
          </button>
          {drafting && <span className="xs muted">Дождитесь текущего черновика</span>}
        </div>
      </section>

      {proposal?.status === "pending" && (
        <div className="notice notice-ok t-inqueue">
          <span className="grow">
            <strong>Черновик в очереди.</strong> Решите прямо здесь или во вкладке «Очередь».
          </span>
          <button className="btn btn-sm" onClick={() => navigate("queue")}>
            Открыть очередь
          </button>
        </div>
      )}

      {proposal && (
        <ProposalCard
          proposal={proposal}
          onChange={handleChange}
          onDismiss={isFinal(proposal.status) ? () => setProposal(null) : undefined}
        />
      )}
    </div>
  );
}
