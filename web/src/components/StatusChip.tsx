import type { ProposalKind, ProposalStatus } from "../types";
import { STATUS_LABEL } from "../format";

interface StatusChipProps {
  status: ProposalStatus;
  /** Для эскалаций статус open подписывается «Нужен тренер» и красится красным. */
  kind?: ProposalKind;
  /** Переопределить подпись. */
  label?: string;
}

/** Чип статуса предложения/эскалации. drafting показывает мини-спиннер. */
export default function StatusChip({ status, kind, label }: StatusChipProps) {
  const cls = kind === "escalation" && status === "open" ? "status-escalation" : `status-${status}`;
  return (
    <span className={`chip ${cls} ${status === "drafting" ? "" : "chip-dot"}`}>
      {status === "drafting" && <span className="spinner spinner-sm" aria-hidden="true" />}
      {label ?? STATUS_LABEL[status] ?? status}
    </span>
  );
}
