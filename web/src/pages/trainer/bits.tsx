// Мелкие общие куски карточек тренера: шапка с клиентом, чип источника, цитата.
import type { ReactNode } from "react";
import type { Proposal } from "../../types";
import { initials } from "../../format";
import { SOURCE_TEXT } from "./util";

interface CardHeadProps {
  name: string;
  meta: ReactNode;
  clientId?: string;
  onOpenClient?: (clientId: string) => void;
  /** Чипы справа (источник, статус). */
  aside?: ReactNode;
  tone?: "default" | "danger";
}

/** Шапка карточки: аватар, имя клиента (ссылка на карточку клиента), подпись и чипы. */
export function CardHead({ name, meta, clientId, onOpenClient, aside, tone = "default" }: CardHeadProps) {
  const who = (
    <>
      <span className={`avatar t-avatar ${tone === "danger" ? "is-danger" : ""}`} aria-hidden="true">
        {initials(name)}
      </span>
      <span className="t-who-text">
        <span className="t-who-name truncate">{name}</span>
        <span className="t-who-meta">{meta}</span>
      </span>
    </>
  );
  return (
    <header className="t-card-head">
      {onOpenClient && clientId ? (
        <button
          type="button"
          className="t-who is-link"
          onClick={() => onOpenClient(clientId)}
          title="Открыть карточку клиента"
        >
          {who}
        </button>
      ) : (
        <div className="t-who">{who}</div>
      )}
      {aside && <div className="t-card-aside">{aside}</div>}
    </header>
  );
}

export function SourceChip({ source }: { source: Proposal["source"] }) {
  return <span className={`chip t-source is-${source}`}>{SOURCE_TEXT[source] ?? source}</span>;
}

/** Текст запроса клиента или тренера в кавычках-ёлочках. */
export function Quote({ children, label }: { children: string; label?: string }) {
  const text = children.trim().replace(/^[«"]+|[»"]+$/g, "");
  return (
    <figure className="t-quote">
      {label && <figcaption className="label">{label}</figcaption>}
      <blockquote>«{text || "—"}»</blockquote>
    </figure>
  );
}

/** Секция внутри карточки с подписью-лейблом. */
export function Section({ label, aside, children }: { label: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="t-section">
      <div className="t-section-head">
        <span className="label">{label}</span>
        {aside}
      </div>
      {children}
    </section>
  );
}
