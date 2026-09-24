import type { ReactNode } from "react";

interface EmptyProps {
  title: string;
  text?: ReactNode;
  /** Кнопка или ссылка под текстом. */
  action?: ReactNode;
  /** Мелкий значок в рамке над заголовком (символ или svg). */
  mark?: ReactNode;
}

/** Пустое состояние списка или экрана. */
export default function Empty({ title, text, action, mark }: EmptyProps) {
  return (
    <div className="empty">
      {mark !== undefined && <div className="empty-mark">{mark}</div>}
      <div className="empty-title">{title}</div>
      {text && <div className="empty-text">{text}</div>}
      {action && <div style={{ marginTop: 8 }}>{action}</div>}
    </div>
  );
}
