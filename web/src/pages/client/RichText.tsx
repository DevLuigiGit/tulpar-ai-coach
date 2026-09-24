// Лёгкий рендер ответа коуча: абзацы, маркированные списки, **жирный** и сноски [n].
// Без innerHTML — текст модели не может внедрить разметку.
import type { ReactNode } from "react";

const INLINE = /(\*\*[^*]+\*\*|\[\d{1,2}\])/g;

function inline(text: string, keyBase: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const key = `${keyBase}-${i}`;
    if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (/^\[\d{1,2}\]$/.test(part)) return <sup key={key} className="cite-ref">{part.slice(1, -1)}</sup>;
    return part;
  });
}

const BULLET = /^\s*(?:[-*•]|\d+[.)])\s+/;

export default function RichText({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) {
      const k = `p${blocks.length}`;
      blocks.push(
        <p key={k}>
          {para.map((l, i) => (
            <span key={i}>
              {i > 0 && <br />}
              {inline(l, `${k}-${i}`)}
            </span>
          ))}
        </p>,
      );
      para = [];
    }
  };
  const flushList = () => {
    if (list.length) {
      const k = `l${blocks.length}`;
      blocks.push(
        <ul key={k}>
          {list.map((l, i) => (
            <li key={i}>{inline(l, `${k}-${i}`)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      flushPara();
      flushList();
    } else if (BULLET.test(line)) {
      flushPara();
      list.push(line.replace(BULLET, ""));
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return <div className="rich">{blocks}</div>;
}
