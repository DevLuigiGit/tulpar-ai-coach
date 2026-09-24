// Поле ввода чата: текст, фото, голосовое (файлом), отправка. Состояние черновика
// держит ChatPage — чтобы вернуть его в поле, если запрос не прошёл.
import { useLayoutEffect, useRef, type ChangeEvent, type KeyboardEvent } from "react";
import Spinner from "../../components/Spinner";
import { fileSizeLabel } from "./chatModel";
import { CameraIcon, CloseIcon, MicIcon, SendIcon, WaveIcon } from "./icons";
import "./composer.css";

export interface Draft {
  text: string;
  photo: File | null;
  audio: File | null;
}

interface ComposerProps {
  draft: Draft;
  photoPreview: string | null;
  busy: boolean;
  onChange: (next: Draft) => void;
  onSubmit: () => void;
}

export const draftReady = (d: Draft) => !!(d.text.trim() || d.photo || d.audio);

export default function Composer({ draft, photoPreview, busy, onChange, onSubmit }: ComposerProps) {
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const photoRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);
  const canSend = !busy && draftReady(draft);

  // Авто-высота textarea: от одной строки до ~5.
  useLayoutEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 132)}px`;
  }, [draft.text]);

  const pick = (key: "photo" | "audio") => (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    e.target.value = "";
    if (file) onChange({ ...draft, [key]: file });
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (canSend) onSubmit();
    }
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSend) onSubmit();
      }}
    >
      {(draft.photo || draft.audio) && (
        <div className="composer-atts">
          {draft.photo && (
            <span className="att-chip">
              {photoPreview ? (
                <img className="att-thumb" src={photoPreview} alt="" />
              ) : (
                <CameraIcon size={14} />
              )}
              <span className="truncate">{draft.photo.name || "Фото"}</span>
              <span className="att-size">{fileSizeLabel(draft.photo.size)}</span>
              <button
                type="button"
                className="att-remove"
                aria-label="Убрать фото"
                disabled={busy}
                onClick={() => onChange({ ...draft, photo: null })}
              >
                <CloseIcon size={12} />
              </button>
            </span>
          )}
          {draft.audio && (
            <span className="att-chip">
              <WaveIcon size={14} />
              <span className="truncate">{draft.audio.name || "Голосовое"}</span>
              <span className="att-size">{fileSizeLabel(draft.audio.size)}</span>
              <button
                type="button"
                className="att-remove"
                aria-label="Убрать голосовое"
                disabled={busy}
                onClick={() => onChange({ ...draft, audio: null })}
              >
                <CloseIcon size={12} />
              </button>
            </span>
          )}
        </div>
      )}

      <div className="composer-row">
        <button
          type="button"
          className={`composer-tool ${draft.photo ? "is-on" : ""}`}
          aria-label="Прикрепить фото еды"
          title="Фото еды"
          disabled={busy}
          onClick={() => photoRef.current?.click()}
        >
          <CameraIcon />
        </button>
        <button
          type="button"
          className={`composer-tool ${draft.audio ? "is-on" : ""}`}
          aria-label="Прикрепить голосовое"
          title="Голосовое сообщение"
          disabled={busy}
          onClick={() => audioRef.current?.click()}
        >
          <MicIcon />
        </button>
        <textarea
          ref={areaRef}
          className="composer-input"
          rows={1}
          placeholder={draft.photo || draft.audio ? "Добавьте подпись…" : "Спросите коуча…"}
          value={draft.text}
          maxLength={2000}
          onChange={(e) => onChange({ ...draft, text: e.target.value })}
          onKeyDown={onKey}
        />
        <button type="submit" className="composer-send" aria-label="Отправить" disabled={!canSend}>
          {busy ? <Spinner size="sm" /> : <SendIcon size={18} />}
        </button>
      </div>

      <input ref={photoRef} type="file" accept="image/*" hidden onChange={pick("photo")} />
      <input ref={audioRef} type="file" accept="audio/*" hidden onChange={pick("audio")} />
    </form>
  );
}
