// Кнопка «Прослушать» под ответом коуча: POST /api/tts → MP3 → общий <audio>.
// Одновременно звучит один ответ: новая кнопка останавливает предыдущую, повторное нажатие — стоп.
// MP3 кешируется на время сессии, поэтому повторное прослушивание не ходит на сервер.
import { useCallback, useEffect, useState } from "react";
import { errorText, speech } from "../../api";
import { SpeakerIcon, StopIcon } from "./icons";

type State = "idle" | "loading" | "playing";

// Пустой WAV: iOS Safari разрешает play() только внутри жеста пользователя, а MP3 приходит
// после запроса. Запускаем «тишину» синхронно в клике — дальше этот элемент играет без жеста.
const SILENT_WAV = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

const cache = new Map<string, string>(); // текст ответа → object URL с MP3
let player: HTMLAudioElement | null = null;
let owner: (() => void) | null = null; // сброс кнопки, которая сейчас владеет плеером

function getPlayer(): HTMLAudioElement {
  if (!player) player = new Audio();
  return player;
}

/** Останавливает звук и возвращает кнопку-владельца в исходное состояние. */
function release(): void {
  player?.pause();
  const reset = owner;
  owner = null;
  reset?.();
}

async function audioUrl(text: string): Promise<string> {
  const hit = cache.get(text);
  if (hit) return hit;
  const url = URL.createObjectURL(await speech(text));
  cache.set(text, url);
  return url;
}

function playError(e: unknown): string {
  if (e instanceof DOMException && e.name === "NotAllowedError") return "Нажмите ещё раз";
  return errorText(e);
}

export default function SpeakButton({ text }: { text: string }) {
  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState<string | null>(null);
  const reset = useCallback(() => setState("idle"), []);

  // Ушли со страницы — звук не должен продолжаться без кнопки «стоп».
  useEffect(() => () => {
    if (owner === reset) release();
  }, [reset]);

  async function toggle() {
    if (state !== "idle") {
      if (owner === reset) release();
      return;
    }
    const audio = getPlayer();
    release();
    owner = reset;
    setError(null);
    if (!cache.has(text)) {
      audio.src = SILENT_WAV;
      audio.play().catch(() => undefined);
      setState("loading");
    }
    try {
      const url = await audioUrl(text);
      if (owner !== reset) return; // остановили или включили другой ответ, пока ждали
      audio.src = url;
      audio.onended = () => {
        if (owner === reset) release();
      };
      await audio.play();
      if (owner === reset) setState("playing");
    } catch (e) {
      if (owner !== reset) return;
      release();
      setError(playError(e));
    }
  }

  const label = state === "idle" ? "Прослушать ответ" : state === "loading" ? "Готовлю озвучку…" : "Остановить";
  return (
    <>
      <button
        type="button"
        className={`speak-btn ${state !== "idle" ? "is-active" : ""}`}
        onClick={toggle}
        aria-label={label}
        title={label}
      >
        {state === "loading" ? (
          <span className="spinner spinner-sm" aria-hidden="true" />
        ) : state === "playing" ? (
          <StopIcon size={14} />
        ) : (
          <SpeakerIcon size={14} />
        )}
      </button>
      {error && (
        <span className="speak-error" role="status">
          {error}
        </span>
      )}
    </>
  );
}
