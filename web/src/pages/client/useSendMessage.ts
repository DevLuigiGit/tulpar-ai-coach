// Отправка в чат: черновик (текст + фото + голосовое), оптимистичное сообщение,
// защита от двойной отправки, возврат черновика в поле при ошибке.
// Ответ приходит потоком (POST /api/chat/stream): этапы и текст растут в ожидающем пузыре.
// Если поток не начался — обычный POST /api/chat; если оборвался на середине — ответ
// дописывает сервер, забираем его из истории (повторная отправка дала бы дубль).
import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { ApiError, StreamError, chat, chatForm, chatStream, errorText, history, type StreamStage } from "../../api";
import type { ChatMessage, ChatReply } from "../../types";
import { draftReady, type Draft } from "./Composer";
import { MAX_FILE_BYTES, localMessage, msgKey, replyToMessage } from "./chatModel";

const EMPTY: Draft = { text: "", photo: null, audio: null };
const RECOVER_EVERY_MS = 2500;
const RECOVER_FOR_MS = 90_000;

export interface Sending {
  since: number;
  withMedia: boolean;
  /** Этап хода с сервера (стриминг); null — ещё не пришёл или обычный запрос. */
  stage: StreamStage | null;
  /** Уже показанная часть ответа. */
  text: string;
}

interface Options {
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  refresh: () => Promise<void>;
  /** Наибольший серверный id в ленте — чтобы найти ответ в истории после обрыва потока. */
  lastServerId: () => number;
  /** Сохранить превью фото за сообщением (локальный id). */
  onPreview: (messageId: string, url: string) => void;
  onSent: () => void;
}

/** Обрыв посреди потока: сервер допишет ответ сам — ждём его в истории. */
class Pending extends Error {}

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

async function recoverFromHistory(afterId: number): Promise<ChatMessage[] | null> {
  const until = Date.now() + RECOVER_FOR_MS;
  while (Date.now() < until) {
    await sleep(RECOVER_EVERY_MS);
    try {
      const list = await history(50);
      const fresh = list.filter((m) => typeof m.id === "number" && m.id > afterId);
      const mine = fresh.findIndex((m) => m.role === "user");
      if (mine >= 0 && fresh.slice(mine + 1).some((m) => m.role === "assistant")) return list;
    } catch {
      /* сеть ещё не вернулась — пробуем снова */
    }
  }
  return null;
}

export function useSendMessage({ setMessages, refresh, lastServerId, onPreview, onSent }: Options) {
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [sending, setSending] = useState<Sending | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);
  const urls = useRef<string[]>([]);

  useEffect(() => () => urls.current.forEach((u) => URL.revokeObjectURL(u)), []);

  const updateDraft = useCallback(
    (next: Draft) => {
      for (const f of [next.photo, next.audio]) {
        if (f && f.size > MAX_FILE_BYTES) {
          setError("Файл больше 8 МБ — выберите поменьше");
          return;
        }
      }
      if (next.photo !== draft.photo) {
        if (photoPreview) URL.revokeObjectURL(photoPreview);
        const url = next.photo ? URL.createObjectURL(next.photo) : null;
        if (url) urls.current.push(url);
        setPhotoPreview(url);
      }
      setError(null);
      setDraft(next);
    },
    [draft.photo, photoPreview],
  );

  /** Поток → при сбое до начала хода обычный запрос → при обрыве посреди хода ответ из истории. */
  const deliver = useCallback(
    async (form: FormData, afterId: number): Promise<ChatReply | ChatMessage[]> => {
      try {
        return await chatStream(form, {
          onStage: (stage) => setSending((s) => (s ? { ...s, stage } : s)),
          onDelta: (t) => setSending((s) => (s ? { ...s, stage: "answer", text: s.text + t } : s)),
        });
      } catch (e) {
        if (!(e instanceof StreamError)) throw e;
        if (e.phase === "before") {
          setSending((s) => (s ? { ...s, stage: null, text: "" } : s));
          return await chat(form);
        }
        if (e.phase === "server") throw new ApiError(500, e.message);
        setSending((s) => (s ? { ...s, text: "" } : s));
        const list = await recoverFromHistory(afterId);
        if (list) return list;
        throw new Pending("Связь прервалась. Ответ появится в чате, как только будет готов.");
      }
    },
    [],
  );

  const send = useCallback(
    async (quick?: string) => {
      if (busy.current) return;
      const d: Draft = quick !== undefined ? { ...EMPTY, text: quick } : draft;
      if (!draftReady(d)) return;
      busy.current = true;

      const text = d.text.trim();
      const shown = text || (d.photo ? "[фото]" : "[голосовое]");
      const mine = localMessage("user", shown, { has_photo: !!d.photo, has_audio: !!d.audio });
      const preview = quick === undefined && d.photo ? photoPreview : null;
      if (preview) onPreview(msgKey(mine), preview);
      const afterId = lastServerId();

      setMessages((m) => [...m, mine]);
      if (quick === undefined) {
        setDraft(EMPTY);
        setPhotoPreview(null);
      }
      setError(null);
      setSending({ since: Date.now(), withMedia: !!(d.photo || d.audio), stage: null, text: "" });
      onSent();

      try {
        const out = await deliver(chatForm({ text, photo: d.photo, audio: d.audio }), afterId);
        if (Array.isArray(out)) setMessages(out);
        else setMessages((m) => [...m, replyToMessage(out)]);
        busy.current = false;
        setSending(null);
        void refresh();
      } catch (e) {
        busy.current = false;
        setSending(null);
        if (e instanceof Pending) {
          // Сообщение уже на сервере: черновик не возвращаем, ответ подтянет опрос истории.
          setError(e.message);
          void refresh();
          return;
        }
        setMessages((m) => m.filter((x) => x.id !== mine.id));
        setError(errorText(e));
        if (quick === undefined) {
          // Возвращаем черновик, если пользователь не начал писать новый.
          setDraft((cur) => (draftReady(cur) ? cur : d));
          if (preview) setPhotoPreview((cur) => cur ?? preview);
        }
        void refresh();
      }
    },
    [draft, photoPreview, setMessages, refresh, lastServerId, onPreview, onSent, deliver],
  );

  return {
    draft,
    photoPreview,
    sending,
    error,
    clearError: () => setError(null),
    updateDraft,
    send,
    isBusy: () => busy.current,
  };
}
