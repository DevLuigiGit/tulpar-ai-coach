// Отправка в чат: черновик (текст + фото + голосовое), оптимистичное сообщение,
// защита от двойной отправки, возврат черновика в поле при ошибке.
import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { chat, chatForm, errorText } from "../../api";
import type { ChatMessage } from "../../types";
import { draftReady, type Draft } from "./Composer";
import { MAX_FILE_BYTES, localMessage, msgKey, replyToMessage } from "./chatModel";

const EMPTY: Draft = { text: "", photo: null, audio: null };

export interface Sending {
  since: number;
  withMedia: boolean;
}

interface Options {
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  refresh: () => Promise<void>;
  /** Сохранить превью фото за сообщением (локальный id). */
  onPreview: (messageId: string, url: string) => void;
  onSent: () => void;
}

export function useSendMessage({ setMessages, refresh, onPreview, onSent }: Options) {
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

      setMessages((m) => [...m, mine]);
      if (quick === undefined) {
        setDraft(EMPTY);
        setPhotoPreview(null);
      }
      setError(null);
      setSending({ since: Date.now(), withMedia: !!(d.photo || d.audio) });
      onSent();

      try {
        const reply = await chat(chatForm({ text, photo: d.photo, audio: d.audio }));
        setMessages((m) => [...m, replyToMessage(reply)]);
        busy.current = false;
        setSending(null);
        void refresh();
      } catch (e) {
        busy.current = false;
        setSending(null);
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
    [draft, photoPreview, setMessages, refresh, onPreview, onSent],
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
