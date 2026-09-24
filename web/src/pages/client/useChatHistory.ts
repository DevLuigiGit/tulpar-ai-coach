// История чата: первая загрузка, опрос раз в 10 с (ответы тренера, записи еды)
// и пауза опроса, пока ждём ответ коуча — чтобы не мигали дубли сообщений.
import { useCallback, useEffect, useRef, useState } from "react";
import { errorText, history } from "../../api";
import type { ChatMessage } from "../../types";
import { isLocal, sameHistory } from "./chatModel";

export const POLL_MS = 10_000;

export type LoadState = { phase: "loading" } | { phase: "ready" } | { phase: "error"; message: string };

export function useChatHistory(paused: () => boolean, onFresh?: (list: ChatMessage[]) => void) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [load, setLoad] = useState<LoadState>({ phase: "loading" });
  const alive = useRef(true);
  const onFreshRef = useRef(onFresh);
  onFreshRef.current = onFresh;
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const refresh = useCallback(async (initial = false) => {
    try {
      const list = await history(50);
      if (!alive.current) return;
      onFreshRef.current?.(list);
      if (pausedRef.current()) {
        // Идёт отправка: сохраняем локальные сообщения поверх серверной истории.
        setMessages((prev) => [...list, ...prev.filter(isLocal)]);
      } else {
        setMessages((prev) => (sameHistory(prev, list) ? prev : list));
      }
      setLoad({ phase: "ready" });
    } catch (e) {
      if (!alive.current) return;
      // Ошибку показываем только когда ленты ещё нет; фоновые сбои опроса молча пропускаем.
      if (initial) setLoad({ phase: "error", message: errorText(e) });
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    void refresh(true);
    const id = window.setInterval(() => {
      if (document.hidden || pausedRef.current()) return;
      void refresh();
    }, POLL_MS);
    return () => {
      alive.current = false;
      window.clearInterval(id);
    };
  }, [refresh]);

  const retry = useCallback(() => {
    setLoad({ phase: "loading" });
    void refresh(true);
  }, [refresh]);

  return { messages, setMessages, load, refresh, retry };
}
