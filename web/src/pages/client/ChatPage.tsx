/**
 * ChatPage — вкладка «Коуч» клиента.
 *
 * Props (App/Shells их передают, менять сигнатуру нельзя):
 *   user: User            — текущий клиент (role === "client").
 *   onOpenPlan: () => void — переключить на вкладку «Программа» (например, из ответа kind="proposal").
 *
 * Лента истории (GET /api/chat/history, опрос раз в 10 с), пузыри по ролям, ссылки на
 * источники, карточки еды, заметки тренера; снизу — поле ввода с фото и голосовым.
 */
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage, User } from "../../types";
import Spinner from "../../components/Spinner";
import { ageMs, asReply, attachments, isLocal, loggedCardIds, msgKey, transcriptAfter } from "./chatModel";
import { useChatHistory } from "./useChatHistory";
import { useSendMessage } from "./useSendMessage";
import { ChatEmpty, TypingIndicator } from "./ChatBits";
import Composer from "./Composer";
import MessageBubble from "./MessageBubble";
import DaySeparator, { dayKey } from "./DaySeparator";
import "./client.css";
import "./chat.css";
import "./chat-parts.css";

export interface ChatPageProps {
  user: User;
  onOpenPlan: () => void;
}

const RECENT_MS = 3 * 60_000;

export default function ChatPage({ onOpenPlan }: ChatPageProps) {
  const busyRef = useRef<() => boolean>(() => false);
  const paused = useCallback(() => busyRef.current(), []);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [loggedHere, setLoggedHere] = useState<Set<string>>(() => new Set());
  const pendingPreview = useRef<string | null>(null);
  const feedRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  // Фото, отправленное в этой сессии, показываем и после замены локального id на серверный.
  const adoptPreview = useCallback((list: ChatMessage[]) => {
    const url = pendingPreview.current;
    if (!url) return;
    const mine = [...list]
      .reverse()
      .find((m) => m.role === "user" && attachments(m).photo && ageMs(m.created_at) < RECENT_MS);
    if (!mine) return;
    pendingPreview.current = null;
    setPreviews((p) => ({ ...p, [msgKey(mine)]: url }));
  }, []);

  const { messages, setMessages, load, refresh, retry } = useChatHistory(paused, adoptPreview);

  const sender = useSendMessage({
    setMessages,
    refresh: () => refresh(),
    onPreview: (id, url) => {
      pendingPreview.current = url;
      setPreviews((p) => ({ ...p, [id]: url }));
    },
    onSent: () => {
      stick.current = true;
    },
  });
  busyRef.current = sender.isBusy;

  const logged = useMemo(() => {
    const s = loggedCardIds(messages);
    loggedHere.forEach((id) => s.add(id));
    return s;
  }, [messages, loggedHere]);

  // Автопрокрутка вниз, если пользователь и так был внизу ленты.
  useLayoutEffect(() => {
    const el = feedRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages, sender.sending, load.phase]);

  const onScroll = () => {
    const el = feedRef.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  const onMealLogged = useCallback(
    (cardId: string) => {
      setLoggedHere((s) => new Set(s).add(cardId));
      void refresh();
    },
    [refresh],
  );

  const showEmpty = load.phase === "ready" && messages.length === 0 && !sender.sending;

  return (
    <div className="page-fill chat">
      <div className="chat-feed" ref={feedRef} onScroll={onScroll}>
        {load.phase === "loading" && messages.length === 0 && <Spinner label="Загружаем переписку" />}
        {load.phase === "error" && messages.length === 0 && (
          <div className="notice notice-error chat-load-error">
            <span>Не удалось загрузить переписку: {load.message}</span>
            <button className="btn btn-sm btn-outline" onClick={retry}>
              Повторить
            </button>
          </div>
        )}
        {showEmpty && <ChatEmpty onPick={(t) => void sender.send(t)} disabled={!!sender.sending} />}

        {messages.map((m, i) => {
          const prev = messages[i - 1];
          const newDay = !prev || dayKey(prev.created_at) !== dayKey(m.created_at);
          const afterVoice = !!prev && prev.role === "user" && attachments(prev).audio;
          const cardId = asReply(m.payload)?.meal?.card_id;
          return (
            <div key={m.id} className={`chat-row ${isLocal(m) ? "is-local" : ""}`}>
              {newDay && <DaySeparator iso={m.created_at} />}
              <MessageBubble
                message={m}
                transcript={m.role === "user" && attachments(m).audio ? transcriptAfter(messages, i) : null}
                hideTranscript={afterVoice}
                preview={previews[msgKey(m)]}
                mealLogged={!!cardId && logged.has(cardId)}
                onMealLogged={onMealLogged}
                onOpenPlan={onOpenPlan}
              />
            </div>
          );
        })}

        {sender.sending && <TypingIndicator since={sender.sending.since} withMedia={sender.sending.withMedia} />}
      </div>

      <div className="chat-bottom">
        {sender.error && (
          <div className="notice notice-error chat-send-error" role="alert">
            <span className="grow">{sender.error}</span>
            <button className="btn btn-ghost btn-sm" onClick={sender.clearError}>
              Скрыть
            </button>
          </div>
        )}
        <Composer
          draft={sender.draft}
          photoPreview={sender.photoPreview}
          busy={!!sender.sending}
          onChange={sender.updateDraft}
          onSubmit={() => void sender.send()}
        />
      </div>
    </div>
  );
}
