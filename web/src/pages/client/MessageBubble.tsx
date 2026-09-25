// Одно сообщение ленты: пузырь клиента справа, ответы коуча/тренера слева,
// служебные заметки по центру. Карточка еды рендерится под ответом коуча,
// под ответом коуча — время и оценка 👍/👎.
import type { ChatMessage, Citation, ConfirmMealResponse } from "../../types";
import { SOURCE_LABEL, timeShort } from "../../format";
import { asReply, asSystem, attachments, canRate, userText } from "./chatModel";
import AnswerFeedback from "./AnswerFeedback";
import MealCard from "./MealCard";
import RichText from "./RichText";
import { AlertIcon, ArrowRightIcon, CheckIcon, ImageIcon, UserTieIcon, WaveIcon } from "./icons";

interface BubbleProps {
  message: ChatMessage;
  /** Для голосового клиента: распознанный текст из следующего ответа. */
  transcript?: string | null;
  /** Не показывать «Распознано» у ответа — уже показано у голосового. */
  hideTranscript?: boolean;
  /** Локальное превью фото, пока сессия жива. */
  preview?: string;
  mealLogged?: boolean;
  onMealLogged?: (cardId: string, r: ConfirmMealResponse) => void;
  onOpenPlan: () => void;
}

export default function MessageBubble(props: BubbleProps) {
  const { message } = props;
  if (message.role === "user") return <UserBubble {...props} />;
  const system = asSystem(message.payload);
  if (system === "meal_logged") {
    return (
      <div className="msg-system" role="note">
        <CheckIcon size={14} />
        <span>{message.text}</span>
      </div>
    );
  }
  if (system === "trainer_reply") return <TrainerBubble message={message} />;
  return <CoachBubble {...props} />;
}

function UserBubble({ message, transcript, preview }: BubbleProps) {
  const att = attachments(message);
  const text = userText(message);
  return (
    <div className="msg msg-user">
      <div className={`bubble bubble-user ${att.photo && preview ? "has-photo" : ""}`}>
        {att.photo &&
          (preview ? (
            <img className="bubble-photo" src={preview} alt="Фото, отправленное коучу" />
          ) : (
            <span className="att-tag">
              <ImageIcon size={14} /> Фото
            </span>
          ))}
        {att.audio && (
          <span className="att-tag">
            <WaveIcon size={14} /> Голосовое
          </span>
        )}
        {text && <div className="bubble-text">{text}</div>}
        {transcript && (
          <div className="bubble-transcript">
            <span className="bubble-transcript-label">Распознано:</span> {transcript}
          </div>
        )}
      </div>
      <time className="msg-time">{timeShort(message.created_at)}</time>
    </div>
  );
}

function TrainerBubble({ message }: { message: ChatMessage }) {
  const text = message.text.replace(/^Тренер ответил:\s*/, "");
  return (
    <div className="msg msg-assistant">
      <div className="bubble bubble-trainer">
        <div className="bubble-head">
          <span className="bubble-head-icon is-trainer">
            <UserTieIcon size={14} />
          </span>
          Ответ тренера
        </div>
        <RichText text={text} />
      </div>
      <time className="msg-time">{timeShort(message.created_at)}</time>
    </div>
  );
}

function CoachBubble({ message, hideTranscript, mealLogged, onMealLogged, onOpenPlan }: BubbleProps) {
  const p = asReply(message.payload);
  const kind = p?.kind ?? "answer";
  const citations = p?.citations ?? [];
  const cls =
    kind === "escalated" ? "bubble-escalated" : kind === "refusal" ? "bubble-muted" : "bubble-coach";

  return (
    <div className="msg msg-assistant">
      <div className={`bubble ${cls}`}>
        {kind === "escalated" && (
          <div className="bubble-head is-danger">
            <span className="bubble-head-icon is-danger">
              <AlertIcon size={14} />
            </span>
            Передано тренеру
          </div>
        )}
        {p?.transcript && !hideTranscript && (
          <div className="bubble-transcript is-coach">
            <span className="bubble-transcript-label">Распознано:</span> {p.transcript}
          </div>
        )}
        <RichText text={message.text} />
        {citations.length > 0 && <Citations items={citations} />}
        {kind === "proposal" && (
          <div className="bubble-note">
            <div className="bubble-note-text">
              Запрос ушёл тренеру. Программа изменится только после его решения.
            </div>
            <button className="btn btn-sm btn-outline bubble-note-btn" onClick={onOpenPlan}>
              Статус запроса <ArrowRightIcon size={14} />
            </button>
          </div>
        )}
      </div>
      {p?.meal && p.meal.card_id && (
        <div className="msg-card">
          <MealCard
            card={p.meal}
            initiallyLogged={mealLogged}
            onLogged={(r) => onMealLogged?.(p.meal!.card_id, r)}
          />
        </div>
      )}
      <div className="msg-foot">
        <time className="msg-time">{timeShort(message.created_at)}</time>
        {canRate(message) && (
          <AnswerFeedback messageId={message.id as number} initial={message.feedback ?? null} />
        )}
      </div>
    </div>
  );
}

function Citations({ items }: { items: Citation[] }) {
  return (
    <div className="cites" aria-label="Источники">
      {items.map((c) => {
        const label = `[${c.n}] ${c.title}${c.page != null ? `, стр. ${c.page}` : ""}`;
        return (
          <span key={`${c.n}-${c.title}`} className="cite" title={SOURCE_LABEL[c.source] ?? c.source}>
            {label}
          </span>
        );
      })}
    </div>
  );
}
