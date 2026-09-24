// Пустая очередь: объясняем, откуда здесь появляются карточки.
import { navigate } from "../../route";

export default function QueueEmpty() {
  return (
    <div className="t-empty">
      <div className="t-empty-mark" aria-hidden="true">
        <span />
      </div>
      <div className="t-empty-title">Очередь пуста</div>
      <p className="t-empty-text">
        Здесь появятся черновики правок программы и вопросы, на которые AI не ответил сам. Ничего не
        применяется к клиенту без вашего решения.
      </p>
      <div className="t-empty-steps">
        <div className="t-step">
          <span className="t-step-n">1</span>
          <span>Клиент пишет коучу «замените присед, болит колено» — или вы просите AI в карточке клиента</span>
        </div>
        <div className="t-step">
          <span className="t-step-n">2</span>
          <span>AI готовит черновик по Skill и проверяет его валидатором</span>
        </div>
        <div className="t-step">
          <span className="t-step-n">3</span>
          <span>Вы принимаете, поправляете комментарием или отклоняете</span>
        </div>
      </div>
      <button className="btn btn-outline btn-sm" onClick={() => navigate("clients")}>
        Открыть клиентов
      </button>
    </div>
  );
}
