# Tulpar AI Coach — веб-интерфейс

Vite + React + TypeScript, обычный CSS без фреймворков. Два режима по роли:
клиент (колонка до 480 px, как Telegram Mini App) и тренер (до 1000 px).

## Запуск

```bash
# 1. API (из корня репозитория, порт 8089, демо-режим по умолчанию)
.venv/bin/python -m tulpar_ai

# 2. Фронтенд в режиме разработки: http://localhost:5173
cd web
npm install
npm run dev        # /api и /health проксируются на http://localhost:8089
```

Сборка для прода:

```bash
cd web
npm run build      # tsc --noEmit + vite build → web/dist
```

FastAPI отдаёт `web/dist` на `/` (SPA-фоллбэк) и `web/dist/assets` на `/assets`,
поэтому после `npm run build` интерфейс открывается прямо на http://localhost:8089.
`npm run typecheck` — только проверка типов.

## Экраны

| Роль | Вкладка | Файл | Что делает |
|---|---|---|---|
| — | Вход | `src/pages/Login.tsx` | Демо-вход: «Войти как клиент» / «Войти как тренер» |
| Клиент | Коуч | `src/pages/client/ChatPage.tsx` | Чат: текст, фото еды, голосовое; ответы со ссылками на источники |
| Клиент | — | `src/pages/client/MealCard.tsx` | Карточка КБЖУ: правка граммов, выбор приёма пищи, «Записать» |
| Клиент | Программа | `src/pages/client/PlanPage.tsx` | Активная программа и статусы предложений |
| Тренер | Очередь | `src/pages/trainer/QueuePage.tsx` | Предложения правок (drafting/pending) и эскалации (open) |
| Тренер | — | `src/pages/trainer/ProposalCard.tsx` | Правка программы: изменения, нарушения, до/после; принять / отклонить / доработать |
| Тренер | — | `src/pages/trainer/EscalationCard.tsx` | Вопрос клиента для тренера: ответить в чат или закрыть |
| Тренер | Клиенты | `src/pages/trainer/ClientsPage.tsx` | Список клиентов, переход в карточку |
| Тренер | — | `src/pages/trainer/ClientDetail.tsx` | Профиль, заметка о травме, питание за 14 дней, программа, чат, запрос правки |
| Тренер | История | `src/pages/trainer/HistoryPage.tsx` | Закрытые решения: applied / rejected / failed / resolved |

Вкладка хранится в адресе: `#/coach`, `#/plan`, `#/queue`, `#/clients/<id>`, `#/history`.

## Устройство

- `src/api.ts` — типизированные запросы; токен в `localStorage["tac_token"]`,
  на 401 токен стирается и отправляется событие `tac:logout`. `chat` и `decide` ждут до 120 с.
  `chatForm({ text, photo, audio })` собирает FormData, `pollProposal(id, cb)` опрашивает drafting.
- `src/types.ts` — типы всех объектов контракта API.
- `src/format.ts` — русские подписи (статусы, цели, уровни, оборудование, дни недели),
  форматирование чисел и дат, `portion(item, grams)` — КБЖУ порции (значения в каталоге на 100 г).
- `src/Shells.tsx` — шапка, вкладки, роутинг по hash; `src/App.tsx` — проверка токена и `/api/me`.
- `src/components/` — `Spinner`, `StatusChip`, `PlanView` (с режимом сравнения `compareTo`), `Empty`.
- `src/styles.css` — токены и общие классы (`.btn`, `.card`, `.chip`, `.notice`, `.status-*`).
  Стили экранов — в `pages/client/*.css` и `pages/trainer/*.css`; в них только токены:
  радиусы `--r-*`, отступы `--s-*`, размеры `--fs-*` (включая `--fs-2xs`), статусные
  рамки и текст `--{amber,green,red}-line` / `--{amber,green,red}-text`. Новые цвета
  заводим токеном в `:root`, а не хексом в экранном CSS.

Сигнатуры пропсов каждого экрана описаны комментарием в начале файла; App и Shells
передают ровно их.

## Особенности контракта

- `id` сообщений чата — целое число (у локальных, ещё не сохранённых — строка `local-…`);
  для ключей словарей используйте `msgKey(m)` из `pages/client/chatModel.ts`.
- `payload` сообщения клиента — `{has_photo, has_audio}`; у служебной заметки `meal_logged`
  нет `card_id`, поэтому записанная карточка определяется по порядку сообщений.
- КБЖУ в `MealItem` — на 100 г; ккал порции округляются «к чётному» (`fmtKcal`), как `round()`
  в Python, чтобы число в карточке совпадало с текстом ответа.
- У эскалаций нет поля `changes`; у `nutrition_14d` без записей есть только `days_logged: 0`.
- Без ключей LLM (демо) черновики программы сразу уходят в `failed` — карточка показывает
  причину из `reply`, клиенту такие черновики тренера не показываются.

## Быстрая проверка

```bash
npm run build && cd .. && .venv/bin/python -m tulpar_ai &    # :8089
curl -s localhost:8089/health                                 # {"ok":true,"mode":"demo"}
TOKEN=$(curl -s -XPOST localhost:8089/api/auth/demo-login -H 'content-type: application/json' \
  -d '{"role":"client"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -s -XPOST localhost:8089/api/chat -H "Authorization: Bearer $TOKEN" -F 'text=съел 200 г плов'
```

