<!-- Источник: Tulpar docs/nutrition.md, выгружено tools/export_from_tulpar.py -->
# Питание — конфигуратор (архитектура)

Дневник питания: калькулятор нормы, лог приёмов из базы продуктов, фото-распознавание, ИИ-планы/рецепты, штрихкод, вода. Код (актуально 2026-08-18) — **пакет** `backend/app/routers/nutrition/` (`ai/common/diary/foods/photo/plans/recipes/structure/targets/templates/water`) + фронт-пакет `frontend/src/pages/nutrition/` и общий редактор `frontend/src/components/NutritionPlanStructure.tsx` (+ `frontend/src/components/BarcodeScanner.tsx`). Помимо дневника/нормы есть: `GET/POST /me/water`, `GET /me/diary-summary`, структурированные рационы `GET /me/nutrition`, `POST /nutrition`, `PUT /nutrition/{id}/structure` и перенос назначенного дня `POST /me/nutrition/{id}/transfer`.

Структура рациона — `день → приёмы → продукты`. `day_index = null` означает
«любой день», а не пропуск. Позиция с `food_id` переносится в дневник; свободная
позиция без каталожного продукта остаётся видимой в рационе, но не получает
выдуманные КБЖУ и возвращается в `items_without_food`. Перенос идемпотентен на
пару `план + дата`: повтор не восстанавливает удалённые или изменённые учеником
строки. Создание плана и запись структуры пока остаются двумя запросами; если
второй падает после подтверждённого `POST`, клиент повторяет `PUT` в тот же
`plan_id`, а не создаёт второй пустой план.

## Эндпоинты

| Группа | Эндпоинты | Что делает |
|---|---|---|
| **Норма (A4)** | `GET /me/nutrition-target?activity&goal`, `PUT /me/nutrition-goal` | Mifflin-St Jeor BMR × активность → калории+БЖУ. При «точной цели» (вес+дата) калории считаются от безопасного темпа; обратная связь по тренду веса. |
| **Продукты** | `GET/POST /foods` | Пресет-каталог (`is_preset`) + личные (`created_by_user_id`), поиск по имени. |
| **Дневник** | `GET/POST/PATCH/DELETE /me/diary` | Лог приёмов на дату. **Снимок БЖУ на 100 г** хранится в записи (см. ниже). `grams>0`, макросы `≥0` валидируются. |
| **Фото (B)** | `POST /nutrition/photo-estimate` | Фото блюда → Ollama vision → редактируемый черновик. |
| **ИИ-нутрициолог (C3)** | `POST /nutrition/ai/day-plan`, `POST /nutrition/ai/tip` | План дня под норму; совет от съеденного vs норма. |
| **Шаблоны (C1)** | `GET/POST/DELETE /nutrition/templates` | Пресеты + сохранённый «свой день»; применяются в дневник. |
| **Рецепты (C4)** | `GET/POST/DELETE /nutrition/recipes`, `POST /nutrition/recipes/ai` | Каталог + ИИ-генерация (черновик, `id=""`); добавляется в дневник одним блюдом. |
| **Штрихкод (C2)** | `GET /nutrition/barcode/{code}` | Наш каталог → OpenFoodFacts → кэш как пресет (флайвил). **Ключ Ollama не нужен.** |

## Ключевые инварианты

- **Снимок на 100 г.** `DiaryEntry` и items шаблонов/рецептов/фото хранят `kcal/protein/fat/carbs` **на 100 г** + `grams` отдельно. Поэтому правка/удаление исходного `Food` не переписывает уже залогированный день. Модель vision/ИИ возвращает БЖУ **всей порции** → сервер делит на per-100 г: `k = 100/grams` (`_norm_item`). Рецепт логируется одним блюдом: combined per-100 г = `total*100/totalGrams`.
- **Флайвил OFF.** `barcode_lookup` кэширует находку OpenFoodFacts как `is_preset=True` Food → следующий скан (у любого) мгновенный, и продукт ищется по имени.
- **Деградация без ключа.** Реальный гейт ИИ — наличие **`OLLAMA_API_KEY`** (НЕ `AI_ENABLED`): фото/план/рецепт → `503`, совет → `tip:null`. Штрихкод (OpenFoodFacts) работает без ключа.
- **Безопасный темп нормы.** `KCAL_PER_KG=7700`, потеря ≤0.75%/нед, набор ≤`SAFE_GAIN_KG_WK=0.35` кг/нед, абсолютный пол 1500 (м) / 1200 (ж). `bmr*1.1` НЕ используется как пол (маскировал активность — фикс 2026-06-26).

## Тесты
Бэк: `tests/test_nutrition_*.py` (diary, goal, photo, ai, templates, recipes, barcode, qa). Парсинг Ollama/OFF тестируется через monkeypatch `_ollama_chat` / `httpx.AsyncClient` (без сети); живые вызовы — ручное QA (`docs/qa/on-device-native.md`). Фронт: `pages/Nutrition.test.tsx`, `pages/StudentDetail.test.tsx`, `components/BarcodeScanner.test.tsx` и реальный сценарий `e2e/nutrition-plan-transfer.spec.ts`.