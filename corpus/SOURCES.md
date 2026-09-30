# Источники корпуса

| Файл | Что это | Источник и лицензия |
|---|---|---|
| `exercises.jsonl` | 196 упражнений: техника, польза, противопоказания на русском | Сиды Tulpar (`backend/scripts/seed.py`, `backend/data/b18_ru_*.json`), выгружено `tools/export_from_tulpar.py`. Собственность проекта Tulpar |
| `nutrition.md` | Правила питания Tulpar: снимок КБЖУ на 100 г, безопасный темп, нижние пределы калорий | `docs/nutrition.md` Tulpar |
| `who_2020_physical_activity.pdf` | WHO guidelines on physical activity and sedentary behaviour, 2020, ISBN 978-92-4-001512-8, 104 стр. | © World Health Organization 2020. Лицензия CC BY-NC-SA 3.0 IGO. Используется некоммерчески, с указанием авторства |
| `evidence/schoenfeld_2021_repetition_continuum.docx` | Нагрузка и число повторений для силы, роста мышц и выносливости: пересмотр «континуума повторений» | Schoenfeld BJ, Grgic J, Van Every DW, Plotkin DL. Sports. 2021;9(2):32. doi:10.3390/sports9020032. CC BY 4.0 |
| `evidence/singer_2024_rest_intervals.docx` | Отдых между подходами и рост мышц: систематический обзор с байесовским метаанализом | Singer A, Wolf M, Generoso L, et al. Front Sports Act Living. 2024;6:1429789. doi:10.3389/fspor.2024.1429789. CC BY 4.0 |
| `evidence/melby_2017_weight_regain.docx` | Почему вес возвращается после похудения: энергетический разрыв, голод, расход энергии | Melby CL, Paris HL, Foright RM, Peth J. Nutrients. 2017;9(5):468. doi:10.3390/nu9050468. CC BY 4.0 |
| `evidence/lopez_minguez_2019_meal_timing.docx` | Время завтрака, обеда и ужина: влияние на ожирение и обмен веществ | Lopez-Minguez J, Gómez-Abellán P, Garaulet M. Nutrients. 2019;11(11):2624. doi:10.3390/nu11112624. CC BY 4.0 |
| `evidence/taylor_2021_sitting_breaks.docx` | Перерывы в сидении: сосуды, глюкоза и инсулин после еды, рандомизированное перекрёстное исследование | Taylor FC, Dunstan DW, Fletcher E, et al. PLoS One. 2021;16(1):e0244841. doi:10.1371/journal.pone.0244841. CC BY 4.0 |
| `evidence/xu_2024_daily_steps_umbrella.docx` | Шаги в день и здоровье: зонтичный обзор метаанализов | Xu et al. BMJ Open. 2024. doi:10.1136/bmjopen-2024-088524. CC BY-NC 4.0 — некоммерческое использование, как и PDF ВОЗ |

**Открытые статьи в `evidence/`** добавлены 30.09.2026, чтобы закрыть вопросы, которые симулированные клиенты задавали, а база знаний не могла ответить: подходы и повторения, отдых между подходами, плато веса, поздний ужин, шаги в день, перерывы в сидячей работе (`docs/user-simulation.md`, находка 4). Текст взят из Europe PMC (`tools/fetch_open_articles.py`) без изменений, кроме удалённых разделов о методах и результатах, таблиц, рисунков, ссылок на литературу и списка литературы; первая строка каждого документа это указывает, как требует CC BY. Статьи на английском, как и PDF ВОЗ: русские вопросы находят их через многоязычные эмбеддинги.
