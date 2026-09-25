# Обработка документов Tulpar

PDF/DOCX → Parser → Cleaning → Chunking → Metadata → существующий Tulpar RAG
→ Embeddings → embedded Qdrant → Retrieval → reranking → LangGraph → LLM → Citations.

Это выделенная из текущей реализации обработка документов. Здесь нет отдельного
RAG, embeddings, vector store или backend.

- `parser.py`: pypdfium2 (PDFium, лицензии Apache-2.0 / BSD-3-Clause) читает PDF
  постранично, страницы начинаются с 1; берётся только текст в границах страницы.
  PyMuPDF заменён, потому что он под AGPL-3.0, а пакет переиспользуется в
  коммерческом Tulpar SaaS. Текст бэкендов сравнивает `evals/parsing_backends.py`
  (без сети): на PDF ВОЗ последовательности слов pypdfium2 и PyMuPDF совпадают
  на 99,8%, а pypdf добавляет 8,5% слов, почти все — с соседней страницы разворота.
  Поиск эту разницу не различает. python-docx читает абзацы и таблицы DOCX в порядке
  документа (объединённые ячейки — один раз, вложенные таблицы — тоже); `page=None`,
  поскольку DOCX не имеет надёжной физической пагинации. Parser возвращает
  `text`, `source` (имя файла), `page`, `file_type`. OCR не выполняется: для PDF
  без текстового слоя пишется предупреждение в лог, фрагментов нет.
- `cleaner.py`: удаляет NUL и прочие управляющие символы, мягкие переносы и
  символы нулевой ширины, раскрывает лигатуры (`ﬁ` → `fi`), приводит CRLF/CR к LF,
  неразрывные и прочие пробелы — к обычному, сокращает лишние пустые строки,
  сохраняет абзацы. Буквы `t`/`n` и буквальные последовательности `\\t`/`\\n` сохраняются.
- `chunker.py`: приоритет разреза: абзац → строка → предложение → слово, overlap
  выравнивается по словам. Каждый следующий фрагмент заканчивается дальше
  предыдущего — перекрытие не становится отдельным фрагментом-дублем.
  Фрагменты PDF короче 81 символа (колонтитул, номер страницы, ISBN) отбрасываются,
  как и в индексаторе до пакета; их `chunk_index` пропускается, ID остальных не сдвигаются.
  `PARSING_VERSION` входит в имя коллекции Qdrant: при изменении разбора поднимите
  его, и сохранённый индекс пересоберётся.
  `chunk_document` по умолчанию использует размер 400 и overlap 120, как Index.
  Настройки передаются из существующего загрузчика; его отдельный default 800
  также сохранён. Карточки упражнений обрабатываются прежним кодом Index.

`chunk_document` возвращает словари для существующего `Chunk`, а не новую сущность.
Metadata: `id`, `source`, `title`, `text`, `page`, `file_type`, `chunk_index`.
`source` становится относительным путём внутри corpus, чтобы различать одинаковые
имена в подпапках. `chunk_index` начинается с 0 на каждой странице PDF / в документе
DOCX. ID детерминированы; старые ID и alias `who2020` сохранены. Citation использует
существующий `Chunk.id` как `chunk_id`.

## Подключение

`tulpar_ai/rag/index.py` вызывает `chunk_document()` и создаёт `Chunk(**payload)`.
Index, embedder, upsert, Qdrant, retrieval и citations остаются существующими.
Нечитаемый документ пропускается с предупреждением, остальной индекс собирается.
При следующем `Index.build()` индекс ищет документы корпуса без точек в коллекции
(по полю `source`) и добавляет только их; уже проиндексированное не эмбеддится заново.
Старые импорты `tulpar_ai.rag.parsing.normalize_text`, `parse_document` и
`tulpar_ai.rag.index.split_text` продолжают работать. Dockerfile копирует пакет.

```python
from pathlib import Path
from parsing.chunker import chunk_document

payloads = chunk_document(
    Path("corpus/who_2020_physical_activity.pdf"),
    corpus_root=Path("corpus"),
    size=400,
    overlap=120,
)
```

Зависимости уже есть в корневом `requirements.txt`: pypdfium2 и python-docx.
Для тестов нужен pytest из `requirements-dev.txt`. Из корня проекта в PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest parsing/tests -q
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты создают небольшие локальные PDF (`tests/pdfgen.py`, без PDF-библиотек) и DOCX
во временной папке и не обращаются к сети.
