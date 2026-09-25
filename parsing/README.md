# Обработка документов Tulpar

PDF/DOCX → Parser → Cleaning → Chunking → Metadata → существующий Tulpar RAG
→ Embeddings → embedded Qdrant → Retrieval → reranking → LangGraph → LLM → Citations.

Это выделенная из текущей реализации обработка документов. Здесь нет отдельного
RAG, embeddings, vector store или backend.

- `parser.py`: PyMuPDF читает PDF постранично, страницы начинаются с 1.
  python-docx читает абзацы и таблицы DOCX в порядке документа; `page=None`,
  поскольку DOCX не имеет надёжной физической пагинации. Parser возвращает
  `text`, `source` (имя файла), `page`, `file_type`. OCR не выполняется.
- `cleaner.py`: удаляет NUL, приводит CRLF/CR к LF, нормализует настоящие tabs
  и пробелы, сокращает лишние пустые строки, сохраняет абзацы.
  Буквы `t`/`n` и буквальные последовательности `\\t`/`\\n` сохраняются.
- `chunker.py`: существующий `split_text` перенесён без изменения алгоритма.
  Приоритет: абзац → строка → предложение → слово, overlap выравнивается по словам.
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

Зависимости уже есть в корневом `requirements.txt`: PyMuPDF и python-docx.
Для тестов нужен pytest из `requirements-dev.txt`. Из корня проекта в PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest parsing/tests -q
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты создают небольшие локальные PDF/DOCX во временной папке и не обращаются к сети.
