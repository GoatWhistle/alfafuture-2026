# Веб-панель AutoResearch

Локальная панель для запуска и наблюдения за ML-агентом. Не входит в конкурсный ZIP:
`scripts/build_submission.py` упаковывает только `solution.py` и `ml_harness/`.

```bash
make web-setup      # npm ci + зависимости web_api
make web-demo       # только интерфейс, демонстрационные данные (http://127.0.0.1:5173)
make web-api        # API на 127.0.0.1:8765, запускает solution.py
make web-ui         # интерфейс с подключением к API (запускать вместе с web-api)
make web-check      # сборка, тесты интерфейса и API
```

Данные исследований API хранит в `runs/web/<id>/` (игнорируется Git). Ключи Qwen берутся из
`.env` или окружения сервера и в браузер не передаются. Для работы без Vite: `npm run build`
в `web/`, затем `make web-api` отдаёт собранный интерфейс на `http://127.0.0.1:8765`
(сборка должна быть с `VITE_DATA_SOURCE=api`).

## Проверки в браузере (e2e)

Скрипты в `web/e2e/` управляют Chrome через DevTools Protocol без сторонних пакетов
(путь к Chrome можно задать через `CHROME`). `<work>` — любой временный каталог.

```bash
node web/e2e/make-data.mjs <work>               # синтетическая задача в <work>/syn
make web-demo & node web/e2e/demo.mjs <work>     # демо: клавиатура, диалоги, состояния, наложения

# подключённая версия с fake LLM (без обращений к Qwen):
python web_api/tests/fake_llm.py 8799 <work>/llm.log &
(cd web_api && OPENAI_BASE_URL=http://127.0.0.1:8799/v1 OPENAI_API_KEY=fake \
  OPENAI_MODEL=fake-qwen WEB_API_RUNS=<work>/runs-web uv run python -m app.main) &
make web-ui & node web/e2e/api.mjs <work>
```
