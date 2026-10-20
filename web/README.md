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
