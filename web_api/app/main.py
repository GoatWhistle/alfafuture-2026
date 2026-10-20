"""FastAPI app: loopback only, Origin/Host checked, one active ML run."""

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import FILE_NAMES, MAX_DESCRIPTION_CHARS, MAX_UPLOAD_BYTES, Settings
from .service import Service, qwen_status
from .store import NotFound, RunStore, iso_now
from .supervisor import AbortFailed, Conflict, Supervisor
from .task_text import TaskTextError, build_task_md

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    size = 0
    head = b""
    with destination.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            if not head:
                head = chunk[:65536]
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"Файл {destination.name} слишком большой")
            out.write(chunk)
    if size == 0:
        raise HTTPException(422, f"Файл {destination.name} пуст")
    try:
        head.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        if error.start < len(head) - 4:  # a cut multibyte sequence at the chunk edge is fine
            raise HTTPException(422, f"Файл {destination.name} не в кодировке UTF-8") from None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = RunStore(settings.runs_root)
    supervisor = Supervisor(settings, store)
    service = Service(settings, store, supervisor)
    app = FastAPI(title="AutoResearch API", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.headers.get("host", "") not in settings.allowed_hosts:
            return JSONResponse({"detail": "Недопустимый Host"}, status_code=403)
        if (
            request.method in _MUTATING
            and request.headers.get("origin") not in settings.allowed_origins
        ):
            return JSONResponse({"detail": "Недопустимый Origin"}, status_code=403)
        return await call_next(request)

    def run_or_404(run_id: str) -> dict:  # type: ignore[type-arg]
        try:
            return service.run(run_id)
        except NotFound:
            raise HTTPException(404, "Исследование не найдено") from None

    @app.get("/api/qwen")
    def qwen() -> dict:  # type: ignore[type-arg]
        return qwen_status()

    @app.get("/api/runs")
    def list_runs() -> list:  # type: ignore[type-arg]
        return service.summaries()

    @app.post("/api/runs", status_code=201)
    async def create_run(
        title: Annotated[str, Form()],
        description: Annotated[str, Form()],
        train: Annotated[UploadFile, File()],
        test: Annotated[UploadFile, File()],
        dictionary: Annotated[UploadFile, File()],
        sample: Annotated[UploadFile, File()],
        time_sec: Annotated[int, Form(ge=60, le=10800)] = 1800,
        tokens: Annotated[int, Form(ge=1000, le=500000)] = 32000,
        threads: Annotated[int, Form(ge=1, le=8)] = 4,
        seed: Annotated[int, Form(ge=0, lt=2**32)] = 42,
        memory_gb: Annotated[int, Form(ge=1, le=27)] = 8,
        worker_timeout_sec: Annotated[int, Form(ge=60, le=7200)] = 600,
        mode: Annotated[str, Form(pattern="^(qwen|optuna|baseline)$")] = "qwen",
    ) -> dict:  # type: ignore[type-arg]
        if not title.strip() or len(title) > 200:
            raise HTTPException(422, "Название обязательно (до 200 символов)")
        if len(description) > MAX_DESCRIPTION_CHARS:
            raise HTTPException(422, "Описание слишком длинное")
        try:
            task_md = build_task_md(title, description)
        except TaskTextError as error:
            raise HTTPException(422, str(error)) from None
        run_id = store.new_id()
        run_dir = store.create(
            run_id,
            {
                "title": title.strip(),
                "created_at": iso_now(),
                # Модель фиксируется при создании: позже конфигурация сервера может смениться
                "model": qwen_status()["model"] if mode == "qwen" else "Без LLM",
                "options": {
                    "time_sec": time_sec,
                    "tokens": tokens,
                    "threads": threads,
                    "seed": seed,
                    "memory_gb": memory_gb,
                    "worker_timeout_sec": worker_timeout_sec,
                    "mode": mode,
                },
            },
        )
        uploads = {"train": train, "test": test, "dictionary": dictionary, "sample": sample}
        for key, upload in uploads.items():
            await _save_upload(upload, run_dir / "inputs" / FILE_NAMES[key])
        (run_dir / "inputs" / "task.md").write_text(task_md, encoding="utf-8")
        return {"id": run_id}

    @app.post("/api/runs/{run_id}/start")
    def start(run_id: str) -> dict:  # type: ignore[type-arg]
        run_or_404(run_id)
        try:
            supervisor.start(run_id)
        except Conflict as error:
            raise HTTPException(409, str(error)) from None
        return {"id": run_id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:  # type: ignore[type-arg]
        return run_or_404(run_id)

    @app.get("/api/runs/{run_id}/experiments/{experiment_id}")
    def get_experiment(run_id: str, experiment_id: str) -> dict:  # type: ignore[type-arg]
        for experiment in run_or_404(run_id)["experiments"]:
            if experiment["id"] == experiment_id:
                return experiment
        raise HTTPException(404, "Эксперимент не найден")

    @app.post("/api/runs/{run_id}/abort")
    def abort(run_id: str) -> dict:  # type: ignore[type-arg]
        run_or_404(run_id)
        try:
            supervisor.abort(run_id)
        except Conflict as error:
            raise HTTPException(409, str(error)) from None
        except AbortFailed as error:
            raise HTTPException(504, str(error)) from None
        return service.run(run_id)

    @app.get("/api/runs/{run_id}/download/{kind}")
    def download(run_id: str, kind: str):  # type: ignore[no-untyped-def]
        run = run_or_404(run_id)
        if kind == "report":
            return PlainTextResponse(
                service.report_text(run_id),
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{run_id}-report.md"'},
            )
        if kind not in ("current", "final"):
            raise HTTPException(404, "Неизвестный артефакт")
        if kind == "final" and not run["finalCsv"]:
            raise HTTPException(409, "Итогового CSV пока нет")
        path = service.csv_path(run_id)
        if path is None:
            raise HTTPException(404, "CSV ещё не создан")
        return FileResponse(path, media_type="text/csv", filename=f"{run_id}-{kind}.csv")

    dist = settings.repo_root / "web" / "dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def ui(path: str):  # type: ignore[no-untyped-def]
            if path.startswith("api/"):
                raise HTTPException(404, "Не найдено")
            return FileResponse(dist / "index.html")

    return app


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host="127.0.0.1", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
