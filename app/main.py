import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from app.engine import Anonymizer, NatashaNER
from app.extract import DocumentError, extract_text
from app.ollama import OllamaNER

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
# PDF parser warnings may include excerpts from an invalid source document.
logging.getLogger("pypdf").setLevel(logging.CRITICAL)


@asynccontextmanager
async def lifespan(app):
    # Model loads at startup, not on every document. A broken model prevents startup.
    ner = NatashaNER() if os.getenv("NER_ENABLED", "1") == "1" else None
    llm = OllamaNER() if os.getenv("OLLAMA_ENABLED", "1") == "1" else None
    app.state.engine = Anonymizer(Path(os.getenv("RULES_PATH", ROOT / "config/rules.json")), ner, llm)
    app.state.busy = asyncio.Lock()
    yield


app = FastAPI(title="Анонимизатор", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/healthz")
async def health():
    return {"status": "ok", "ner": app.state.engine.ner is not None, "ollama": app.state.engine.llm is not None}


async def read_body(request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BYTES + 1024 * 1024:
            raise DocumentError("Превышен размер загрузки.")
    return bytes(data)


@app.post("/api/anonymize")
@app.post("/webhook/anonymize-upload")
async def anonymize(request: Request):
    # One CPU task at a time; no unbounded queue of sensitive documents in memory.
    if app.state.busy.locked():
        return JSONResponse({"error": "Сервис занят. Повторите загрузку через несколько секунд."}, status_code=503, headers={"Retry-After": "5"})
    async with app.state.busy:
        try:
            body = await read_body(request)
            content_type = request.headers.get("content-type", "")
            if content_type.startswith("multipart/form-data"):
                # Cache the bounded body for Starlette's parser. Uploaded files can spool
                # to tmpfs; use a private RuntimeDirectory on the server.
                request._body = body
                async with request.form(max_files=1, max_fields=5, max_part_size=MAX_BYTES) as form:
                    files = [value for _, value in form.multi_items() if isinstance(value, UploadFile)]
                    if len(files) != 1:
                        raise DocumentError("Загрузите один файл.")
                    filename = files[0].filename or "document.txt"
                    data = await files[0].read(MAX_BYTES + 1)
            elif content_type.startswith("application/json"):
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                    raise DocumentError("Ожидается JSON с полем text.")
                filename, data = "document.txt", payload["text"].encode("utf-8")
            else:
                filename = request.headers.get("x-filename", "document.txt")
                data = body
            if len(data) > MAX_BYTES:
                raise DocumentError("Файл превышает допустимый размер.")
            text = await run_in_threadpool(extract_text, data, filename)
            result = await run_in_threadpool(app.state.engine.anonymize, text)
            return Response(result, media_type="text/plain; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="anonymized.txt"'})
        except (DocumentError, ValueError) as exc:
            message = str(exc) if isinstance(exc, DocumentError) else "Неверный формат запроса."
            return JSONResponse({"error": message}, status_code=400)
        except HTTPException:
            return JSONResponse({"error": "Неверная форма загрузки. Отправьте один файл."}, status_code=400)
        except Exception:
            # Don't include parser exceptions: they may contain original text.
            return JSONResponse({"error": "Не удалось обработать документ."}, status_code=500)


# Put the existing website here without rewriting its appearance.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="website")
