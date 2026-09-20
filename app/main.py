"""FastAPI server: JSON API + the static front end."""
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import rag  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("resume-analyzer")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Mini Resume Analyzer")
store = rag.SessionStore()


class Turn(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=1000)
    job_role: str = Field(default="", max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=12)


def _ai_error(exc: Exception) -> HTTPException:
    """Turn a raw AI-provider error into a message a person can act on."""
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return HTTPException(429, "The AI service is rate-limited right now. Wait a minute and try again.")
    if "api key" in low or "API_KEY_INVALID" in msg or "401" in msg or "403" in msg:
        return HTTPException(502, "The AI service rejected the API key. Check GEMINI_API_KEY on the server.")
    if "404" in msg and "model" in low:
        return HTTPException(502, "The configured AI model was not found. Update GEN_MODEL / EMBED_MODEL.")
    return HTTPException(502, "The AI service returned an error. Try again in a moment.")


@app.get("/api/health")
def health():
    return {"ok": True, "api_key_configured": bool(rag.api_key())}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    name = file.filename or "resume.pdf"
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is larger than 5 MB. Upload a smaller PDF.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a PDF file.")

    try:
        ri = rag.build_index(name, data)
    except rag.UserError as exc:
        raise HTTPException(400, str(exc))
    except rag.AIServiceError as exc:
        raise HTTPException(500, str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Indexing failed")
        raise _ai_error(exc)

    session_id = uuid.uuid4().hex
    store.put(session_id, ri)
    return {
        "session_id": session_id,
        "filename": ri.filename,
        "pages": ri.pages,
        "characters": ri.characters,
        "embedding_dim": ri.dim,
        "chunks": ri.chunks,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    ri = store.get(req.session_id)
    if ri is None:
        raise HTTPException(404, "This session has expired. Upload the resume again.")
    try:
        result = rag.answer(
            ri,
            req.question.strip(),
            req.job_role.strip(),
            [t.model_dump() for t in req.history],
        )
    except rag.UserError as exc:
        raise HTTPException(400, str(exc))
    except rag.AIServiceError as exc:
        raise HTTPException(500, str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Answering failed")
        raise _ai_error(exc)
    return result


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    store.delete(session_id)
    return {"ok": True}


# Must come last so it doesn't shadow the /api routes.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
