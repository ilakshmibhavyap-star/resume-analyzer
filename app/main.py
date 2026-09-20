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


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

STATIC_DIR = (
    Path(__file__).resolve().parent.parent / "static"
)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mini Resume Analyzer",
)


store = rag.SessionStore()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str
    content: str = Field(
        max_length=4000
    )


class AskRequest(BaseModel):
    session_id: str

    question: str = Field(
        min_length=1,
        max_length=1000,
    )

    job_role: str = Field(
        default="",
        max_length=2000,
    )

    history: list[Turn] = Field(
        default_factory=list,
        max_length=12,
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _ai_error(exc: Exception) -> HTTPException:
    """
    Convert raw Gemini errors into messages suitable for the frontend.
    """

    message = str(exc)
    lower = message.lower()

    if (
        "429" in message
        or "resource_exhausted" in lower
    ):
        return HTTPException(
            status_code=429,
            detail=(
                "The AI service is rate-limited right now. "
                "Wait a moment and try again."
            ),
        )

    if any(
        value in lower
        for value in (
            "api key",
            "api_key_invalid",
            "401",
            "403",
            "permission denied",
        )
    ):
        return HTTPException(
            status_code=502,
            detail=(
                "The AI service rejected the API key. "
                "Check GEMINI_API_KEY on the server."
            ),
        )

    if (
        "404" in message
        and "model" in lower
    ):
        return HTTPException(
            status_code=502,
            detail=(
                "The configured Gemini model was not found. "
                "Check GEN_MODEL and EMBED_MODEL."
            ),
        )

    return HTTPException(
        status_code=502,
        detail=(
            "The AI service returned an error. "
            "Try again in a moment."
        ),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "api_key_configured": bool(
            rag.api_key()
        ),
        "generation_model": rag.GEN_MODEL,
        "embedding_model": rag.EMBED_MODEL,
    }


# ---------------------------------------------------------------------------
# Upload resume
# ---------------------------------------------------------------------------

@app.post("/api/upload")
def upload(
    file: UploadFile = File(...),
):
    name = file.filename or "resume.pdf"

    if not name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    data = file.file.read(
        MAX_UPLOAD_BYTES + 1
    )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "That file is larger than 5 MB. "
                "Upload a smaller PDF."
            ),
        )

    if not data.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid PDF file.",
        )

    try:
        resume_index = rag.build_index(
            name,
            data,
        )

    except rag.UserError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except rag.AIServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except Exception as exc:  # noqa: BLE001
        log.exception(
            "Resume indexing failed"
        )

        raise _ai_error(exc)

    session_id = uuid.uuid4().hex

    store.put(
        session_id,
        resume_index,
    )

    return {
        "session_id": session_id,
        "filename": resume_index.filename,
        "pages": resume_index.pages,
        "characters": resume_index.characters,
        "embedding_dim": resume_index.dim,
        "chunks": resume_index.chunks,
    }


# ---------------------------------------------------------------------------
# Ask question
# ---------------------------------------------------------------------------

@app.post("/api/ask")
def ask(
    req: AskRequest,
):
    resume_index = store.get(
        req.session_id
    )

    if resume_index is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "This session has expired. "
                "Upload the resume again."
            ),
        )

    question = req.question.strip()
    job_role = req.job_role.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        result = rag.answer(
            resume_index,
            question,
            job_role,
            [
                turn.model_dump()
                for turn in req.history
            ],
        )

    except rag.UserError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except rag.AIServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except Exception as exc:  # noqa: BLE001
        log.exception(
            "Answering failed"
        )

        raise _ai_error(exc)

    return result


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------

@app.delete(
    "/api/session/{session_id}"
)
def delete_session(
    session_id: str,
):
    store.delete(session_id)

    return {
        "ok": True
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

# Must be mounted last so it doesn't shadow /api routes.
app.mount(
    "/",
    StaticFiles(
        directory=STATIC_DIR,
        html=True,
    ),
    name="static",
)
