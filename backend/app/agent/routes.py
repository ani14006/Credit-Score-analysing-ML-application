"""HTTP API for the AI Financial Wellness Agent.

Access control: every route requires the X-Prediction-Token issued by POST /predict for the
prediction the session belongs to. Unknown sessions and wrong tokens both return 404.
"""

from typing import Literal

from fastapi import APIRouter, Header, Path, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.app.agent import service
from backend.app.agent.context import UserFinancialContext
from backend.app.agent.llm import NOT_CONFIGURED, AgentError

router = APIRouter(prefix="/agent", tags=["agent"])

SessionId = Path(pattern=r"^[A-Za-z0-9_-]{16,64}$")
TokenHeader = Header(alias="X-Prediction-Token", min_length=20, max_length=128)


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: int = Field(ge=1)
    user_context: UserFinancialContext | None = None


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1000)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_days: Literal[30, 60, 90] = 30
    goals: str | None = Field(None, max_length=300)


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: int = Field(ge=1)
    question: str | None = Field(None, min_length=1, max_length=1000)


class TaskUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    done: bool


class _Fail(Exception):
    def __init__(self, status, detail, headers=None):
        self.status, self.detail, self.headers = status, detail, headers


def error_response(exc):
    if isinstance(exc, AgentError):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code, headers=headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status, headers=exc.headers)


def _state(request):
    s = request.app.state
    if not s.store.ready:
        raise _Fail(503, s.store.error)
    if s.agent_store is None:
        raise _Fail(409, "The agent needs stored predictions, but prediction history is disabled "
                         "(CREDIT_STORE_HISTORY=false).")
    return s


def _require_provider(request):
    provider = request.app.state.agent_provider
    if provider is None:
        raise NOT_CONFIGURED
    wait = request.app.state.agent_rate_limiter.check(request.client.host if request.client else "unknown")
    if wait:
        raise _Fail(429, "Too many AI requests. Please wait before trying again.", {"Retry-After": str(wait)})
    return provider


def _authorized_session(s, session_id, token):
    session = s.agent_store.get_session(session_id)
    if session is None or s.log.get_authorized(session["prediction_id"], token) is None:
        raise _Fail(404, "Agent session not found.")
    return session


def _handle(fn):
    try:
        return fn()
    except (AgentError, _Fail) as exc:
        return error_response(exc)


@router.get("/status")
def status(request: Request):
    provider = request.app.state.agent_provider
    return {
        "configured": provider is not None,
        "provider": provider.name if provider else None,
        "model": provider.model if provider else None,
        "message": None if provider else NOT_CONFIGURED.message,
        "history_enabled": request.app.state.agent_store is not None,
    }


@router.post("/sessions", status_code=201)
def start_session(body: StartSessionRequest, request: Request, token: str = TokenHeader):
    def run():
        s = _state(request)
        prediction = s.log.get_authorized(body.prediction_id, token)
        if prediction is None:
            raise _Fail(404, "Prediction not found.")
        provider = _require_provider(request)
        return service.start_session(provider, s.store, s.agent_store, prediction, body.user_context)
    return _handle(run)


@router.get("/sessions/{session_id}")
def get_session(request: Request, session_id: str = SessionId, token: str = TokenHeader):
    def run():
        s = _state(request)
        return service.session_payload(s.agent_store, _authorized_session(s, session_id, token))
    return _handle(run)


@router.post("/sessions/{session_id}/messages")
def ask(body: QuestionRequest, request: Request, session_id: str = SessionId, token: str = TokenHeader):
    def run():
        s = _state(request)
        session = _authorized_session(s, session_id, token)
        return service.ask(_require_provider(request), s.agent_store, session, body.question.strip())
    return _handle(run)


@router.post("/sessions/{session_id}/plan", status_code=201)
def create_plan(body: PlanRequest, request: Request, session_id: str = SessionId, token: str = TokenHeader):
    def run():
        s = _state(request)
        session = _authorized_session(s, session_id, token)
        return service.create_plan(_require_provider(request), s.agent_store, session, body.horizon_days,
                                   body.goals.strip() if body.goals else None)
    return _handle(run)


@router.get("/sessions/{session_id}/plan")
def get_plan(request: Request, session_id: str = SessionId, token: str = TokenHeader):
    def run():
        s = _state(request)
        session = _authorized_session(s, session_id, token)
        plan = s.agent_store.latest_plan(session["id"])
        if plan is None:
            raise _Fail(404, "No plan has been generated for this session yet.")
        return plan
    return _handle(run)


@router.patch("/sessions/{session_id}/plan/tasks/{task_id}")
def update_task(body: TaskUpdate, request: Request, session_id: str = SessionId,
                task_id: int = Path(ge=1), token: str = TokenHeader):
    def run():
        s = _state(request)
        session = _authorized_session(s, session_id, token)
        plan = s.agent_store.set_task_done(session["id"], task_id, body.done)
        if plan is None:
            raise _Fail(404, "Task not found.")
        return plan
    return _handle(run)


@router.post("/text", response_class=PlainTextResponse)
def text(body: TextRequest, request: Request, token: str = TokenHeader):
    """Simple text API: plain-text explanation of a stored prediction, or an answer to `question`.
    Stateless - no session or message is stored. Errors are plain text with the usual status codes."""
    try:
        s = _state(request)
        prediction = s.log.get_authorized(body.prediction_id, token)
        if prediction is None:
            raise _Fail(404, "Prediction not found.")
        provider = _require_provider(request)
        question = body.question.strip() if body.question else None
        return PlainTextResponse(service.text_answer(provider, s.store, prediction, question))
    except AgentError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return PlainTextResponse(exc.message, status_code=exc.status_code, headers=headers)
    except _Fail as exc:
        return PlainTextResponse(exc.detail, status_code=exc.status, headers=exc.headers)
