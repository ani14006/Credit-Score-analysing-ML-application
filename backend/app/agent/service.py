"""Agent orchestration: verified context -> LLM -> validated output -> persistence.

The ML prediction is read from the database and never modified here. LLM output is validated
before anything is stored; on any failure nothing is persisted and an AgentError is raised.
"""

import json
import logging

from pydantic import ValidationError

from backend.app.agent import prompts
from backend.app.agent.context import build_context
from backend.app.agent.llm import AgentError

logger = logging.getLogger(__name__)

MAX_QUESTIONS_PER_SESSION = 40


def _parse(model_cls, text):
    try:
        return model_cls.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("LLM returned output that failed validation: %s", type(exc).__name__)
        raise AgentError(502, "The AI returned a response in an unexpected format. Try again.")


def _history(transcript):
    return [{"role": t["role"], "content": t["content"]} for t in transcript]


def _evidence(context):
    """The verified (non-AI) part of the context, returned to the UI as evidence."""
    return {k: context[k] for k in ("prediction", "model_sensitivity_analysis",
                                    "model_evaluation_on_held_out_test_set", "limitations",
                                    "information_not_collected", "user_supplied_unverified")}


def session_payload(agent_store, session):
    return {
        "session_id": session["id"],
        "prediction_id": session["prediction_id"],
        "created_at": session["created_at"],
        "model": session["model"],
        "explanation": session["explanation"],
        "evidence": _evidence(session["context"]),
        "messages": agent_store.chat(session["id"]),
        "plan": agent_store.latest_plan(session["id"]),
    }


def start_session(provider, model_store, agent_store, prediction, user_context):
    context = build_context(prediction, model_store, user_context)
    request_text = prompts.explanation_request(context)
    result = provider.complete(prompts.SYSTEM_PROMPT, [{"role": "user", "content": request_text}],
                               json_schema=prompts.EXPLANATION_SCHEMA, effort="medium", max_tokens=8000)
    explanation = _parse(prompts.Explanation, result.text)
    session = agent_store.create_session(prediction["id"], result.model, context,
                                         explanation.model_dump(), request_text, result.text)
    return session_payload(agent_store, session)


def text_answer(provider, model_store, prediction, question=None):
    """Stateless plain-text answer about one stored prediction. Nothing is persisted."""
    context = build_context(prediction, model_store)
    result = provider.complete(prompts.SYSTEM_PROMPT,
                               [{"role": "user", "content": prompts.text_request(context, question)}],
                               effort="low", max_tokens=4000)
    return result.text


def ask(provider, agent_store, session, question):
    if agent_store.question_count(session["id"]) >= MAX_QUESTIONS_PER_SESSION:
        raise AgentError(429, f"This session has reached {MAX_QUESTIONS_PER_SESSION} questions. Start a new analysis.")
    plan = agent_store.latest_plan(session["id"])
    snapshot = None
    if plan:
        snapshot = {"title": plan["title"], "horizon_days": plan["horizon_days"],
                    "tasks": [{"title": t["title"], "week": t["week"], "done": t["done"]} for t in plan["tasks"]]}
    request_text = prompts.question_request(question, snapshot)
    messages = _history(agent_store.transcript(session["id"])) + [{"role": "user", "content": request_text}]
    result = provider.complete(prompts.SYSTEM_PROMPT, messages, effort="low", max_tokens=4000)
    agent_store.add_turn(session["id"], ("question", "answer"), request_text, result.text, display_text=question)
    return {"answer": result.text, "model": result.model, "messages": agent_store.chat(session["id"])}


def create_plan(provider, agent_store, session, horizon_days, goals):
    request_text = prompts.plan_request(horizon_days, goals)
    messages = _history(agent_store.transcript(session["id"])) + [{"role": "user", "content": request_text}]
    result = provider.complete(prompts.SYSTEM_PROMPT, messages, json_schema=prompts.PLAN_SCHEMA,
                               effort="medium", max_tokens=8000)
    draft = _parse(prompts.PlanDraft, result.text)
    max_week = horizon_days // 7 + (1 if horizon_days % 7 else 0)
    if any(t.week > max_week for t in draft.tasks):
        raise AgentError(502, "The AI scheduled tasks outside the requested plan length. Try again.")
    agent_store.add_turn(session["id"], ("plan_request", "plan"), request_text, result.text)
    return agent_store.save_plan(session["id"], horizon_days, goals, draft, result.model)
