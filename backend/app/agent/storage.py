"""SQLite persistence for agent sessions, conversation turns and improvement plans.

Every row hangs off a prediction (ON DELETE CASCADE), so the prediction-history retention purge
also removes the agent data. Conversation turns are append-only: each stored `content` is exactly
what was sent to / received from the LLM, so replayed history never changes.
"""

import json
import secrets
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

from backend.app.store import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    context TEXT NOT NULL,
    explanation TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    display_text TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    horizon_days INTEGER NOT NULL,
    goals TEXT,
    title TEXT NOT NULL,
    overview TEXT NOT NULL,
    start_date TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_plan_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES agent_plans(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    why_it_matters TEXT NOT NULL,
    priority TEXT NOT NULL,
    week INTEGER NOT NULL,
    target_date TEXT NOT NULL,
    basis TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_agent_plans_session ON agent_plans(session_id, id);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id():
    return secrets.token_urlsafe(18)


class AgentStore:
    def __init__(self, path):
        self.path = path
        with closing(connect(path)) as conn, conn:
            conn.executescript(SCHEMA)

    def create_session(self, prediction_id, model, context, explanation, request_text, response_text):
        session_id, now = new_session_id(), _now()
        with closing(connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO agent_sessions (id, prediction_id, created_at, model, context, explanation) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, prediction_id, now, model, json.dumps(context), json.dumps(explanation)))
            conn.executemany(
                "INSERT INTO agent_messages (session_id, role, kind, content, display_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(session_id, "user", "explanation_request", request_text, None, now),
                 (session_id, "assistant", "explanation", response_text, None, now)])
        return self.get_session(session_id)

    def get_session(self, session_id):
        with closing(connect(self.path)) as conn:
            row = conn.execute("SELECT * FROM agent_sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["context"] = json.loads(d["context"])
        d["explanation"] = json.loads(d["explanation"])
        return d

    def transcript(self, session_id):
        """Turns exactly as sent to the LLM, oldest first."""
        with closing(connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT role, kind, content, display_text, created_at FROM agent_messages "
                "WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def chat(self, session_id):
        return [{"role": t["role"], "text": t["display_text"] if t["role"] == "user" else t["content"],
                 "created_at": t["created_at"]}
                for t in self.transcript(session_id) if t["kind"] in ("question", "answer")]

    def question_count(self, session_id):
        with closing(connect(self.path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM agent_messages WHERE session_id = ? AND kind = 'question'",
                                (session_id,)).fetchone()[0]

    def add_turn(self, session_id, kinds, request_text, response_text, display_text=None):
        now = _now()
        with closing(connect(self.path)) as conn, conn:
            conn.executemany(
                "INSERT INTO agent_messages (session_id, role, kind, content, display_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(session_id, "user", kinds[0], request_text, display_text, now),
                 (session_id, "assistant", kinds[1], response_text, None, now)])

    def save_plan(self, session_id, horizon_days, goals, draft, model, start=None):
        start = start or date.today()
        now = _now()
        with closing(connect(self.path)) as conn, conn:
            cur = conn.execute(
                "INSERT INTO agent_plans (session_id, horizon_days, goals, title, overview, start_date, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, horizon_days, goals, draft.title, draft.overview, start.isoformat(), model, now))
            plan_id = cur.lastrowid
            tasks = sorted(draft.tasks, key=lambda t: t.week)
            conn.executemany(
                "INSERT INTO agent_plan_tasks (plan_id, position, title, description, why_it_matters, priority, "
                "week, target_date, basis) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(plan_id, i, t.title, t.description, t.why_it_matters, t.priority, t.week,
                  min(start + timedelta(days=t.week * 7 - 1), start + timedelta(days=horizon_days - 1)).isoformat(),
                  t.basis) for i, t in enumerate(tasks)])
        return self.get_plan(plan_id)

    def get_plan(self, plan_id):
        with closing(connect(self.path)) as conn:
            plan = conn.execute("SELECT * FROM agent_plans WHERE id = ?", (plan_id,)).fetchone()
            if plan is None:
                return None
            tasks = conn.execute("SELECT * FROM agent_plan_tasks WHERE plan_id = ? ORDER BY position",
                                 (plan_id,)).fetchall()
        tasks = [{**dict(t), "done": bool(t["done"])} for t in tasks]
        done = sum(t["done"] for t in tasks)
        return {**dict(plan), "tasks": tasks,
                "progress": {"completed": done, "total": len(tasks),
                             "percent": round(100 * done / len(tasks)) if tasks else 0}}

    def latest_plan(self, session_id):
        with closing(connect(self.path)) as conn:
            row = conn.execute("SELECT id FROM agent_plans WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                               (session_id,)).fetchone()
        return self.get_plan(row["id"]) if row else None

    def set_task_done(self, session_id, task_id, done):
        """Update a task that belongs to one of this session's plans; returns the plan or None."""
        with closing(connect(self.path)) as conn, conn:
            row = conn.execute(
                "SELECT t.plan_id FROM agent_plan_tasks t JOIN agent_plans p ON p.id = t.plan_id "
                "WHERE t.id = ? AND p.session_id = ?", (task_id, session_id)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE agent_plan_tasks SET done = ?, completed_at = ? WHERE id = ?",
                         (int(done), _now() if done else None, task_id))
        return self.get_plan(row["plan_id"])
