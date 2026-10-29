# Security

## Intended deployment

This project is built for **local / academic use**. The commands in the README bind the API and
the Vite dev server to `127.0.0.1` only. Do not expose the backend to a network without the
controls listed under *Before deploying publicly*.

## What the application protects against

| Control | Where |
|---|---|
| Strict request validation (types, bounds, allowed categories; unknown fields rejected) | `backend/app/schemas.py` |
| Request body limit, default 16 KiB (HTTP 413), including chunked bodies | `backend/app/limits.py`, `CREDIT_MAX_BODY_BYTES` |
| CORS restricted to configured origins; no credentials | `backend/app/main.py`, `CORS_ORIGINS` |
| Parameterised SQL only | `backend/app/store.py` |
| Errors name files only, never server paths or exception text; details go to the server log | `backend/app/store.py` |
| History retention: rows older than 30 days are purged; history can be switched off | `CREDIT_HISTORY_RETENTION_DAYS`, `CREDIT_STORE_HISTORY` |
| Per-prediction access tokens for history and agent sessions; operator token for the full listing | `backend/app/store.py`, `backend/app/main.py` |
| LLM key read only by the backend (env or git-ignored `.env`); never sent to the browser | `backend/app/agent/llm.py`, `tests/test_frontend_secrets.py` |
| Agent: no tools, fixed system prompt, schema-validated output, nothing stored on provider failure, timeouts, rate limit | `backend/app/agent/` |
| Agent data (sessions, messages, plans) is cascade-deleted with its prediction by the retention purge | `backend/app/agent/storage.py` |
| No app-level logging of applicant inputs (the uvicorn access log records method and path only) | — |
| Frontend renders API text through React escaping; links from API data must be http(s) | `frontend/src/format.js` |

## Before deploying publicly (not implemented)

* **Authentication and authorisation.** There are no user accounts. Stored predictions and agent
  sessions are protected by unguessable per-prediction access tokens (only their SHA-256 hash is
  stored), and `GET /predictions` (all rows) requires the operator token `CREDIT_ADMIN_TOKEN`.
  Tokens live in the browser's local storage, so anyone with access to that browser profile can
  read its predictions. For multi-user deployments put the API behind real authentication.
  Client-side validation in the frontend is a usability feature, not a security control.
* **LLM costs and abuse.** The built-in agent rate limit is per process and per client IP. Behind a
  proxy every client shares the proxy's IP, so enforce per-user limits and spend caps at the gateway
  and in the Anthropic console.
* **Rate limiting.** Each `/predict` call runs a 500-tree Random Forest. Rate-limit at the proxy.
* **TLS** termination at the proxy, plus a `CORS_ORIGINS` value listing only the real frontend origin.
* **Production frontend.** Serve `npm run build` output (`frontend/dist`) from a static host.
  Never use the Vite dev server publicly.
* **API docs.** `/docs` and `/openapi.json` are public; the frontend reads `/openapi.json` to
  build its form.

## Model artifact trust

`artifacts/model.joblib` is a pickle. Loading a pickle can execute arbitrary code. The API only
loads artifacts from the operator-configured `CREDIT_ARTIFACT_DIR` and never accepts uploaded
models. Treat any change to `artifacts/model.joblib` in a pull request as a code change:
regenerate it yourself with `python -m ml.train` rather than trusting a submitted binary.

## Data

Agent requests send the stored prediction's features, model output and any details the user chose
to add to Anthropic's API for processing. Users should not enter names, account numbers or other
identifiers in the free-text fields.

`data/raw/german.data` is the public UCI Statlog German Credit dataset (CC BY 4.0). It contains
coded attributes only: no names, identifiers or contact details. Do not commit real customer
data or the local prediction database (`var/`, `*.db` are git-ignored).

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository rather than a public issue.
