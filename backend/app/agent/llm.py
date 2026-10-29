"""Server-side LLM providers (Anthropic Claude or Google Gemini). API keys are read from the backend
environment only; they are never sent to the browser. Failures raise AgentError - no substitute
text is produced."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


class AgentError(Exception):
    def __init__(self, status_code, message, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


NOT_CONFIGURED = AgentError(
    503, "The AI agent is not configured: set GEMINI_API_KEY or ANTHROPIC_API_KEY in the backend "
         "environment (see .env.example) and restart the backend.")


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key, model=DEFAULT_MODEL, timeout=90.0, max_retries=2):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def complete(self, system, messages, *, json_schema=None, effort="medium", max_tokens=8000):
        a = self._anthropic
        output_config = {"effort": effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config=output_config,
                cache_control={"type": "ephemeral"},
                # On a safety-classifier refusal, re-run on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (a.AuthenticationError, a.PermissionDeniedError):
            logger.error("LLM provider rejected the configured credentials")
            raise AgentError(502, "The AI provider rejected the backend's credentials. Check ANTHROPIC_API_KEY.")
        except a.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
            raise AgentError(503, "The AI provider is rate-limiting requests. Try again shortly.", retry_after)
        except a.APITimeoutError:
            raise AgentError(504, "The AI provider did not respond in time. Try again.")
        except a.APIConnectionError:
            raise AgentError(502, "Could not reach the AI provider. Check the backend's network connection.")
        except a.BadRequestError as exc:
            logger.error("LLM request rejected: %s", getattr(exc, "message", exc))
            raise AgentError(502, "The AI provider rejected the request.")
        except a.APIStatusError as exc:
            logger.error("LLM provider error status=%s", exc.status_code)
            raise AgentError(502, "The AI provider is unavailable right now. Try again later.")

        if response.stop_reason == "refusal":
            raise AgentError(502, "The AI model declined to answer this request.")
        if response.stop_reason == "max_tokens":
            raise AgentError(502, "The AI response was cut off before it finished. Try again.")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise AgentError(502, "The AI provider returned an empty response.")
        usage = getattr(response, "usage", None)
        return LLMResult(text=text, model=response.model,
                         input_tokens=getattr(usage, "input_tokens", None),
                         output_tokens=getattr(usage, "output_tokens", None))


class GeminiProvider:
    name = "gemini"

    # Finish reasons that mean the model stopped for a policy or content reason.
    _DECLINED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "OTHER"}

    def __init__(self, api_key, model=DEFAULT_GEMINI_MODEL, timeout=90.0, max_retries=1):
        import httpx
        from google import genai
        from google.genai import errors, types

        self._httpx, self._errors, self._types = httpx, errors, types
        self.model = model
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=int(timeout * 1000),  # milliseconds
            retry_options=types.HttpRetryOptions(attempts=max_retries + 1),
        ))

    def complete(self, system, messages, *, json_schema=None, effort="medium", max_tokens=8000):
        types, errors, httpx = self._types, self._errors, self._httpx
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])]) for m in messages]
        config = types.GenerateContentConfig(
            system_instruction=system,
            # Gemini's thinking tokens count against this limit, so leave headroom.
            max_output_tokens=max(max_tokens, 16384),
            response_mime_type="application/json" if json_schema is not None else None,
            response_json_schema=json_schema,
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        except errors.APIError as exc:
            code, message = getattr(exc, "code", None), (getattr(exc, "message", "") or "")
            if code in (401, 403) or (code == 400 and "api key" in message.lower()):
                logger.error("LLM provider rejected the configured credentials")
                raise AgentError(502, "The AI provider rejected the backend's credentials. Check GEMINI_API_KEY.")
            if code == 429:
                raise AgentError(503, "The AI provider is rate-limiting requests or the quota is used up. Try again shortly.")
            if code == 504:
                raise AgentError(504, "The AI provider did not respond in time. Try again.")
            if code == 503:
                raise AgentError(503, "The AI provider is overloaded right now. Try again in a moment.")
            if code == 404:
                logger.error("LLM model not available: %s", message[:200])
                raise AgentError(502, "The configured AI model is not available. Check GEMINI_MODEL.")
            if code == 400:
                logger.error("LLM request rejected: %s", message[:200])
                raise AgentError(502, "The AI provider rejected the request.")
            logger.error("LLM provider error status=%s", code)
            raise AgentError(502, "The AI provider is unavailable right now. Try again later.")
        except httpx.TimeoutException:
            raise AgentError(504, "The AI provider did not respond in time. Try again.")
        except httpx.TransportError:
            raise AgentError(502, "Could not reach the AI provider. Check the backend's network connection.")

        feedback = getattr(response, "prompt_feedback", None)
        if not response.candidates:
            if feedback is not None and getattr(feedback, "block_reason", None):
                raise AgentError(502, "The AI model declined to answer this request.")
            raise AgentError(502, "The AI provider returned an empty response.")
        reason = response.candidates[0].finish_reason
        reason = getattr(reason, "name", reason)
        if reason == "MAX_TOKENS":
            raise AgentError(502, "The AI response was cut off before it finished. Try again.")
        if reason in self._DECLINED:
            raise AgentError(502, "The AI model declined to answer this request.")
        text = (response.text or "").strip()
        if not text:
            raise AgentError(502, "The AI provider returned an empty response.")
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(text=text, model=getattr(response, "model_version", None) or self.model,
                         input_tokens=getattr(usage, "prompt_token_count", None),
                         output_tokens=getattr(usage, "candidates_token_count", None))


def build_provider():
    """The configured provider, or None when no API key is set.

    CREDIT_AGENT_PROVIDER selects `anthropic` or `gemini`; when unset, whichever key is present is
    used (Anthropic first)."""
    timeout = float(os.environ.get("CREDIT_AGENT_TIMEOUT_SECONDS", "90"))
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    choice = os.environ.get("CREDIT_AGENT_PROVIDER", "").strip().lower()
    if not choice:
        choice = "anthropic" if anthropic_key else "gemini" if gemini_key else ""
    if choice == "anthropic" and anthropic_key:
        return AnthropicProvider(api_key=anthropic_key, model=os.environ.get("CREDIT_AGENT_MODEL", DEFAULT_MODEL),
                                 timeout=timeout)
    if choice == "gemini" and gemini_key:
        return GeminiProvider(api_key=gemini_key, model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
                              timeout=timeout)
    if choice not in ("", "anthropic", "gemini"):
        logger.error("Unknown CREDIT_AGENT_PROVIDER=%r", choice)
    return None
