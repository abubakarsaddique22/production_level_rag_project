"""
Shared LLM-judge wrapper for DeepEval, backed by Ollama.

DeepEval's built-in metrics default to an OpenAI judge. This wraps an
Ollama model (local OR cloud) behind DeepEval's `DeepEvalBaseLLM`
interface so metrics.py (retrieval) and ragas_runner.py (generation)
can both pass `model=get_judge()` into any DeepEval metric.

Two ways to run the SAME code, switched by environment variable only
(12-factor style -- see blueprint Step C), no code change needed:

  Local dev (model already pulled with `ollama pull <model>`):
    OLLAMA_BASE_URL=http://localhost:11434/v1
    OLLAMA_API_KEY=ollama        # ignored by local Ollama, but the
                                  # OpenAI client requires *some* value

  Deployed / CI (e.g. GitHub Actions runner, no local Ollama, no GPU):
    OLLAMA_BASE_URL=https://ollama.com/v1
    OLLAMA_API_KEY=<real key from ollama.com/settings/keys>

Ollama's "-cloud" tagged models (e.g. gpt-oss:20b-cloud) run on
Ollama's own infrastructure, not on the calling machine -- so pointing
at the cloud endpoint needs no local GPU and no model weights on the
deployment host.
"""

from __future__ import annotations

import asyncio
import os
import re

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM

from ..core.logging import get_logger

log = get_logger(__name__)

DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-oss:20b-cloud")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # dummy is fine for local

# Ollama Cloud rejects requests beyond its own concurrency limit with a
# 429 ("too many concurrent requests"). DeepEval's own AsyncConfig only
# limits concurrency at the TEST CASE level -- a single test case can
# still fan out several parallel judge calls internally (e.g.
# ContextualRelevancy issues one call per retrieved chunk). This
# semaphore caps the REAL number of in-flight requests to Ollama,
# regardless of how much DeepEval parallelizes above it.
OLLAMA_MAX_CONCURRENT = int(os.getenv("OLLAMA_MAX_CONCURRENT", "2"))


class OllamaJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge that talks to Ollama's OpenAI-compatible
    /v1 endpoint (local server or Ollama Cloud -- same code, different URL).
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        max_retries: int = 5,
    ):
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries

        # Generous timeout: calls now queue behind a semaphore (see
        # OLLAMA_MAX_CONCURRENT), so a request may sit waiting for its
        # turn before it even starts -- a short client timeout would
        # # fire while it's still queued, not actually stuck.
        # self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        # self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._semaphore = asyncio.Semaphore(OLLAMA_MAX_CONCURRENT)

    def load_model(self):
        return self.client

    def _build_kwargs(self, prompt: str, schema: type[BaseModel] | None):
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8000,
            "reasoning_effort": "low"
        }
        # DeepEval passes a pydantic schema for structured judge output
        # (score + reason); ask Ollama to return exactly that shape.
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "deepeval_output",
                    "strict": False,
                    "schema": schema.model_json_schema(),
                },
            }
        return kwargs

    @staticmethod
    def _parse(content: str, schema: type[BaseModel] | None):
        if schema is None:
            return content
        # Some local models wrap JSON in ```json fences despite the
        # response_format instruction -- strip them defensively.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        return schema.model_validate_json(cleaned)

    @staticmethod
    def _extract(response) -> str:
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise ValueError(
                f"Ollama judge returned an empty response "
                f"(finish_reason={choice.finish_reason})"
            )
        return content

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    **self._build_kwargs(prompt, schema)
                )
                return self._parse(self._extract(response), schema)
            except Exception as e:  # noqa: BLE001 -- retry any transient error
                last_err = e
                log.warning(
                    "ollama_judge_retry",
                    extra={"attempt": attempt, "error": str(e)},
                )
        raise ValueError(f"Ollama judge failed after {self.max_retries} retries: {last_err}")

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        last_err = None
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await self.async_client.chat.completions.create(
                        **self._build_kwargs(prompt, schema)
                    )
                    return self._parse(self._extract(response), schema)
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    is_rate_limit = "429" in str(e) or "too many concurrent" in str(e).lower()
                    # Rate limits need a real cooldown, not a quick retry --
                    # a short backoff just hits the same 429 again.
                    wait_s = (10 * (attempt + 1)) if is_rate_limit else (2 ** attempt)
                    log.warning(
                        "ollama_judge_retry",
                        extra={"attempt": attempt, "wait_s": wait_s, "error": str(e)},
                    )
                    await asyncio.sleep(wait_s)
        raise ValueError(f"Ollama judge failed after {self.max_retries} retries: {last_err}")

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"


_cached_judge: OllamaJudge | None = None


def get_judge() -> OllamaJudge:
    """Single shared judge instance for the whole evaluation run --
    both metrics.py and ragas_runner.py should call this instead of
    constructing OllamaJudge themselves, so a config change (env var)
    affects every metric consistently."""
    global _cached_judge
    if _cached_judge is None:
        _cached_judge = OllamaJudge()
        log.info(
            "judge_initialized",
            extra={"model": DEFAULT_JUDGE_MODEL, "base_url": DEFAULT_BASE_URL},
        )
    return _cached_judge