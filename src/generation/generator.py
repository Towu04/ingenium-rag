"""Citation-grounded answer generator with a minimal, pluggable local-LLM interface.

Defaults to `ExtractiveStubLLM` so the full pipeline and tests run end-to-end without
requiring a running LLM server. When a local OpenAI-compatible server (Ollama, vLLM,
llama.cpp) is ready, plug in `OpenAICompatibleLLM(base_url=..., model=...)`.
"""

from collections.abc import Sequence
from typing import Protocol

import httpx

from domain import Answer, SearchResult
from generation.citations import citations_for, format_citation
from generation.prompts import SYSTEM_PROMPT, build_user_prompt


class LLMClient(Protocol):
    """Minimal chat-completion interface for local or remote LLM backends."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class ExtractiveStubLLM:
    """Deterministic zero-dependency stub used until the local LLM backend is wired."""

    def __init__(self, max_excerpt_chars: int = 600) -> None:
        self.max_excerpt_chars = max_excerpt_chars

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Surface the question and compact numbered excerpts so retrieval quality can be inspected.
        marker = "\n\nQuestion:\n"
        if marker in user_prompt:
            context_part, question_part = user_prompt.split(marker, 1)
        else:
            context_part, question_part = user_prompt, ""
        excerpts = [
            block.strip()[: self.max_excerpt_chars]
            for block in context_part.removeprefix("Context:\n").split("\n\n---\n\n")
            if block.strip()
        ]
        body = "\n\n".join(excerpts)
        return f"[Stub LLM — retrieval preview for '{question_part.strip()}']\n\n{body}"


class OpenAICompatibleLLM:
    """Minimal client for local OpenAI-compatible servers (Ollama, vLLM, llama.cpp)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = httpx.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


class GroundedGenerator:
    """Synthesizes a citation-grounded `Answer` from reranked `SearchResult` parent chunks."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm: LLMClient = llm or ExtractiveStubLLM()

    def generate(self, question: str, results: Sequence[SearchResult]) -> Answer:
        result_list = list(results)
        if not result_list:
            return Answer(
                text="I could not find supporting information in the indexed documents.",
                citations=[],
                retrieved_results=[],
            )

        user_prompt = build_user_prompt(question, result_list)
        text = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        citations = citations_for(result_list)
        return Answer(
            text=text,
            citations=citations,
            retrieved_results=result_list,
        )

    @staticmethod
    def format_references(answer: Answer) -> str:
        return "\n".join(
            f"[{idx}] {format_citation(c)}" for idx, c in enumerate(answer.citations, start=1)
        )
