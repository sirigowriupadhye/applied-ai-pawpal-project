"""Context-grounded answer generation using a local Ollama service."""

import json
import os
from typing import Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .retrieval import KnowledgeChunk


class GenerationError(RuntimeError):
    """Raised when the configured local generation service is unavailable."""


class AnswerGenerator(Protocol):
    @property
    def label(self) -> str: ...

    def generate(self, question: str, context: Sequence[KnowledgeChunk]) -> str: ...


class OllamaGenerator:
    """Use a local Ollama model; no API key or paid service is required."""

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 25.0,
    ) -> None:
        self.model = model or os.getenv("PAWPAL_OLLAMA_MODEL", "gemma3:1b")
        self.endpoint = endpoint or os.getenv(
            "PAWPAL_OLLAMA_URL", "http://localhost:11434/api/generate"
        )
        self.timeout = timeout

    @property
    def label(self) -> str:
        return f"Local Ollama model ({self.model})"

    def generate(self, question: str, context: Sequence[KnowledgeChunk]) -> str:
        if not context:
            raise GenerationError("Generation requires retrieved context.")
        references = "\n\n".join(
            f"[{chunk.id}] {chunk.title}: {chunk.content}" for chunk in context
        )
        prompt = (
            "You are PawPal+, a general pet-care information assistant. Answer the "
            "user using only the supplied reference passages. Do not add facts that "
            "are not supported by them. Do not diagnose, recommend medication, or "
            "give treatment instructions. If the passages do not answer the question, "
            "say that you cannot confidently answer. For possible poisoning, serious "
            "symptoms, or urgent concerns, direct the user to a veterinarian or an "
            "emergency veterinary clinic. Keep the answer concise.\n\n"
            f"Reference passages:\n{references}\n\n"
            f"Question: {question}\nAnswer:"
        )
        request = Request(
            self.endpoint,
            data=json.dumps(
                {"model": self.model, "prompt": prompt, "stream": False}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise GenerationError("The local model service request failed.") from error

        answer = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError("The local model returned no answer.")
        return answer.strip()


class EvidenceOnlyGenerator:
    """Make a transparent extractive response when no model can be reached."""

    @property
    def label(self) -> str:
        return "Evidence-only extractive fallback (no generative model used)"

    def generate(self, question: str, context: Sequence[KnowledgeChunk]) -> str:
        del question
        if not context:
            raise GenerationError("Generation requires retrieved context.")
        passages = " ".join(chunk.content for chunk in context[:2])
        return f"The available references say: {passages}"