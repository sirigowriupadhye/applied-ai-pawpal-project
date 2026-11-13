"""Orchestrate retrieval, safety checks, and context-grounded generation."""

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from .generation import (
    AnswerGenerator,
    EvidenceOnlyGenerator,
    GenerationError,
    OllamaGenerator,
)
from .retrieval import KnowledgeChunk, KnowledgeRetriever, tokenize


logger = logging.getLogger(__name__)
_HIGH_RISK_TERMS = re.compile(
    r"\b(diagnos\w*|treat\w*|dosage|dose|medication|medicine|symptom\w*|"
    r"vomit\w*|diarr\w*|seizure\w*|bleed\w*|breath\w*|collapse|pain|"
    r"poison\w*|swallow\w*|ingest\w*)\b|\bate\b",
    re.IGNORECASE,
)
_UNSAFE_OUTPUT_TERMS = re.compile(
    r"\b(diagnos\w*|prescrib\w*|dosage|dose|medication|medicine|treat\w*)\b",
    re.IGNORECASE,
)
_GROUNDING_STOP_WORDS = {
    "available", "reference", "references", "say", "says", "based", "answer",
    "general", "pet", "pets", "care", "should", "could", "would", "please",
}


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    sources: tuple[KnowledgeChunk, ...]
    generation_method: str
    insufficient: bool = False
    safety_redirect: bool = False
    warning: str | None = None


class PetCareAssistant:
    """Retrieve sourced care information before generating any response."""

    def __init__(
        self,
        retriever: KnowledgeRetriever | None = None,
        generator: AnswerGenerator | None = None,
        fallback_generator: AnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever or KnowledgeRetriever()
        self.generator = generator or OllamaGenerator()
        self.fallback_generator = fallback_generator or EvidenceOnlyGenerator()

    def ask(self, question: str) -> AssistantResponse:
        """Answer a validated question using retrieved passages as context."""
        clean_question = self._validate_question(question)
        sources = tuple(self.retriever.retrieve(clean_question))
        logger.info("Pet-care retrieval completed: %d passage(s).", len(sources))

        if _HIGH_RISK_TERMS.search(clean_question):
            return AssistantResponse(
                answer=(
                    "I can't diagnose a condition or provide medication or treatment "
                    "instructions. For a possible harmful ingestion, concerning "
                    "symptoms, or an urgent health concern, contact your veterinarian "
                    "or an emergency veterinary clinic promptly."
                ),
                sources=sources,
                generation_method="Safety response (no model used)",
                safety_redirect=True,
            )

        if not sources:
            return AssistantResponse(
                answer=(
                    "I couldn't find enough relevant information in PawPal+'s "
                    "knowledge base to answer confidently. Please consult a "
                    "veterinarian for advice specific to your pet."
                ),
                sources=(),
                generation_method="Not generated (no relevant sources)",
                insufficient=True,
            )

        try:
            answer = self.generator.generate(clean_question, sources)
        except GenerationError as error:
            logger.warning(
                "Model service failed (%s); using retrieved text only.",
                type(error).__name__,
            )
            warning = "The local Ollama service is unavailable. Showing retrieved text only."
        except Exception as error:
            logger.warning(
                "Answer generation failed (%s); using retrieved text only.",
                type(error).__name__,
            )
            warning = "Answer generation failed. Showing retrieved text only."
        else:
            if self._uses_context(answer, sources):
                return AssistantResponse(
                    answer=answer,
                    sources=sources,
                    generation_method=self.generator.label,
                )
            logger.warning("Model response failed safety or context checks.")
            warning = (
                "The model response did not pass safety or context checks. "
                "Showing retrieved text only."
            )

        fallback_answer = self.fallback_generator.generate(clean_question, sources)
        return AssistantResponse(
            answer=fallback_answer,
            sources=sources,
            generation_method=self.fallback_generator.label,
            warning=warning,
        )

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str):
            raise ValueError("Enter your question as text.")
        clean_question = question.strip()
        if len(clean_question) < 3:
            raise ValueError("Enter a question with at least 3 characters.")
        if len(clean_question) > 500:
            raise ValueError("Keep your question under 500 characters.")
        if any(ord(character) < 32 and character not in "\t\n\r" for character in clean_question):
            raise ValueError("The question contains unsupported control characters.")
        return clean_question

    @staticmethod
    def _uses_context(answer: str, sources: Sequence[KnowledgeChunk]) -> bool:
        if not isinstance(answer, str) or not answer.strip():
            return False
        if _UNSAFE_OUTPUT_TERMS.search(answer):
            return False
        answer_terms = tokenize(answer) - _GROUNDING_STOP_WORDS
        context_terms = set().union(
            *(tokenize(f"{chunk.title} {chunk.content}") for chunk in sources)
        )
        return bool(answer_terms & context_terms)