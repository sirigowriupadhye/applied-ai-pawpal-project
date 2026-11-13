"""Retrieval-augmented pet-care question answering for PawPal+."""

from .assistant import AssistantResponse, PetCareAssistant
from .generation import EvidenceOnlyGenerator, GenerationError, OllamaGenerator
from .retrieval import KnowledgeChunk, KnowledgeRetriever

__all__ = [
    "AssistantResponse",
    "EvidenceOnlyGenerator",
    "GenerationError",
    "KnowledgeChunk",
    "KnowledgeRetriever",
    "OllamaGenerator",
    "PetCareAssistant",
]