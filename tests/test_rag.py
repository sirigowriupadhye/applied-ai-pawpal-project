import pytest

from rag_assistant import (
    EvidenceOnlyGenerator,
    GenerationError,
    KnowledgeRetriever,
    PetCareAssistant,
)
from rag_assistant.generation import OllamaGenerator


class RecordingGenerator:
    label = "Fake local model for tests"

    def __init__(self, answer="Keep clean, fresh water available."):
        self.answer = answer
        self.calls = []

    def generate(self, question, context):
        self.calls.append((question, tuple(context)))
        return self.answer


def test_retriever_finds_food_safety_passage_and_source():
    results = KnowledgeRetriever().retrieve("Can dogs eat chocolate?")

    assert results[0].id == "unsafe-foods"
    assert "ASPCA" in results[0].source_name
    assert results[0].source_url.startswith("https://")


def test_retriever_drops_weak_matches_for_specific_question():
    results = KnowledgeRetriever().retrieve(
        "What foods can be harmful, such as chocolate?"
    )

    assert [chunk.id for chunk in results] == ["unsafe-foods"]


def test_food_safety_question_does_not_include_unrelated_diet_passage():
    results = KnowledgeRetriever().retrieve("What foods can be harmful to pets?")

    assert [chunk.id for chunk in results] == ["unsafe-foods"]


def test_retriever_keeps_activity_question_with_its_matching_topic():
    results = KnowledgeRetriever().retrieve("What kind of activity is suitable?")

    assert [chunk.id for chunk in results] == ["safe-activity"]


def test_retriever_returns_no_result_for_unrelated_question():
    assert KnowledgeRetriever().retrieve("How do telescopes observe distant stars?") == []


def test_missing_knowledge_file_has_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="knowledge base could not be loaded"):
        KnowledgeRetriever(tmp_path / "missing.json")


def test_assistant_retrieves_context_before_generation_and_cites_it():
    generator = RecordingGenerator()
    result = PetCareAssistant(generator=generator).ask("How should I provide water?")

    assert len(generator.calls) == 1
    assert generator.calls[0][1] == result.sources
    assert result.sources[0].id == "daily-basics"
    assert result.answer == generator.answer
    assert result.generation_method == generator.label


def test_insufficient_information_does_not_call_generator():
    generator = RecordingGenerator()
    result = PetCareAssistant(generator=generator).ask("How do telescopes work?")

    assert result.insufficient is True
    assert result.sources == ()
    assert generator.calls == []
    assert "couldn't find enough" in result.answer


def test_retrieval_error_returns_safe_message_and_does_not_log_question(caplog):
    class BrokenRetriever:
        def retrieve(self, question):
            raise OSError("private retrieval detail")

    generator = RecordingGenerator()
    question = "My private pet record retrieval question"
    result = PetCareAssistant(retriever=BrokenRetriever(), generator=generator).ask(question)

    assert result.insufficient is True
    assert result.sources == ()
    assert result.generation_method == "Not generated (retrieval error)"
    assert "could not access" in result.answer
    assert generator.calls == []
    assert "OSError" in caplog.text
    assert question not in caplog.text
    assert "private retrieval detail" not in caplog.text


@pytest.mark.parametrize("question", ["", "  ", "x", None, "a" * 501, "hi\x00there"])
def test_invalid_question_is_rejected(question):
    with pytest.raises(ValueError):
        PetCareAssistant().ask(question)


def test_model_error_falls_back_to_retrieved_text():
    class BrokenGenerator:
        label = "Unavailable model"

        def generate(self, question, context):
            raise GenerationError("service unavailable")

    result = PetCareAssistant(generator=BrokenGenerator()).ask("How should I provide water?")

    assert result.warning is not None
    assert "Ollama service is unavailable" in result.warning
    assert result.generation_method == EvidenceOnlyGenerator().label
    assert "fresh water" in result.answer
    assert result.sources[0].id == "daily-basics"


def test_ungrounded_model_answer_is_replaced_with_evidence_only_answer():
    class UngroundedGenerator:
        label = "Fake local model"

        def generate(self, question, context):
            return "The moon is made of cheese."

    result = PetCareAssistant(generator=UngroundedGenerator()).ask(
        "How should I provide water?"
    )

    assert result.warning is not None
    assert "did not pass safety or context checks" in result.warning
    assert result.generation_method == EvidenceOnlyGenerator().label
    assert "fresh water" in result.answer


def test_unsafe_model_answer_is_replaced_with_evidence_only_answer():
    class UnsafeGenerator:
        label = "Fake local model"

        def generate(self, question, context):
            return "Give your pet a medication dose now."

    result = PetCareAssistant(generator=UnsafeGenerator()).ask(
        "How should I provide water?"
    )

    assert result.warning is not None
    assert result.generation_method == EvidenceOnlyGenerator().label
    assert "fresh water" in result.answer


def test_ollama_service_error_is_reported_as_generation_error(monkeypatch):
    from urllib.error import URLError

    def unavailable(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr("rag_assistant.generation.urlopen", unavailable)
    context = KnowledgeRetriever().retrieve("How should I provide water?")

    with pytest.raises(GenerationError):
        OllamaGenerator().generate("How should I provide water?", context)


def test_health_diagnosis_request_gets_safety_response_without_generation():
    generator = RecordingGenerator()
    result = PetCareAssistant(generator=generator).ask(
        "Can you diagnose my dog's symptoms?"
    )

    assert result.safety_redirect is True
    assert generator.calls == []
    assert "veterinarian" in result.answer