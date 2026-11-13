"""Run a deterministic retrieval and grounding check without a model service."""

from rag_assistant import EvidenceOnlyGenerator, PetCareAssistant
from rag_assistant.retrieval import tokenize


CASES = (
    ("How do I provide fresh water?", "daily-basics"),
    ("What foods can be harmful, such as chocolate?", "unsafe-foods"),
    ("How often should my pet have a checkup?", "preventive-care"),
    ("What kind of activity is suitable?", "safe-activity"),
)


def main() -> None:
    assistant = PetCareAssistant(generator=EvidenceOnlyGenerator())
    retrieval_hits = 0
    grounding_scores = []

    for question, expected_id in CASES:
        result = assistant.ask(question)
        retrieved_ids = {source.id for source in result.sources}
        hit = expected_id in retrieved_ids
        retrieval_hits += int(hit)

        answer_terms = tokenize(result.answer)
        context_terms = set().union(
            *(tokenize(source.content) for source in result.sources)
        ) if result.sources else set()
        overlap = len(answer_terms & context_terms) / max(len(answer_terms), 1)
        grounding_scores.append(overlap)
        print(
            f"{'PASS' if hit else 'FAIL'} retrieval: expected {expected_id}; "
            f"got {', '.join(sorted(retrieved_ids)) or 'no sources'}"
        )
        print(f"  answer-context overlap: {overlap:.0%}")

    retrieval_rate = retrieval_hits / len(CASES)
    mean_grounding = sum(grounding_scores) / len(grounding_scores)
    print(f"Retrieval hit rate: {retrieval_rate:.0%} ({retrieval_hits}/{len(CASES)})")
    print(f"Mean answer-context token overlap: {mean_grounding:.0%}")
    print("Grounding overlap is a simple lexical indicator, not a factuality proof.")


if __name__ == "__main__":
    main()