"""Run a deterministic retrieval and grounding check without a model service."""

import sys

from rag_assistant import EvidenceOnlyGenerator, PetCareAssistant
from rag_assistant.retrieval import tokenize


CASES = (
    ("How do I provide fresh water?", "daily-basics"),
    ("What foods can be harmful, such as chocolate?", "unsafe-foods"),
    ("How often should my pet have a checkup?", "preventive-care"),
    ("What kind of activity is suitable?", "safe-activity"),
)
MIN_ANSWER_CONTEXT_OVERLAP = 0.5


def main() -> int:
    assistant = PetCareAssistant(generator=EvidenceOnlyGenerator())
    passed_checks = 0
    total_checks = 0
    grounding_scores = []

    for question, expected_id in CASES:
        result = assistant.ask(question)
        retrieved_ids = {source.id for source in result.sources}
        hit = expected_id in retrieved_ids
        references_present = bool(result.sources) and all(
            source.source_name.strip() and source.source_url.startswith("https://")
            for source in result.sources
        )

        answer_terms = tokenize(result.answer)
        context_terms = set().union(
            *(tokenize(source.content) for source in result.sources)
        ) if result.sources else set()
        overlap = len(answer_terms & context_terms) / max(len(answer_terms), 1)
        grounding_scores.append(overlap)
        grounded = overlap >= MIN_ANSWER_CONTEXT_OVERLAP

        for label, passed in (
            (f"expected passage retrieved ({expected_id})", hit),
            ("source name and HTTPS URL returned", references_present),
            (
                f"answer/context overlap meets {MIN_ANSWER_CONTEXT_OVERLAP:.0%} minimum",
                grounded,
            ),
        ):
            total_checks += 1
            passed_checks += int(passed)
            print(f"{'PASS' if passed else 'FAIL'} {label}")
        print(f"  retrieved: {', '.join(sorted(retrieved_ids)) or 'no sources'}")
        print(f"  answer-context overlap: {overlap:.0%}")

    unsupported = assistant.ask("How do telescopes observe distant stars?")
    unsupported_handled = (
        unsupported.insufficient
        and not unsupported.sources
        and unsupported.generation_method == "Not generated (no relevant sources)"
    )
    total_checks += 1
    passed_checks += int(unsupported_handled)
    print(
        f"{'PASS' if unsupported_handled else 'FAIL'} unsupported question "
        "returns no-source response without generation"
    )

    retrieval_rate = sum(
        expected_id in {source.id for source in assistant.retriever.retrieve(question)}
        for question, expected_id in CASES
    ) / len(CASES)
    mean_grounding = sum(grounding_scores) / len(grounding_scores)
    print(f"Expected retrieval hit rate: {retrieval_rate:.0%} ({len(CASES)} cases)")
    print(f"Mean answer-context token overlap: {mean_grounding:.0%}")
    print("Grounding overlap is a simple lexical indicator, not a factuality proof.")
    print(f"Evaluation checks: {passed_checks}/{total_checks} passed")
    return 0 if passed_checks == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())