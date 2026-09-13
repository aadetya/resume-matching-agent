"""Evaluation must distinguish finite shortlist capacity from perfect recall."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "selection_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_selection.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_ten_slots_cannot_recall_sixteen_positive_candidates():
    grades = {str(n): 3 for n in range(16)}
    result = benchmark.selection_metrics(list(grades), grades)
    assert result["recall_at_10"] == 10 / 16
    assert result["attainable_recall_at_10"] == 1
    assert result["strongest_recall_at_10"] == 10 / 16


def test_set_quality_is_order_independent_and_rewards_stronger_evidence():
    grades = {"strong": 3, "suitable": 2, "borderline": 1}
    best = benchmark.selection_metrics(["strong", "suitable"], grades, k=2)
    reordered = benchmark.selection_metrics(["suitable", "strong"], grades, k=2)
    weaker = benchmark.selection_metrics(["suitable", "borderline"], grades, k=2)
    assert best == reordered and best["set_gain_at_10"] == 1
    assert weaker["set_gain_at_10"] == 0.4
    assert weaker["strongest_recall_at_10"] == 0


def test_query_with_no_positive_judgments_has_undefined_positive_recall():
    result = benchmark.selection_metrics(["A"], {"A": 1})
    assert result["recall_at_10"] is None
    assert result["attainable_recall_at_10"] is None
    assert result["strongest_recall_at_10"] is None


def test_unjudged_candidates_cannot_be_treated_as_negative_labels():
    with pytest.raises(ValueError, match="frozen judgment"):
        benchmark.selection_metrics(["UNJUDGED"], {"A": 3})


def test_conversation_benchmark_sends_each_actual_paraphrase_without_expected_requirements():
    from screening_agent.contracts import Requirements

    sent, closed = [], []
    expected = Requirements(must_have=[["python"]], minimum_years=3).model_dump(mode="json")
    queries = [
        {
            "job_id": "literal",
            "query": "Find Python candidates with three years overall experience.",
            "requirements": expected,
        },
        {
            "job_id": "paraphrase",
            "query": "Three years in work and Python skills, please.",
            "requirements": expected,
        },
    ]

    class Session:
        def send(self, text):
            sent.append(text)
            return {
                "requirements": expected,
                "shortlist": [{"candidate_id": "A", "assessments": []}],
                "search_result": {"retrieval_audit": {"context_query": "Python; 3 overall years"}},
            }

        def close(self):
            closed.append(True)

    rows = benchmark.evaluate_conversations(
        SimpleNamespace(candidates={"A": None}),
        queries,
        {q["job_id"]: {"A": 3} for q in queries},
        session_factory=lambda _: Session(),
    )
    assert sent == [q["query"] for q in queries] and len(closed) == 2
    assert all(r["interpretation_tested"] and not r["interpretation_errors"] for r in rows)
    # Equivalent meanings can legitimately yield the same ranking input. The
    # critical distinction is whether each original string reached the graph.
    assert rows[0]["ranking_input"] == rows[1]["ranking_input"]


def test_wrong_interpretation_cannot_receive_a_successful_ranking_accuracy_measure():
    from screening_agent.contracts import Requirements

    actual = Requirements(must_have=[["python"]]).model_dump(mode="json")
    wanted = Requirements(must_have=[["python"], ["sql"]]).model_dump(mode="json")
    session = SimpleNamespace(
        send=lambda text: {"requirements": actual, "shortlist": [{"candidate_id": "A"}]},
        close=lambda: None,
    )
    rows = benchmark.evaluate_conversations(
        SimpleNamespace(candidates={"A": None}),
        [{"job_id": "both", "query": "Python and SQL", "requirements": wanted}],
        {"both": {"A": 3}},
        session_factory=lambda _: session,
    )
    assert rows[0]["interpretation_errors"]
    assert rows[0]["metrics"] is None


def test_generated_paraphrases_preserve_conditions_and_are_not_called_held_out():
    generator_spec = importlib.util.spec_from_file_location(
        "query_generator", Path(__file__).resolve().parents[1] / "scripts/generate_dataset.py"
    )
    generator = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator)
    profiles = [
        generator._profile(family, n) for family in generator.FAMILIES for n in range(1, 21)
    ]
    queries = generator.build_queries(profiles)
    for family in generator.FAMILIES:
        rows = {q["scenario"]: q for q in queries if q["job_id"].startswith(family + "_")}
        initial, paraphrase, duration = (
            rows[name] for name in ["development", "paraphrase", "skill_duration"]
        )

        def without_source(q):
            return {k: v for k, v in q["requirements"].items() if k != "source_text"}

        assert initial["query"] != paraphrase["query"]
        assert without_source(initial) == without_source(paraphrase)
        assert duration["requirements"]["minimum_years"] == 0
        assert duration["requirements"]["skill_years"]
        for q in rows.values():
            assert q["split"] == "development"
            assert "holdout" not in q["job_id"]
            assert all(
                skill.casefold() in q["query"].casefold()
                for skill in q["requirements"]["nice_to_have"]
            )


def test_semantic_reviewer_is_blinded_and_requires_real_source_quotes():
    review_spec = importlib.util.spec_from_file_location(
        "interpretation_reviewer",
        Path(__file__).resolve().parents[1] / "scripts/review_interpretations.py",
    )
    review = importlib.util.module_from_spec(review_spec)
    sys.modules[review_spec.name] = review
    review_spec.loader.exec_module(review)
    row = {
        "query": "Python or Java",
        "requirements": {},
        "expected_requirements": {},
        "top10": ["SECRET_CANDIDATE"],
        "state": {"rankings": []},
    }
    assert set(review.review_payload(row)) == {
        "request",
        "interpretation",
        "reference_requirements",
    }
    findings = [
        {
            "dimension": dimension,
            "faithful": True,
            "reason": "Equivalent meaning",
            "request_quote": "Python or Java",
        }
        for dimension in [
            "mandatory_conditions",
            "preferences",
            "experience_scope",
            "added_restrictions",
        ]
    ]
    parsed = review.InterpretationReview(
        findings=findings, reference_matches_request=True, reference_reason="Same request"
    )
    review.validate_review(parsed, row["query"])
    parsed.findings[0].request_quote = "Python and Java"
    with pytest.raises(ValueError, match="absent"):
        review.validate_review(parsed, row["query"])
    parsed.findings[0].request_quote = ""
    parsed.findings[0].dimension = "preferences"
    with pytest.raises(ValueError, match="exactly once"):
        review.validate_review(parsed, row["query"])
