import random

from ptqa import judge


def test_number_fault_skips_citations():
    rng = random.Random(0)
    text = "Pay 12 euros within 30 days [2][3]."
    changed = judge.change_number(text, rng)
    assert changed != text
    assert changed.endswith("days [2][3].")
    assert changed.count("[") == 2


def test_year_moves_by_one():
    assert judge.change_number("Since 2024 [1].", random.Random(0)) == "Since 2025 [1]."


def test_agency_fault_and_no_op():
    assert judge.change_agency("Go to AIMA [1].", random.Random(0)) == "Go to Segurança Social [1]."
    assert judge.change_agency("Bring a passport [1].", random.Random(0)) is None


def test_faults_only_on_answered_questions():
    base = {"id": 1, "lang": "en", "answerable": True, "not_found": False}
    runs = {
        "rag": [
            {**base, "answer": "AIMA wants 2 photos [1]."},
            {
                **base,
                "id": 2,
                "answer": "The sources do not answer this question.",
                "not_found": True,
            },
            {**base, "id": 3, "answerable": False, "answer": "AIMA 5 [1]."},
        ]
    }
    cases = judge.inject_faults(runs)
    assert [(c["id"], c["fault"]) for c in cases] == [(1, "number"), (1, "agency")]
    assert all(c["mode"] == "rag" for c in cases)


def test_summary_counts_faults_on_accepted_answers():
    ok = {"declines": False, "supported": True, "reason": ""}
    bad = {"declines": False, "supported": False, "reason": "wrong number"}
    base = {"lang": "en", "answerable": True, "not_found": False}
    runs = {"rag": [{**base, "id": 1, "judge": ok}, {**base, "id": 2, "judge": bad}]}
    faults = [
        {**base, "id": 1, "mode": "rag", "fault": "number", "judge": bad},
        {**base, "id": 2, "mode": "rag", "fault": "number", "judge": bad},
    ]
    table = judge.summarize_judge(runs, faults)
    assert "| rag | 2 | 50% |" in table
    assert "| en | 100% of 1 | - of 0 |" in table
