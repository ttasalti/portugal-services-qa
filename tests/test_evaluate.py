from ptqa import evaluate, rag


def test_evaluate_and_summarize():
    questions = [
        {"id": 1, "lang": "en", "question": "q1", "answerable": True, "expected_urls": ["a"]},
        {"id": 2, "lang": "en", "question": "q2", "answerable": False, "expected_urls": []},
    ]
    hits = [({"url": u, "title": "t", "section": "s"}, "text") for u in ("b", "a")]
    replies = {"q1": "You need a passport [2].", "q2": rag.NOT_FOUND["en"]}

    def pipeline(question):
        text = replies[question]
        return {
            "hits": hits,
            "answer": text,
            "cited": rag.cited_urls(text, hits),
            "searches": [],
            "llm_calls": 1,
        }

    records = evaluate.evaluate(questions, pipeline, {}, lambda q: rag.NOT_FOUND["en"])
    first, second = records
    assert first["rank"] == 2 and first["cites_expected"] and not first["not_found"]
    assert second["not_found"] and second["closed_book_not_found"]

    table = evaluate.summarize(records)
    assert "| en | 1 | 0% | 100% | 100% | 0.50 |" in table
    assert "| en | 1 | 100% | 0% | 100% | 1 | 100% | 100% |" in table


def test_multi_page_coverage_and_compare():
    question = {
        "id": 1,
        "lang": "en",
        "question": "q",
        "answerable": True,
        "multi": True,
        "expected_urls": ["tax", "aima"],
    }

    def pipeline(found):
        def run(q):
            hits = [({"url": u, "title": "t", "section": "s"}, u) for u in found]
            return {
                "hits": hits,
                "answer": "Done [1][2].",
                "cited": found,
                "searches": [("tax", "x")] * len(found),
                "llm_calls": len(found) + 1,
            }

        return run

    rag_run = evaluate.evaluate([question], pipeline(["tax"]), {})
    agent_run = evaluate.evaluate([question], pipeline(["tax", "aima"]), {})
    assert rag_run[0]["coverage"] == 0.5 and not rag_run[0]["cites_all_expected"]
    assert agent_run[0]["coverage"] == 1 and agent_run[0]["cites_all_expected"]
    table = evaluate.compare({"rag": rag_run, "agent": agent_run})
    assert "| rag | - | 50% | 0% | 0% | - | 1.0 | 2.0 |" in table
    assert "| agent | - | 100% | 100% | 100% | - | 2.0 | 3.0 |" in table
