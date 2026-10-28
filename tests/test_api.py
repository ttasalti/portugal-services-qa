from fastapi.testclient import TestClient

from ptqa import api


def fake(question):
    hits = [({"url": "u1", "title": "NIF"}, "a"), ({"url": "u2", "title": "AIMA"}, "b")]
    return {
        "hits": hits,
        "answer": "Ask online [2], see also [9].",
        "cited": ["u2"],
        "searches": [("tax", "nif")],
        "llm_calls": 2,
    }


def test_ask_returns_answer_with_references():
    api.app.state.run = {"rag": fake, "agent": fake, "multi": fake}
    client = TestClient(api.app)
    resp = client.post("/ask", json={"question": "How do I get a NIF?", "mode": "agent"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["references"] == {"2": {"url": "u2", "title": "AIMA"}}
    assert body["sources"] == [{"url": "u2", "title": "AIMA"}]
    assert body["searches"] == [["tax", "nif"]]
    assert body["llm_calls"] == 2


def test_rejects_unknown_mode_and_serves_page():
    client = TestClient(api.app)
    assert client.post("/ask", json={"question": "NIF?", "mode": "x"}).status_code == 422
    page = client.get("/")
    assert page.status_code == 200 and "Portugal services QA" in page.text
