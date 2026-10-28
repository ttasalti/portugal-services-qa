from ptqa import agent
from ptqa.config import AGENCIES


def test_search_session_numbers_hits_across_searches():
    def fake(agency):
        return lambda q: [({"url": f"{agency}/1"}, f"T / S\n\n{agency} text")]

    session = agent.SearchSession({a: fake(a) for a in AGENCIES}, max_searches=2)
    assert session.search("tax", "nif") == "[1] T / S\n\ntax text"
    assert session.search("aima", "visto") == "[2] T / S\n\naima text"
    assert "limit" in session.search("social", "niss")
    assert [m["url"] for m, _ in session.hits] == ["tax/1", "aima/1"]

    again = agent.SearchSession({"tax": fake("tax")})
    again.search("tax", "nif")
    assert again.search("tax", "nif") == "[1] already listed above"
    assert len(again.hits) == 1


def test_graph_loops_between_plan_and_search():
    chart = agent.mermaid()
    assert "__start__ --> plan;" in chart
    assert "search --> plan;" in chart
    assert "respond --> __end__;" in chart
