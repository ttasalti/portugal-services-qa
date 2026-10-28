from ptqa import llm, rag, store


def test_citations_map_to_urls_and_ignore_bad_numbers():
    hits = [({"url": "a"}, ""), ({"url": "b"}, "")]
    assert rag.cited_urls("Yes [2][1], see also [2] and [7].", hits) == ["b", "a"]
    assert rag.is_not_found("As fontes não respondem a esta pergunta.")
    assert not rag.is_not_found("Precisa do passaporte [1].")
    assert llm.clean_answer("<think>\n</think>\nOk [1].") == "Ok [1]."


def test_chunks_carry_heading_and_agency_flags():
    page = {
        "url": "u",
        "title": "NIF",
        "agencies": ["tax", "social"],
        "text": "## Quem\n" + "Qualquer pessoa pode pedir. " * 80,
    }
    docs = store.chunk_documents([page], chunk_size=500)
    assert len(docs) > 1
    assert all(d.page_content.startswith("NIF / Quem\n\n") for d in docs)
    assert docs[0].metadata == {
        "title": "NIF",
        "section": "Quem",
        "url": "u",
        "aima": False,
        "tax": True,
        "social": True,
    }


def make_retriever(pages, mode):
    import chromadb
    from langchain_chroma import Chroma
    from langchain_core.embeddings import DeterministicFakeEmbedding

    db = Chroma(
        client=chromadb.EphemeralClient(),
        collection_name=f"test-{mode}",
        embedding_function=DeterministicFakeEmbedding(size=16),
    )
    db.add_documents(store.chunk_documents(pages))
    return store.Retriever(db, mode)


PAGES = [
    {"url": "a", "title": "A", "agencies": ["aima"], "text": "## s\nautorização art. 88"},
    {"url": "b", "title": "B", "agencies": ["tax", "social"], "text": "## s\nNIF e NISS"},
    {"url": "c", "title": "C", "agencies": ["social"], "text": "## s\nsubsídio de desemprego"},
]


def test_agency_filter_in_every_mode():
    for mode in ("dense", "bm25", "hybrid"):
        r = make_retriever(PAGES, mode)
        assert [m["url"] for m, _ in r.search("x", k=5, agency="aima")] == ["a"]
        assert {m["url"] for m, _ in r.search("x", k=5, agency="social")} == {"b", "c"}
        assert len(r.search("x", k=5)) == 3


def test_bm25_matches_exact_terms():
    r = make_retriever(PAGES, "bm25")
    assert r.search("Artigo 88", k=1)[0][0]["url"] == "a"
    assert r.search("Como peço o NISS?", k=1)[0][0]["url"] == "b"


def test_rrf_rewards_agreement():
    a, b, c = [({"url": u}, u) for u in "abc"]
    assert store.rrf([a, b, c], [b, c, a]) == [b, a, c]
    assert store.rrf([a], [c]) == [a, c]


def test_reply_reports_token_limit():
    from langchain_core.messages import AIMessage

    cut = AIMessage("Passaporte e", response_metadata={"finish_reason": "length"})
    done = AIMessage(
        "<think>\n</think>\nPassaporte [1].", response_metadata={"finish_reason": "stop"}
    )
    assert llm.read_reply(cut) == ("Passaporte e", True)
    assert llm.read_reply(done) == ("Passaporte [1].", False)
    assert llm.read_reply("plain text") == ("plain text", False)
