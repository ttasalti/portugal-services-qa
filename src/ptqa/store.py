import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from ptqa.config import (
    AGENCIES,
    CANDIDATES,
    CHROMA,
    COLLECTION,
    EMBED_MODEL,
    RERANK_MODEL,
    RRF_K,
    TOP_K,
)
from ptqa.text import load_pages, split_sections, tokens


def chunk_documents(pages, chunk_size=1500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 8)
    docs = []
    for page in pages:
        flags = {a: a in page.get("agencies", []) for a in AGENCIES}
        for head, body in split_sections(page["text"]):
            label = f"{page['title']} / {head}" if head else page["title"]
            meta = {"title": page["title"], "section": head, "url": page["url"], **flags}
            for piece in splitter.split_text(body):
                docs.append(Document(page_content=f"{label}\n\n{piece}", metadata=meta))
    return docs


def embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
    )


def vector_store():
    client = chromadb.PersistentClient(
        path=str(CHROMA), settings=Settings(anonymized_telemetry=False)
    )
    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def rrf(*rankings, k=RRF_K):
    scores, hits = {}, {}
    for ranking in rankings:
        for rank, (meta, text) in enumerate(ranking, 1):
            key = (meta["url"], text)
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            hits[key] = (meta, text)
    return [hits[key] for key in sorted(scores, key=scores.get, reverse=True)]


class Retriever:
    def __init__(self, store, mode="hybrid"):
        self.store = store
        self.mode = mode
        data = store.get(include=["documents", "metadatas"])
        self.chunks = list(zip(data["metadatas"], data["documents"], strict=True))
        self.keyword_index = BM25Okapi([tokens(text) for _, text in self.chunks])
        self.reranker = None

    def dense(self, question, k, agency=None):
        docs = self.store.similarity_search(
            question, k=k, filter={agency: True} if agency else None
        )
        return [(d.metadata, d.page_content) for d in docs]

    def bm25(self, question, k, agency=None):
        scores = self.keyword_index.get_scores(tokens(question))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        hits = [self.chunks[i] for i in order if not agency or self.chunks[i][0][agency]]
        return hits[:k]

    def hybrid(self, question, k, agency=None):
        dense = self.dense(question, CANDIDATES, agency)
        keyword = self.bm25(question, CANDIDATES, agency)
        return rrf(dense, keyword)[:k]

    def rerank(self, question, hits, k):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(RERANK_MODEL)
        scores = self.reranker.predict([(question, text) for _, text in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [hits[i] for i in order[:k]]

    def search(self, question, k=TOP_K, agency=None, mode=None):
        base, _, rerank = (mode or self.mode).partition("+")
        if not rerank:
            return getattr(self, base)(question, k, agency)
        return self.rerank(question, getattr(self, base)(question, CANDIDATES, agency), k)


def build_index(chunk_size=1500):
    docs = chunk_documents(load_pages(), chunk_size)
    store = vector_store()
    store.reset_collection()
    store.add_documents(docs)
    print(f"{len(docs)} chunks -> {CHROMA}")
