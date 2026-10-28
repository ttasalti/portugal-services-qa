import re
import time
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ptqa.agent import agency_retrievers, run_agent
from ptqa.config import PLANNER_TOKENS, RETRIEVAL
from ptqa.llm import make_chat_model
from ptqa.multi_agent import run_multi_agent
from ptqa.rag import ask
from ptqa.store import Retriever, vector_store


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["rag", "agent", "multi"] = "rag"


class Source(BaseModel):
    url: str
    title: str


class Answer(BaseModel):
    answer: str
    references: dict[int, Source]
    sources: list[Source]
    searches: list[tuple[str, str]]
    llm_calls: int
    seconds: float


def pipelines(retriever, planner, llm):
    retrievers = agency_retrievers(retriever)
    return {
        "rag": lambda q: ask(retriever, llm, q),
        "agent": lambda q: run_agent(planner, llm, retrievers, q),
        "multi": lambda q: run_multi_agent(planner, llm, retriever, q),
    }


@asynccontextmanager
async def lifespan(app):
    retriever = Retriever(vector_store(), RETRIEVAL)
    retriever.search("NIF", 1)
    app.state.run = pipelines(
        retriever, make_chat_model(max_tokens=PLANNER_TOKENS), make_chat_model()
    )
    yield


app = FastAPI(title="Portugal services QA", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return files("ptqa").joinpath("static/index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
async def answer(q: Question):
    started = time.perf_counter()
    out = await run_in_threadpool(app.state.run[q.mode], q.question)
    hits = out["hits"]
    refs = {}
    for n in map(int, re.findall(r"\[(\d+)\]", out["answer"])):
        if 1 <= n <= len(hits) and out["cited"]:
            refs[n] = Source(url=hits[n - 1][0]["url"], title=hits[n - 1][0]["title"])
    titles = {meta["url"]: meta["title"] for meta, _ in hits}
    return Answer(
        answer=out["answer"],
        references=refs,
        sources=[Source(url=url, title=titles[url]) for url in out["cited"]],
        searches=out["searches"],
        llm_calls=out["llm_calls"],
        seconds=round(time.perf_counter() - started, 2),
    )
