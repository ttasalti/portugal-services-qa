import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from ptqa.config import AGENCIES, SPECIALIST_K
from ptqa.llm import read_reply
from ptqa.rag import NOT_FOUND, answer, cited_urls, is_not_found, language_line
from ptqa.store import rrf
from ptqa.text import detect_lang

SUPERVISOR_PROMPT = """You route questions from immigrants in Portugal to the agencies that can
answer them. You do not answer the question yourself.

Agencies:
- aima: residence permits, visas, renewals, family reunification
- tax: NIF, opening activity, tax representative, IRS, VAT (Financas)
- social: NISS, contributions, quarterly declaration, unemployment benefit, SNS number

Return only the agencies the question needs. For each one, write one short search query in
European Portuguese that covers the part of the question that agency handles."""

MERGE_PROMPT = """You combine partial answers from different agencies into one answer for an
immigrant in Portugal.

Rules:
- Use only the partial answers. Do not add facts.
- Keep every citation number exactly as it is, for example [3] or [4][7].
- Leave out partial answers that say the sources do not answer.
- Be short and concrete, and follow the order of the question.
- Write in the language named on the last line, even when the partial answers use another one.
- Write Portuguese in European Portuguese as used in Portugal, not Brazilian Portuguese."""


class Task(BaseModel):
    agency: Literal["aima", "tax", "social"]
    query: str = Field(description="short search query in European Portuguese")


class Plan(BaseModel):
    tasks: list[Task]


class State(TypedDict):
    question: str
    tasks: list[Task]
    partials: Annotated[list[dict], operator.add]
    result: dict


class SpecialistState(TypedDict):
    question: str
    task: Task


def plan_tasks(router, question):
    try:
        plan = router.invoke([SystemMessage(SUPERVISOR_PROMPT), HumanMessage(question)])
        tasks = {t.agency: t for t in plan.tasks}
    except Exception:
        tasks = {}
    if not tasks:
        return [Task(agency=a, query=question) for a in AGENCIES]
    return [tasks[a] for a in AGENCIES if a in tasks]


def renumber(text, offset, size):
    def shift(m):
        n = int(m.group(1))
        return f"[{n + offset}]" if 1 <= n <= size else ""

    return re.sub(r"\[(\d+)\]", shift, text)


def merge_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MERGE_PROMPT),
            ("human", "Question: {question}\n\nPartial answers:\n\n{partials}\n\n{language}"),
        ]
    )
    return prompt | llm


def build_graph(supervise, specialist, merge):
    def fan_out(state):
        return [
            Send("specialist", {"question": state["question"], "task": t}) for t in state["tasks"]
        ]

    graph = StateGraph(State)
    graph.add_node("supervise", supervise)
    graph.add_node("specialist", specialist)
    graph.add_node("merge", merge)
    graph.add_edge(START, "supervise")
    graph.add_conditional_edges("supervise", fan_out, ["specialist"])
    graph.add_edge("specialist", "merge")
    graph.add_edge("merge", END)
    return graph.compile()


def run_multi_agent(chat_model, llm, retriever, question, k=SPECIALIST_K):
    router = chat_model.with_structured_output(Plan, method="json_schema")

    def supervise(state):
        return {"tasks": plan_tasks(router, state["question"])}

    def specialist(state):
        task = state["task"]
        hits = rrf(
            retriever.search(task.query, k, task.agency),
            retriever.search(state["question"], k, task.agency),
        )[:k]
        focus = f"Answer only the part handled by this agency: {task.query}\n"
        text, _, truncated = answer(llm, state["question"], hits, focus)
        partial = {"agency": task.agency, "query": task.query, "hits": hits, "text": text}
        return {"partials": [{**partial, "truncated": truncated}]}

    def merge(state):
        partials = sorted(state["partials"], key=lambda p: AGENCIES.index(p["agency"]))
        return {"result": combine(llm, state["question"], partials)}

    graph = build_graph(supervise, specialist, merge)
    return graph.invoke({"question": question, "partials": []})["result"]


def combine(llm, question, partials):
    hits, useful = [], []
    truncated = any(p["truncated"] for p in partials)
    for p in partials:
        text = renumber(p["text"], len(hits), len(p["hits"]))
        hits += p["hits"]
        if not is_not_found(p["text"]):
            useful.append(text)
    if not useful:
        text = NOT_FOUND[detect_lang(question)]
    elif len(useful) == 1:
        text = useful[0]
    else:
        reply = merge_chain(llm).invoke(
            {
                "question": question,
                "partials": "\n\n".join(useful),
                "language": language_line(question),
            }
        )
        text, cut = read_reply(reply)
        truncated = truncated or cut
    return {
        "hits": hits,
        "answer": text,
        "cited": [] if is_not_found(text) else cited_urls(text, hits),
        "searches": [(p["agency"], q) for p in partials for q in (p["query"], question)],
        "llm_calls": 1 + len(partials) + (len(useful) > 1),
        "truncated": truncated,
    }


def mermaid():
    def noop(state):
        return {}

    return build_graph(noop, noop, noop).get_graph().draw_mermaid()
