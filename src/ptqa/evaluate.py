import json
import time

from ptqa.agent import agency_retrievers, run_agent
from ptqa.config import PLANNER_TOKENS, QUESTIONS, RESULTS, RETRIEVAL, TOP_K
from ptqa.llm import make_chat_model
from ptqa.multi_agent import run_multi_agent
from ptqa.rag import answer_closed_book, ask, is_not_found
from ptqa.store import Retriever, vector_store
from ptqa.text import detect_lang, load_pages, near_copy, split_sections

PIPELINES = ("rag", "agent", "multi")


def load_questions(path=QUESTIONS):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(questions, pipeline, sections_by_url, closed_book=None):
    records = []
    for q in questions:
        started = time.perf_counter()
        out = pipeline(q["question"])
        seconds = time.perf_counter() - started
        urls = list(dict.fromkeys(meta["url"] for meta, _ in out["hits"]))
        expected = set(q["expected_urls"])
        found = expected & set(urls)
        gold = [s for u in q["expected_urls"] for s in sections_by_url.get(u, [])]
        record = {
            **q,
            "near_copy": near_copy(q["question"], gold),
            "retrieved": urls,
            "sources": [text for _, text in out["hits"]],
            "rank": next((i for i, u in enumerate(urls, 1) if u in expected), None),
            "coverage": len(found) / len(expected) if expected else None,
            "answer": out["answer"],
            "cited": out["cited"],
            "not_found": is_not_found(out["answer"]),
            "answer_lang": detect_lang(out["answer"]),
            "cites_expected": bool(expected & set(out["cited"])),
            "cites_all_expected": bool(expected) and expected <= set(out["cited"]),
            "searches": out["searches"],
            "llm_calls": out["llm_calls"],
            "seconds": round(seconds, 2),
            "truncated": out.get("truncated", False),
        }
        if closed_book:
            record["closed_book"] = closed_book(q["question"])
            record["closed_book_not_found"] = is_not_found(record["closed_book"])
        records.append(record)
        print(f"{q['id']:>3} {q['lang']} rank={record['rank']} not_found={record['not_found']}")
    return records


def pct(n, d):
    return f"{n / d:.0%}" if d else "-"


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0


def summarize(records):
    single = [r for r in records if not r.get("multi")]
    groups = [(lang, [r for r in single if r["lang"] == lang]) for lang in ("tr", "en", "pt")]
    groups.append(("all", single))
    closed = "closed_book" in records[0]

    out = ["### Retrieval (single-page questions)", ""]
    out += ["| Language | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|---|"]
    for lang, rs in groups:
        a = [r for r in rs if r["answerable"]]
        hit = [sum(r["rank"] is not None and r["rank"] <= k for r in a) for k in (1, 3, 5)]
        mrr = mean(1 / r["rank"] if r["rank"] else 0 for r in a)
        cells = [pct(h, len(a)) for h in hit]
        out.append(f"| {lang} | {len(a)} | {' | '.join(cells)} | {mrr:.2f} |")

    out += ["", "### Answers (single-page questions)", ""]
    head = (
        '| Language | Answerable | Cites a right page | Wrong "not found" | Same language '
        '| Unanswerable | "Not found" with sources |'
    )
    rule = "|---|---|---|---|---|---|---|"
    if closed:
        head += ' "Not found" without sources |'
        rule += "---|"
    out += [head, rule]
    for lang, rs in groups:
        a = [r for r in rs if r["answerable"]]
        u = [r for r in rs if not r["answerable"]]
        same = sum(r["answer_lang"] == r["lang"] for r in rs)
        row = (
            f"| {lang} | {len(a)} | {pct(sum(r['cites_expected'] for r in a), len(a))} "
            f"| {pct(sum(r['not_found'] for r in a), len(a))} | {pct(same, len(rs))} "
            f"| {len(u)} | {pct(sum(r['not_found'] for r in u), len(u))} |"
        )
        if closed:
            row += f" {pct(sum(r['closed_book_not_found'] for r in u), len(u))} |"
        out.append(row)

    multi = [r for r in records if r.get("multi")]
    if multi:
        out += ["", "### Multi-page questions", ""]
        out += [
            "| Questions | Page coverage | All pages found | Cites all needed pages "
            "| Same language |",
            "|---|---|---|---|---|",
            f"| {len(multi)} | {mean(r['coverage'] for r in multi):.0%} "
            f"| {pct(sum(r['coverage'] == 1 for r in multi), len(multi))} "
            f"| {pct(sum(r['cites_all_expected'] for r in multi), len(multi))} "
            f"| {pct(sum(r['answer_lang'] == r['lang'] for r in multi), len(multi))} |",
        ]

    authors = sorted({r.get("author", "draft") for r in records})
    if len(authors) > 1:
        out += ["", "### Retrieval by question author", ""]
        out += ["| Author | Questions | Hit@1 | Hit@5 |", "|---|---|---|---|"]
        for author in authors:
            a = [r for r in single if r.get("author", "draft") == author and r["answerable"]]
            h1 = sum(r["rank"] == 1 for r in a)
            h5 = sum(r["rank"] is not None for r in a)
            out.append(f"| {author} | {len(a)} | {pct(h1, len(a))} | {pct(h5, len(a))} |")

    copies = sum(r["near_copy"] for r in records)
    cut = sum(r.get("truncated", False) for r in records)
    out += ["", f"Questions that reuse wording from their source page: {copies}"]
    out += [f"Answers cut at the token limit: {cut}"]
    return "\n".join(out)


def compare(runs):
    out = [
        "| Mode | Single-page Hit@5 | Multi-page coverage | All pages found "
        '| Cites all needed pages | "Not found" on unanswerable | Searches | LLM calls | Seconds |',
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, rs in runs.items():
        single = [r for r in rs if r["answerable"] and not r.get("multi")]
        multi = [r for r in rs if r.get("multi")]
        unans = [r for r in rs if not r["answerable"]]
        out.append(
            f"| {mode} | {pct(sum(r['rank'] is not None for r in single), len(single))} "
            f"| {mean(r['coverage'] for r in multi):.0%} "
            f"| {pct(sum(r['coverage'] == 1 for r in multi), len(multi))} "
            f"| {pct(sum(r['cites_all_expected'] for r in multi), len(multi))} "
            f"| {pct(sum(r['not_found'] for r in unans), len(unans))} "
            f"| {mean(max(1, len(r['searches'])) for r in rs):.1f} "
            f"| {mean(r['llm_calls'] for r in rs):.1f} | {mean(r['seconds'] for r in rs):.1f} |"
        )
    return "\n".join(out)


def make_pipeline(mode, retriever, llm, k):
    if mode == "rag":
        return (lambda q: ask(retriever, llm, q, k)), (lambda q: answer_closed_book(llm, q))
    planner = make_chat_model(max_tokens=PLANNER_TOKENS)
    if mode == "multi":
        return (lambda q: run_multi_agent(planner, llm, retriever, q)), None
    retrievers = agency_retrievers(retriever)
    return (lambda q: run_agent(planner, llm, retrievers, q)), None


def load_runs(modes=PIPELINES):
    runs = {}
    for m in modes:
        path = RESULTS / f"eval_{m}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                runs[m] = [json.loads(line) for line in f]
    return runs


def run_eval(mode="all", k=TOP_K, retrieval=RETRIEVAL):
    retriever, llm = Retriever(vector_store(), retrieval), make_chat_model()
    sections_by_url = {p["url"]: split_sections(p["text"]) for p in load_pages()}
    questions = load_questions()
    modes = list(PIPELINES) if mode == "all" else [mode]
    runs = {}
    for m in modes:
        print(f"\n{m}")
        pipeline, closed = make_pipeline(m, retriever, llm, k)
        runs[m] = evaluate(questions, pipeline, sections_by_url, closed)

    RESULTS.mkdir(exist_ok=True)
    for m, records in runs.items():
        with (RESULTS / f"eval_{m}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    runs = load_runs()
    report = []
    for m, records in runs.items():
        report += [f"## {m}", "", summarize(records), ""]
    if len(runs) > 1:
        report += ["## Comparison", "", compare(runs), ""]
    text = "\n".join(report)
    (RESULTS / "eval.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def hit_rate(ranks, n, questions):
    hits = sum(ranks[q["id"]] is not None and ranks[q["id"]] <= n for q in questions)
    return pct(hits, len(questions))


def retrieval_table(retriever, questions, modes, k=TOP_K):
    out = [
        "| Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR | Hit@1 tr | Hit@1 en | Hit@1 pt "
        "| Multi-page coverage | ms per query |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    print("\n".join(out), flush=True)
    answerable = [q for q in questions if q["answerable"]]
    for mode in modes:
        ranks, coverage, seconds = {}, [], []
        for q in answerable:
            started = time.perf_counter()
            hits = retriever.search(q["question"], k, mode=mode)
            seconds.append(time.perf_counter() - started)
            urls = list(dict.fromkeys(meta["url"] for meta, _ in hits))
            expected = set(q["expected_urls"])
            if q.get("multi"):
                coverage.append(len(expected & set(urls)) / len(expected))
            else:
                ranks[q["id"]] = next((i for i, u in enumerate(urls, 1) if u in expected), None)
        single = [q for q in answerable if not q.get("multi")]
        by_lang = [
            hit_rate(ranks, 1, [q for q in single if q["lang"] == lang])
            for lang in ("tr", "en", "pt")
        ]
        mrr = mean(1 / ranks[q["id"]] if ranks[q["id"]] else 0 for q in single)
        out.append(
            f"| {mode} | {hit_rate(ranks, 1, single)} | {hit_rate(ranks, 3, single)} "
            f"| {hit_rate(ranks, 5, single)} | {mrr:.2f} "
            f"| {' | '.join(by_lang)} | {mean(coverage):.0%} | {1000 * mean(seconds):.0f} |"
        )
        print(out[-1], flush=True)
    return "\n".join(out)


def run_retrieval(modes, k=TOP_K):
    retriever = Retriever(vector_store())
    table = retrieval_table(retriever, load_questions(), modes, k)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.md").write_text(table + "\n", encoding="utf-8")
