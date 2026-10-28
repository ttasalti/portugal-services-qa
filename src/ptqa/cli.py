import argparse

from ptqa.config import PLANNER_TOKENS, RETRIEVAL, TOP_K

MODES = ("dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank")


def retriever(args):
    from ptqa.store import Retriever, vector_store

    return Retriever(vector_store(), args.retrieval)


def cmd_fetch(args):
    from ptqa.fetch import fetch

    fetch(args.refresh)


def cmd_index(args):
    from ptqa.store import build_index

    build_index(args.chunk_size)


def cmd_search(args):
    r = retriever(args)
    for q in args.question:
        print(f"\n> {q}")
        for i, (meta, _) in enumerate(r.search(q, args.k), 1):
            print(f"  {i}. {meta['title'][:50]} / {meta['section'][:30]}")


def cmd_ask(args):
    from ptqa.llm import make_chat_model
    from ptqa.rag import ask

    r, llm = retriever(args), make_chat_model()
    for q in args.question:
        out = ask(r, llm, q, args.k)
        print(f"\n> {q}\n{out['answer']}")
        for url in out["cited"]:
            print(f"  {url}")


def cmd_agent(args):
    from ptqa.agent import agency_retrievers, run_agent
    from ptqa.llm import make_chat_model

    r = retriever(args)
    planner, llm = make_chat_model(max_tokens=PLANNER_TOKENS), make_chat_model()
    for q in args.question:
        out = run_agent(planner, llm, agency_retrievers(r), q)
        print(f"\n> {q}")
        for agency, query in out["searches"]:
            print(f"  search_{agency}: {query}")
        print(out["answer"])
        for url in out["cited"]:
            print(f"  {url}")


def cmd_multi(args):
    from ptqa.llm import make_chat_model
    from ptqa.multi_agent import run_multi_agent

    r = retriever(args)
    planner, llm = make_chat_model(max_tokens=PLANNER_TOKENS), make_chat_model()
    for q in args.question:
        out = run_multi_agent(planner, llm, r, q)
        print(f"\n> {q}")
        for agency, query in out["searches"][::2]:
            print(f"  {agency}: {query}")
        print(out["answer"])
        for url in out["cited"]:
            print(f"  {url}")


def cmd_eval(args):
    from ptqa.evaluate import run_eval

    run_eval(args.mode, args.k, args.retrieval)


def cmd_retrieval(args):
    from ptqa.evaluate import run_retrieval

    run_retrieval(args.modes, args.k)


def cmd_judge(args):
    from ptqa.judge import run_judge
    from ptqa.llm import make_chat_model

    run_judge(make_chat_model(max_tokens=512), args.modes)


def cmd_serve(args):
    import uvicorn

    uvicorn.run("ptqa.api:app", host=args.host, port=args.port)


def cmd_graph(args):
    from ptqa import agent, multi_agent

    print((multi_agent if args.multi else agent).mermaid())


def main():
    parser = argparse.ArgumentParser(prog="ptqa")
    sub = parser.add_subparsers(required=True)

    def command(name, func, help, question=False, k=False, retrieval=False):
        p = sub.add_parser(name, help=help)
        if question:
            p.add_argument("question", nargs="+")
        if k:
            p.add_argument("-k", type=int, default=TOP_K)
        if retrieval:
            p.add_argument("--retrieval", choices=MODES, default=RETRIEVAL)
        p.set_defaults(func=func)
        return p

    p = command("fetch", cmd_fetch, "download sources and extract text")
    p.add_argument("--refresh", action="store_true", help="download again even if cached")
    p = command("index", cmd_index, "chunk pages and build the vector index")
    p.add_argument("--chunk-size", type=int, default=1500, help="characters per chunk")
    command("search", cmd_search, "show retrieved chunks", question=True, k=True, retrieval=True)
    command("ask", cmd_ask, "answer with sources", question=True, k=True, retrieval=True)
    command("agent", cmd_agent, "answer with the search agent", question=True, retrieval=True)
    command("multi", cmd_multi, "answer with one agent per agency", question=True, retrieval=True)
    p = command("eval", cmd_eval, "run the question set and write results/", k=True, retrieval=True)
    p.add_argument("--mode", choices=["rag", "agent", "multi", "all"], default="all")
    p = command("retrieval", cmd_retrieval, "compare retrieval modes without an LLM", k=True)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p = command("judge", cmd_judge, "check saved answers and injected faults with a judge model")
    p.add_argument("--modes", nargs="+", default=["rag", "agent", "multi"])
    p = command("serve", cmd_serve, "run the web API and page")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = command("graph", cmd_graph, "print an agent graph as Mermaid")
    p.add_argument("--multi", action="store_true", help="the multi-agent graph")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
