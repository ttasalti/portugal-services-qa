from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph

from ptqa.config import AGENCIES, AGENT_K, MAX_SEARCHES
from ptqa.rag import answer, cited_urls, is_not_found

PLANNER_PROMPT = f"""You plan searches for questions from immigrants in Portugal about residence
permits, taxes and social security. You do not answer the question yourself.

Tools:
- search_aima: residence permits, visas, renewals, family reunification (AIMA)
- search_tax: NIF, opening activity, tax representative, IRS, VAT (Financas)
- search_social: NISS, contributions, quarterly declaration, unemployment benefit, SNS

Rules:
- Questions often involve more than one agency. Search every agency the question touches.
- Write short search queries in Portuguese.
- Read what the searches return. If a step of the question is still not covered, search again.
- You have at most {MAX_SEARCHES} searches.
- When the sources cover the question, or nothing more can be found, stop calling tools and
  reply with DONE."""

TOOL_DESCRIPTIONS = {
    "aima": "Search AIMA pages: residence permits, visas, renewals, family reunification.",
    "tax": "Search Financas pages: NIF, opening activity, tax representative, IRS, VAT.",
    "social": "Search Seguranca Social and SNS pages: NISS, contributions, quarterly "
    "declaration, unemployment benefit, SNS number.",
}


class State(MessagesState):
    answer: str


class SearchSession:
    def __init__(self, retrievers, max_searches=MAX_SEARCHES):
        self.retrievers = retrievers
        self.max_searches = max_searches
        self.hits = []
        self.queries = []

    def search(self, agency, query):
        if len(self.queries) >= self.max_searches:
            return "Search limit reached. Reply with DONE."
        self.queries.append((agency, query))
        lines = []
        for meta, text in self.retrievers[agency](query):
            key = (meta["url"], text)
            known = [i for i, (m, t) in enumerate(self.hits, 1) if (m["url"], t) == key]
            if known:
                lines.append(f"[{known[0]}] already listed above")
                continue
            self.hits.append((meta, text))
            lines.append(f"[{len(self.hits)}] {text}")
        return "\n\n".join(lines) or "No results."


def agency_retrievers(retriever, k=AGENT_K):
    return {a: lambda q, a=a: retriever.search(q, k, a) for a in AGENCIES}


def search_tools(session):
    def make(agency):
        def search(query: str) -> str:
            return session.search(agency, query)

        return StructuredTool.from_function(
            search, name=f"search_{agency}", description=TOOL_DESCRIPTIONS[agency]
        )

    return [make(a) for a in AGENCIES]


def build_graph(plan, search, respond, route):
    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("respond", respond)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route, ["search", "respond"])
    graph.add_edge("search", "plan")
    graph.add_edge("respond", END)
    return graph.compile()


def run_agent(chat_model, llm, retrievers, question):
    session = SearchSession(retrievers)
    tools = {t.name: t for t in search_tools(session)}
    planner = chat_model.bind_tools(list(tools.values()))
    llm_calls = 0
    truncated = False

    def plan(state):
        nonlocal llm_calls
        llm_calls += 1
        return {"messages": [planner.invoke(state["messages"])]}

    def search(state):
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = tools.get(call["name"])
            text = tool.invoke(call["args"]) if tool else f"Unknown tool {call['name']}."
            results.append(ToolMessage(content=text, tool_call_id=call["id"]))
        return {"messages": results}

    def respond(state):
        nonlocal llm_calls, truncated
        llm_calls += 1
        text, _, truncated = answer(llm, question, session.hits)
        return {"answer": text}

    def route(state):
        wants_tools = bool(getattr(state["messages"][-1], "tool_calls", None))
        return "search" if wants_tools and len(session.queries) < MAX_SEARCHES else "respond"

    graph = build_graph(plan, search, respond, route)
    start = {"messages": [SystemMessage(PLANNER_PROMPT), HumanMessage(question)]}
    result = graph.invoke(start, {"recursion_limit": 4 * MAX_SEARCHES + 4})
    text = result["answer"]
    return {
        "hits": session.hits,
        "answer": text,
        "cited": [] if is_not_found(text) else cited_urls(text, session.hits),
        "searches": session.queries,
        "llm_calls": llm_calls,
        "truncated": truncated,
    }


def mermaid():
    def noop(state):
        return {}

    return build_graph(noop, noop, noop, lambda state: "respond").get_graph().draw_mermaid()
