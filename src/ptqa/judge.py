import json
import random
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ptqa.config import RESULTS
from ptqa.evaluate import load_runs, pct
from ptqa.rag import format_sources

JUDGE_PROMPT = """You check answers written from numbered sources about public services in
Portugal. The answer may be in Turkish, English or Portuguese; the sources are in Portuguese.

Decide two things:
- declines: true if the answer only says that the sources do not answer the question.
- supported: true if every fact in the answer (documents, steps, deadlines, amounts, numbers,
  agency names) is stated in the sources. A wrong number, a wrong agency or a fact that is not in
  the sources makes it false. Wording and translation differences are fine. An answer that
  declines counts as supported.

Give a one-sentence reason that names the unsupported fact, if there is one."""

AGENCY_NAMES = {
    "AIMA": "Segurança Social",
    "Segurança Social": "Finanças",
    "Seguranca Social": "Finanças",
    "Finanças": "AIMA",
    "Autoridade Tributária": "AIMA",
    "SNS": "AIMA",
}
NUMBER = re.compile(r"(?<![\[\d.,])\d+(?:[.,]\d+)?(?![\]\d])")


class Verdict(BaseModel):
    declines: bool
    supported: bool
    reason: str


def change_number(text, rng):
    spots = list(NUMBER.finditer(text))
    if not spots:
        return None
    m = rng.choice(spots)
    n = int(re.sub(r"\D", "", m.group()))
    new = n + 1 if 1900 <= n <= 2100 else n * 2 or 3
    return f"{text[: m.start()]}{new}{text[m.end() :]}"


def change_agency(text, rng):
    names = [name for name in AGENCY_NAMES if name in text]
    if not names:
        return None
    name = rng.choice(names)
    return text.replace(name, AGENCY_NAMES[name])


FAULTS = {"number": change_number, "agency": change_agency}


def inject_faults(runs, seed=0):
    rng = random.Random(seed)
    cases = []
    for mode, records in runs.items():
        for r in records:
            if r["not_found"] or not r["answerable"]:
                continue
            for kind, fault in FAULTS.items():
                wrong = fault(r["answer"], rng)
                if wrong and wrong != r["answer"]:
                    cases.append({**r, "mode": mode, "fault": kind, "answer": wrong})
    return cases


def judge_one(judge, record):
    sources = format_sources([(None, text) for text in record["sources"]])
    message = (
        f"Sources:\n\n{sources}\n\nQuestion: {record['question']}\n\nAnswer:\n{record['answer']}"
    )
    try:
        return judge.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(message)]).model_dump()
    except Exception as e:
        return {"declines": None, "supported": None, "reason": f"judge error: {e}"}


def summarize_judge(runs, faults):
    out = ["### Answers checked by the judge", ""]
    out += [
        "| Mode | Answers | Supported | Declines on unanswerable | Declines on answerable "
        '| Strict "not found" on unanswerable |',
        "|---|---|---|---|---|---|",
    ]
    for mode, rs in runs.items():
        ok = [r for r in rs if r["judge"]["supported"] is not None]
        unans = [r for r in ok if not r["answerable"]]
        ans = [r for r in ok if r["answerable"]]
        out.append(
            f"| {mode} | {len(ok)} | {pct(sum(r['judge']['supported'] for r in ok), len(ok))} "
            f"| {pct(sum(r['judge']['declines'] for r in unans), len(unans))} "
            f"| {pct(sum(r['judge']['declines'] for r in ans), len(ans))} "
            f"| {pct(sum(r['not_found'] for r in unans), len(unans))} |"
        )

    # a fault only counts when the judge accepted the answer before it was changed
    accepted = {(m, r["id"]) for m, rs in runs.items() for r in rs if r["judge"]["supported"]}
    cases = [c for c in faults if (c["mode"], c["id"]) in accepted]
    out += ["", "### Judge on injected faults", ""]
    out += ["| Language | Number changed | Agency changed |", "|---|---|---|"]
    for lang in ("tr", "en", "pt", "all"):
        cells = []
        for kind in FAULTS:
            cs = [c for c in cases if c["fault"] == kind and lang in (c["lang"], "all")]
            caught = sum(c["judge"]["supported"] is False for c in cs)
            cells.append(f"{pct(caught, len(cs))} of {len(cs)}")
        out.append(f"| {lang} | {' | '.join(cells)} |")
    judged = [r for rs in runs.values() for r in rs] + faults
    errors = sum(r["judge"]["supported"] is None for r in judged)
    out += ["", f"Judge calls: {len(judged)}, failed: {errors}"]
    return "\n".join(out)


def run_judge(chat_model, modes):
    judge = chat_model.with_structured_output(Verdict, method="json_schema")
    runs = load_runs(modes)
    faults = inject_faults(runs)
    for r in [r for rs in runs.values() for r in rs] + faults:
        r["judge"] = judge_one(judge, r)
        print(f"{r['id']:>3} {r.get('fault', 'clean'):6} supported={r['judge']['supported']}")

    with (RESULTS / "judge.jsonl").open("w", encoding="utf-8") as f:
        for mode, rs in runs.items():
            for r in rs:
                f.write(json.dumps({"mode": mode, **r}, ensure_ascii=False) + "\n")
        for c in faults:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    text = summarize_judge(runs, faults)
    (RESULTS / "judge.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
