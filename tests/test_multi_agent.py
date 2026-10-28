from langchain_core.runnables import RunnableLambda

from ptqa import multi_agent
from ptqa.multi_agent import Plan, Task
from ptqa.rag import NOT_FOUND


class FakeRetriever:
    def search(self, question, k=5, agency=None, mode=None):
        return [({"url": f"{agency}/{i}"}, f"{agency} text {i}") for i in (1, 2)][:k]


class FakeChatModel:
    def __init__(self, plan):
        self.plan = plan

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda messages: self.plan)


def fake_llm(replies):
    def reply(prompt):
        text = prompt.to_string()
        if "Partial answers" in text:
            return "Merged [1] and [3]."
        return next(r for key, r in replies.items() if key in text)

    return RunnableLambda(reply)


def test_renumber_shifts_and_drops_bad_numbers():
    assert multi_agent.renumber("A [1][2], B [5].", 4, 2) == "A [5][6], B ."


def test_specialists_run_per_agency_and_merge():
    plan = Plan(tasks=[Task(agency="tax", query="nif"), Task(agency="aima", query="visto")])
    llm = fake_llm({"aima text": "Visa [2].", "tax text": "NIF [1]."})
    out = multi_agent.run_multi_agent(FakeChatModel(plan), llm, FakeRetriever(), "q?")
    assert out["answer"] == "Merged [1] and [3]."
    assert out["cited"] == ["aima/1", "tax/1"]
    assert [a for a, _ in out["searches"]] == ["aima", "aima", "tax", "tax"]
    assert out["llm_calls"] == 4


def test_single_useful_part_skips_merge():
    plan = Plan(tasks=[Task(agency="tax", query="nif"), Task(agency="aima", query="visto")])
    llm = fake_llm({"aima text": NOT_FOUND["en"], "tax text": "NIF [2]."})
    out = multi_agent.run_multi_agent(FakeChatModel(plan), llm, FakeRetriever(), "How?")
    assert out["answer"] == "NIF [4]."
    assert out["cited"] == ["tax/2"]
    assert out["llm_calls"] == 3


def test_all_parts_missing_gives_not_found_in_question_language():
    plan = Plan(tasks=[Task(agency="social", query="niss")])
    llm = fake_llm({"social text": NOT_FOUND["tr"]})
    question = "Bu belge için ne gerekiyor ve nasıl başvurulur?"
    out = multi_agent.run_multi_agent(FakeChatModel(plan), llm, FakeRetriever(), question)
    assert out["answer"] == NOT_FOUND["tr"]
    assert out["cited"] == []


def test_failed_plan_falls_back_to_every_agency():
    class Broken:
        def invoke(self, messages):
            raise ValueError("bad json")

    tasks = multi_agent.plan_tasks(Broken(), "q")
    assert [t.agency for t in tasks] == ["aima", "tax", "social"]


def test_specialist_answers_in_question_language():
    seen = []

    def reply(prompt):
        seen.append(prompt.to_string())
        return "NIF [1]."

    plan = Plan(tasks=[Task(agency="tax", query="pedir o número de identificação fiscal")])
    question = "Bu belge için ne gerekiyor ve nasıl başvurulur?"
    multi_agent.run_multi_agent(
        FakeChatModel(plan), RunnableLambda(reply), FakeRetriever(), question
    )
    assert seen[0].endswith("Answer in Turkish.")
