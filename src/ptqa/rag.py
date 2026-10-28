import re

from langchain_core.prompts import ChatPromptTemplate

from ptqa.config import TOP_K
from ptqa.llm import read_reply
from ptqa.text import LANG_NAMES, detect_lang

NOT_FOUND = {
    "tr": "Kaynaklarda bu sorunun cevabı yok.",
    "en": "The sources do not answer this question.",
    "pt": "As fontes não respondem a esta pergunta.",
}

SYSTEM_PROMPT = f"""You answer questions from immigrants in Portugal about residence permits,
taxes and social security. Use only the numbered sources you are given.

Rules:
- Answer in the language of the question (Turkish, English or Portuguese).
- Write Portuguese in European Portuguese as used in Portugal, not Brazilian Portuguese.
- Be short and concrete: list documents, steps, deadlines or amounts when the sources give them.
- Cite the sources you used with their numbers, for example [1] or [2][3].
- Do not add facts that are not in the sources.
- If the sources do not contain the answer, reply with exactly one of these sentences and nothing
  else, in the language of the question:
  Turkish: {NOT_FOUND["tr"]}
  English: {NOT_FOUND["en"]}
  Portuguese: {NOT_FOUND["pt"]}"""

CLOSED_BOOK_PROMPT = f"""You answer questions from immigrants in Portugal about residence permits,
taxes and social security.

Rules:
- Answer in the language of the question (Turkish, English or Portuguese).
- Write Portuguese in European Portuguese as used in Portugal, not Brazilian Portuguese.
- Be short and concrete.
- If you do not know the answer, reply with exactly one of these sentences and nothing else,
  in the language of the question:
  Turkish: {NOT_FOUND["tr"]}
  English: {NOT_FOUND["en"]}
  Portuguese: {NOT_FOUND["pt"]}"""


def format_sources(hits):
    return "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(hits, 1))


def language_line(question):
    return f"Answer in {LANG_NAMES[detect_lang(question)]}."


def is_not_found(answer):
    return any(s.lower() in answer.lower() for s in NOT_FOUND.values())


def cited_urls(answer, hits):
    urls = []
    for n in re.findall(r"\[(\d+)\]", answer):
        i = int(n) - 1
        if 0 <= i < len(hits) and hits[i][0]["url"] not in urls:
            urls.append(hits[i][0]["url"])
    return urls


def answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Sources:\n\n{sources}\n\nQuestion: {question}\n{focus}{language}"),
        ]
    )
    return prompt | llm


def closed_book_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [("system", CLOSED_BOOK_PROMPT), ("human", "Question: {question}\n{language}")]
    )
    return prompt | llm


def answer(llm, question, hits, focus=""):
    reply = answer_chain(llm).invoke(
        {
            "sources": format_sources(hits),
            "question": question,
            "focus": focus,
            "language": language_line(question),
        }
    )
    text, truncated = read_reply(reply)
    return text, [] if is_not_found(text) else cited_urls(text, hits), truncated


def answer_closed_book(llm, question):
    reply = closed_book_chain(llm).invoke(
        {"question": question, "language": language_line(question)}
    )
    return read_reply(reply)[0]


def ask(retriever, llm, question, k=TOP_K):
    hits = retriever.search(question, k)
    text, cited, truncated = answer(llm, question, hits)
    return {
        "hits": hits,
        "answer": text,
        "cited": cited,
        "searches": [],
        "llm_calls": 1,
        "truncated": truncated,
    }
