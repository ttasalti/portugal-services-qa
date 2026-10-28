import os
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw"
PAGES = DATA / "pages.jsonl"
MANUAL = DATA / "manual"
CHROMA = DATA / "chroma"
SOURCES = Path("sources.txt")
QUESTIONS = Path("questions.jsonl")
RESULTS = Path("results")

COLLECTION = "pages"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
AGENCIES = ("aima", "tax", "social")

RETRIEVAL = os.environ.get("RETRIEVAL", "hybrid+rerank")
TOP_K = 5
AGENT_K = 3
SPECIALIST_K = 4
MAX_SEARCHES = 4
ANSWER_TOKENS = 1024
PLANNER_TOKENS = 512
CANDIDATES = 20
RRF_K = 60
