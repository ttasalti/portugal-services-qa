import json
import re
import unicodedata

from ptqa.config import PAGES

STOPWORDS = {
    "tr": set(
        "ve bir bu için ile olarak veya gerekir gerekiyor değil olan var yok mi mı ise kadar "
        "sonra önce nasıl hangi ne neden".split()
    ),
    "en": set(
        "the and to of you is for your with are or be can must not if in on this have do "
        "does how what which".split()
    ),
    "pt": set(
        "que para não os as do dos das uma um com é ou se pode deve no na ao em o a e me meu "
        "minha como onde tenho há já".split()
    ),
}
LANG_NAMES = {"tr": "Turkish", "en": "English", "pt": "European Portuguese"}


def load_pages(path=PAGES):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_sections(text):
    sections, head, body = [], "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if body:
                sections.append((head, "\n".join(body)))
            head, body = line[3:].strip(), []
        else:
            body.append(line)
    if body:
        sections.append((head, "\n".join(body)))
    return [(h, b.strip()) for h, b in sections if b.strip()]


def detect_lang(text):
    found = re.findall(r"\w+", text.lower())
    scores = {lang: sum(t in stop for t in found) for lang, stop in STOPWORDS.items()}
    scores["tr"] += 3 * len(re.findall(r"[ğışöü]", text.lower()))
    scores["pt"] += 3 * len(re.findall(r"[ãõáâéêíóôú]", text.lower()))
    return max(scores, key=scores.get)


def tokens(text):
    text = unicodedata.normalize("NFKD", text.lower().replace("ı", "i"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


def words(text):
    return [w for w in tokens(text) if len(w) > 2]


def trigrams(ws):
    return {tuple(ws[i : i + 3]) for i in range(len(ws) - 2)}


def near_copy(question, sections):
    q = words(question)
    for head, body in sections:
        head_share = len(set(q) & set(words(head))) / max(1, len(set(q)))
        shared = len(trigrams(q) & trigrams(words(f"{head} {body}")))
        if head_share >= 0.6 or shared >= 2:
            return True
    return False
