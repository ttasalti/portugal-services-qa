import hashlib
import json
import re
from urllib.parse import urlparse

import httpx
import trafilatura
from lxml import html as lxml_html

from ptqa.config import MANUAL, PAGES, RAW, SOURCES

# aima.gov.pt serves an incomplete certificate chain
NO_VERIFY_HOSTS = {"aima.gov.pt"}
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) portugal-services-qa"}
BLOCKS = {"p", "li", "h2", "h3", "h4", "h5", "td", "dt", "dd"}
NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
SERVICE_SECTIONS = ["who", "when", "duedate", "how", "otherInfo"]


def read_sources(path=SOURCES):
    sources = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        url, tags = line.split()
        sources.append((url, tags.split(",")))
    return sources


def raw_path(url):
    return RAW / f"{hashlib.sha1(url.encode()).hexdigest()[:12]}.html"


def clean(s):
    return " ".join(s.split())


def own_text(el):
    parts = [el.text or ""]
    for child in el:
        if child.tag not in BLOCKS and not any(d.tag in BLOCKS for d in child.iterdescendants()):
            parts.append(child.text_content())
        parts.append(child.tail or "")
    return re.sub(r"\s+([,.;:])", r"\1", clean(" ".join(parts)))


def block_lines(el):
    lines = []
    bullet = False
    for b in el.iter(*BLOCKS):
        own = own_text(b)
        if not own:
            bullet = bullet or b.tag == "li"
            continue
        if b.tag == "li" or bullet:
            own = f"- {own}"
            bullet = False
        elif b.tag in {"h2", "h3", "h4", "h5"}:
            own = f"### {own}"
        lines.append(own)
    return lines or [clean(el.text_content())]


def md_to_text(s):
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = s.replace("++", "").replace("**", "")
    s = re.sub(r"^\s*[*-]\s+", "- ", s, flags=re.M)
    return "\n".join(ln.rstrip() for ln in s.splitlines() if ln.strip())


def descriptions(block):
    items = block.get("descriptions", [])
    return [md_to_text(d["description"]) for d in items if d.get("description")]


def govpt_service(content):
    parts = descriptions({"descriptions": [content["serviceTitle"]]})
    for key in sorted(SERVICE_SECTIONS, key=lambda k: content.get(k, {}).get("position", 99)):
        sec = content.get(key)
        if not sec:
            continue
        if key == "how":
            for ch in sec.get("serviceChannels", []):
                for tab in ch.get("verticalTabs", []):
                    body = descriptions(tab)
                    if body:
                        head = f"## {ch['title']}: {md_to_text(tab['title'])}"
                        parts.append("\n".join([head, *body]))
        elif key == "otherInfo":
            for item in sec.get("accordion", []):
                body = descriptions(item)
                if body:
                    parts.append("\n".join([f"## {md_to_text(item.get('title', ''))}", *body]))
        else:
            title = md_to_text(sec.get("title", "")) or "Prazo"
            parts.append("\n".join([f"## {title}", *descriptions(sec)]))
    return "\n\n".join(parts)


def govpt_guide(sections):
    parts = []
    for sec in sections:
        lines = [f"## {sec['title']}"] if sec.get("title") else []
        for comp in sec.get("components", []):
            c = comp.get("component", {})
            chunks = [c.get("body", {}).get("html")]
            chunks += [m.get("message", {}).get("html") for m in c.get("messages", [])]
            for chunk in filter(None, chunks):
                lines += block_lines(lxml_html.fromstring(chunk))
        if lines:
            parts.append("\n".join(lines))
    return "\n\n".join(parts)


def govpt_text(html):
    m = NEXT_DATA.search(html)
    if not m:
        return None
    props = json.loads(m.group(1))["props"]["pageProps"]
    if props.get("serverSideContent", {}).get("how"):
        return govpt_service(props["serverSideContent"])
    results = props.get("results", {})
    if results.get("sections"):
        return govpt_guide(results["sections"])
    body = results.get("body", {}).get("html")
    if body:
        return "\n".join(block_lines(lxml_html.fromstring(body)))
    return None


def extract(html, url):
    doc = lxml_html.fromstring(html)
    title = clean(doc.xpath("string(//h1)")) or url

    # trafilatura skips AIMA accordions
    items = doc.xpath('//li[contains(@class, "uk-accordion-item")]')
    if items:
        parts = []
        for item in items:
            head = clean(item.xpath('string(.//*[contains(@class, "uk-accordion-title")])'))
            body = item.xpath('.//*[contains(@class, "uk-accordion-content")]')
            lines = block_lines(body[0]) if body else []
            parts.append("\n".join([f"## {head}", *lines]))
        return title, "\n\n".join(parts)

    # gov.pt fills the how-to tabs client side from __NEXT_DATA__
    text = govpt_text(html) if urlparse(url).hostname == "www.gov.pt" else None
    if text is None:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
    return title, text or ""


def read_manual(path):
    head, _, text = path.read_text(encoding="utf-8").partition("\n\n")
    meta = dict(line.split(": ", 1) for line in head.splitlines())
    return {
        "url": meta["url"],
        "title": meta["title"],
        "agencies": meta["agencies"].split(","),
        "text": text.strip(),
    }


def download(url):
    verify = urlparse(url).hostname not in NO_VERIFY_HOSTS
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30, verify=verify) as c:
        resp = c.get(url)
    resp.raise_for_status()
    return resp.text


def fetch(refresh=False):
    RAW.mkdir(parents=True, exist_ok=True)
    pages = []
    for url, agencies in read_sources():
        path = raw_path(url)
        if refresh or not path.exists():
            path.write_text(download(url), encoding="utf-8")
        title, text = extract(path.read_text(encoding="utf-8"), url)
        pages.append({"url": url, "title": title, "agencies": agencies, "text": text})
        print(f"{len(text.split()):6d} words  {title[:70]}")

    for path in sorted(MANUAL.glob("*.txt")):
        page = read_manual(path)
        pages.append(page)
        print(f"{len(page['text'].split()):6d} words  {page['title'][:70]} (saved copy)")

    with PAGES.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pages)} pages -> {PAGES}")
