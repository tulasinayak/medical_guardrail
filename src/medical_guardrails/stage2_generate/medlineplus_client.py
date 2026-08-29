"""MedlinePlus (NLM/NIH) health topics web service client -- consumer-facing
symptom/condition information for query types where there's no drug to look
up (SYMPTOM/HOME_REMEDY/GENERAL_INFO). Free, no auth, unlimited calls per
NLM's own technical bulletins for this service.

Verified live against the real endpoint while building this: the search
index does token matching against its indexed fields, not natural-language
understanding -- feeding it a raw free-text question (filler words, a
misspelling like "paining" instead of "pain") reliably returns zero hits,
while a clean 1-3 word topic phrase ("back pain") reliably matches.
`extract_symptom_topic` exists to bridge that gap with one small LLM call
before the search is made.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree

import httpx

from medical_guardrails.llm.base import LLMClient

_TAG_RE = re.compile(r"<[^>]+>")

_TOPIC_SYSTEM_PROMPT = (
    "You extract a short search phrase from a user's health question, for looking up a "
    "consumer health topic in a medical topic index. Correct any misspellings and drop filler "
    "words. Respond with ONLY 1-3 lowercase keywords naming the main symptom or health topic -- "
    "no punctuation, no explanation, nothing else."
)


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    return _TAG_RE.sub("", text).strip()


def extract_symptom_topic(query_text: str, llm_client: LLMClient) -> str:
    """Best-effort short topic phrase for a MedlinePlus search. Never
    raises -- returns "" on any failure (bad response, LLM error), which
    the caller treats the same as any other empty retrieval result rather
    than as an error to propagate."""
    try:
        raw = llm_client.chat(
            [
                {"role": "system", "content": _TOPIC_SYSTEM_PROMPT},
                {"role": "user", "content": query_text},
            ]
        )
    except Exception:
        return ""
    topic = raw.strip().strip(".\"'").lower()
    return topic if 0 < len(topic) <= 60 else ""


class MedlinePlusClient:
    def __init__(self, base_url: str = "https://wsearch.nlm.nih.gov/ws/query", timeout: float = 15.0) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def search_health_topics(self, term: str, max_results: int = 1) -> list[dict[str, str]]:
        """Returns up to `max_results` matching topics as {"title",
        "summary", "url"} dicts, HTML-stripped. Returns [] for no matches,
        an empty term, an unreachable service, or a malformed response --
        this is supplementary evidence, so any lookup failure here degrades
        to "no MedlinePlus evidence" rather than raising."""
        if not term:
            return []
        try:
            response = httpx.get(
                self.base_url,
                params={"db": "healthTopics", "term": term, "retmax": max_results, "rettype": "brief"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except (httpx.HTTPError, ElementTree.ParseError):
            return []

        topics: list[dict[str, str]] = []
        for document in root.findall(".//document"):
            title = ""
            summary = ""
            for content in document.findall("content"):
                name = content.get("name")
                if name == "title" and not title:
                    title = _strip_html(content.text)
                elif name == "FullSummary" and not summary:
                    summary = _strip_html(content.text)
            if title:
                topics.append({"title": title, "summary": summary, "url": document.get("url", "")})
        return topics
