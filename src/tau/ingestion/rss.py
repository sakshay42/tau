import hashlib
from datetime import datetime, timezone

import feedparser
from bs4 import BeautifulSoup

from tau.ingestion.models import Document


def clean_html(html):
    if not html:
        return ""

    return BeautifulSoup(
        html,
        "html.parser"
    ).get_text(" ", strip=True)


def make_id(url):
    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()


def parse_datetime(entry):
    if not entry.get("published_parsed"):
        return None

    return datetime(
        *entry.published_parsed[:6],
        tzinfo=timezone.utc
    )


def fetch_rss(feed_url, source, feed_name=None):
    feed = feedparser.parse(feed_url)

    documents = []

    for entry in feed.entries:

        url = entry.get("link")

        if not url:
            continue

        document = Document(
            id=make_id(url),
            title=entry.get("title"),
            text=clean_html(
                entry.get("summary", "")
            ),
            source=source,
            published_at=parse_datetime(entry),
            ingested_at=datetime.now(timezone.utc),
            url=url,
            metadata={
                "feed": feed_name,
                "author": entry.get("author")
            }
        )

        documents.append(document)

    return documents