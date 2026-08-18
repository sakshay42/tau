from datetime import datetime, timezone
import hashlib
import requests

from tau.ingestion.models import Document


BASE_URL = "https://hacker-news.firebaseio.com/v0"


def make_id(hn_id):
    return hashlib.sha256(
        f"hackernews:{hn_id}".encode("utf-8")
    ).hexdigest()


def get_item(item_id):
    url = f"{BASE_URL}/item/{item_id}.json"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return response.json()


def fetch_hackernews(limit=20):
    response = requests.get(
        f"{BASE_URL}/newstories.json",
        timeout=10
    )
    response.raise_for_status()

    story_ids = response.json()[:limit]

    documents = []

    for story_id in story_ids:
        item = get_item(story_id)

        if not item:
            continue

        if item.get("type") != "story":
            continue

        title = item.get("title")
        url = item.get("url")

        if not title:
            continue

        published_at = datetime.fromtimestamp(
            item["time"],
            tz=timezone.utc
        )

        document = Document(
            id=make_id(story_id),
            title=title,
            text=title,
            source="hackernews",
            published_at=published_at,
            ingested_at=datetime.now(timezone.utc),
            url=url,
            metadata={
                "hn_id": story_id,
                "score": item.get("score"),
                "author": item.get("by"),
                "comments": item.get("descendants", 0),
            }
        )

        documents.append(document)

    return documents