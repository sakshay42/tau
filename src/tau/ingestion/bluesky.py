import asyncio
import json
import hashlib
import re
from datetime import datetime, timezone

import websockets

from tau.ingestion.models import Document


JETSTREAM_URL = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)


def make_id(did, rkey):
    raw = f"bluesky:{did}:{rkey}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def event_to_document(event):
    commit = event.get("commit", {})

    # We only care about newly-created posts
    if commit.get("operation") != "create":
        return None

    record = commit.get("record", {})

    text = record.get("text")

    if not text:
        return None

    if not is_relevant(text):
        return None

    did = event.get("did")
    rkey = commit.get("rkey")

    if not did or not rkey:
        return None

    created_at = record.get("createdAt")

    published_at = None

    if created_at:
        published_at = datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        )

    return Document(
        id=make_id(did, rkey),
        title=None,
        text=text,
        source="bluesky",
        published_at=published_at,
        ingested_at=datetime.now(timezone.utc),

        # We don't yet know the user's handle,
        # so don't fabricate a public bsky.app URL.
        url=None,

        metadata={
            "did": did,
            "rkey": rkey,
            "collection": commit.get("collection")
        }
    )


async def fetch_bluesky(limit=20):
    documents = []

    async with websockets.connect(
        JETSTREAM_URL
    ) as websocket:

        while len(documents) < limit:

            message = await websocket.recv()

            event = json.loads(message)

            document = event_to_document(event)

            if document is not None:
                documents.append(document)

    return documents




KEYWORDS = {
    "ai",
    "artificial intelligence",
    "openai",
    "anthropic",
    "claude",
    "chatgpt",
    "llm",
    "machine learning",
    "agent",
    "agents",
    "nvidia",
    "gemini",
    "deepmind",
}


KEYWORDS = {
    "ai",
    "artificial intelligence",
    "openai",
    "anthropic",
    "claude",
    "chatgpt",
    "llm",
    "machine learning",
    "agent",
    "agents",
    "nvidia",
    "gemini",
    "deepmind",
}


def is_relevant(text):
    text = text.lower()

    for keyword in KEYWORDS:

        # phrases like "machine learning"
        if " " in keyword:
            if keyword in text:
                return True

        # single words like "ai", "llm", "agent"
        else:
            pattern = rf"\b{re.escape(keyword)}\b"

            if re.search(pattern, text):
                return True

    return False
