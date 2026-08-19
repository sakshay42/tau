from tau.retrieval.http import post_with_retry


EMBED_MODEL = "voyage-3.5"
EMBED_URL = "https://api.voyageai.com/v1/embeddings"


def _embed(texts, voyage_api_key, input_type, model=EMBED_MODEL, batch_size=128):
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = post_with_retry(
            EMBED_URL,
            headers={
                "Authorization": f"Bearer {voyage_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": batch,
                "model": model,
                "input_type": input_type,
            },
            timeout=60,
        )

        response.raise_for_status()

        vectors.extend(
            item["embedding"] for item in response.json()["data"]
        )

    return vectors


def embed_documents(texts, voyage_api_key, model=EMBED_MODEL, batch_size=128):
    """Embed a list of document texts. Returns one vector per text, same order."""
    return _embed(texts, voyage_api_key, "document", model=model, batch_size=batch_size)


def embed_query(query, voyage_api_key, model=EMBED_MODEL):
    """Embed a single search query. Returns one vector."""
    return _embed([query], voyage_api_key, "query", model=model)[0]
