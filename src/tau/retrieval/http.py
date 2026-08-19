import time

import requests


def post_with_retry(url, headers, json, timeout=60, max_retries=5, backoff_seconds=20):
    """POST with retry on HTTP 429 (rate limited).

    Voyage's reduced-tier keys (no payment method on file) cap out at 3 requests/minute
    and don't send a Retry-After header, so we back off linearly between attempts. Any
    non-429 response (success or another error) is returned immediately.
    """
    response = requests.post(url, headers=headers, json=json, timeout=timeout)

    for attempt in range(max_retries):
        if response.status_code != 429:
            return response

        time.sleep(backoff_seconds * (attempt + 1))
        response = requests.post(url, headers=headers, json=json, timeout=timeout)

    return response
