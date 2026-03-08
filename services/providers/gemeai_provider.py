import time
import json
from typing import List, Optional
import requests

from config import settings


class GemeaiProviderError(Exception):
    pass


def detect_items_from_image(image_bytes: bytes, timeout: int = 30) -> List[str]:
    """Call the Gemeai vision API to detect items in the image.

    Expects `settings.gemeai_api_key` and optionally `settings.gemeai_endpoint`.
    Returns a list of detected item strings.
    """
    api_key = settings.gemeai_api_key or None
    endpoint = settings.gemeai_endpoint or "https://api.geme.ai/v1/vision/detect"

    if not api_key:
        raise GemeaiProviderError("GEMEAI_API_KEY not set in environment")

    headers = {"Authorization": f"Bearer {api_key}"}

    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data = {"response_format": "json"}

    attempts = 3
    backoff = 1.0
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(endpoint, headers=headers, files=files, data=data, timeout=timeout)
            if resp.status_code == 200:
                j = resp.json()
                # Expect provider to return {'items': [...]} or similar
                if isinstance(j, dict) and "items" in j and isinstance(j["items"], list):
                    return [str(x) for x in j["items"]]
                # Try alternate common shapes
                if isinstance(j, dict) and "predictions" in j and isinstance(j["predictions"], list):
                    # Flatten possible label fields
                    out = []
                    for p in j["predictions"]:
                        if isinstance(p, dict) and "label" in p:
                            out.append(str(p["label"]))
                    if out:
                        return out
                # As fallback, try parsing text field
                if isinstance(j, dict) and "text" in j:
                    try:
                        parsed = json.loads(j["text"]) if isinstance(j["text"], str) else None
                        if isinstance(parsed, list):
                            return [str(x) for x in parsed]
                    except Exception:
                        pass
                # Unknown successful response format
                raise GemeaiProviderError(f"Unexpected response format from Gemeai: {j}")

            # Non-200
            err = resp.text
            if attempt < attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise GemeaiProviderError(f"Gemeai API error {resp.status_code}: {err}")

        except requests.RequestException as exc:
            if attempt < attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise GemeaiProviderError(f"Request failed: {exc}")
