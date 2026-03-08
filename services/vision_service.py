import base64
import json
from typing import List, Dict
from PIL import Image
import io

from config import settings
from utils.prompt_templates import DETECT_PROMPT

# Optional provider imports
try:
    from services.providers.gemeai_provider import detect_items_from_image
except Exception:
    detect_items_from_image = None


class VisionService:
    """Vision service that abstracts provider calls for detecting pantry items.

    This implementation provides a mock detector when no provider is configured.
    Swap-in real provider logic in `_call_provider_detect`.
    """

    def __init__(self):
        pass

    def detect_items(self, image_bytes: bytes) -> List[str]:
        """Return a list of detected item names from the pantry image.

        If `settings.provider` is set to a real provider and keys are configured,
        implement `_call_provider_detect` to call the provider API.
        """
        if settings.provider != "mock":
            # Route to configured provider
            return self._call_provider_detect(image_bytes)

        # Mock: very simple heuristic based on image size and random sample labels.
        # This is deterministic so demo runs without API keys.
        sample = [
            "rice",
            "pasta",
            "canned tomatoes",
            "olive oil",
            "salt",
            "eggs",
            "milk",
            "butter",
            "frozen peas",
            "apple",
            "banana",
        ]
        # Use image size to pick a subset
        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            count = max(4, min(len(sample), (w * h) // (100000)))
        except Exception:
            count = 6

        # Simple deterministic selection
        detected = sample[:count]
        return detected

    def _call_provider_detect(self, image_bytes: bytes) -> List[str]:
        """Placeholder for provider-specific detection implementations.

        Replace this with OpenAI/Gemini/Azure calls. The function should return
        a list of ingredient strings extracted by the model.
        """
        provider = settings.provider.lower() if settings.provider else ""
        if provider == "gemeai":
            if detect_items_from_image is None:
                raise NotImplementedError("Gemeai provider module not available")
            try:
                return detect_items_from_image(image_bytes)
            except Exception as e:
                # Gracefully degrade to mock detector on network/SSL/provider errors
                import logging
                logging.warning("GemeAI provider call failed, falling back to mock: %s", e)
                # Reuse mock behavior
                sample = [
                    "rice",
                    "pasta",
                    "canned tomatoes",
                    "olive oil",
                    "salt",
                    "eggs",
                    "milk",
                    "butter",
                    "frozen peas",
                    "apple",
                    "banana",
                ]
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    w, h = img.size
                    count = max(4, min(len(sample), (w * h) // (100000)))
                except Exception:
                    count = 6
                return sample[:count]

        # Other providers can be added here (openai, azure, etc.)
        raise NotImplementedError(f"Provider integration for '{settings.provider}' not implemented yet")

    def categorize_items(self, items: List[str]) -> Dict[str, List[str]]:
        """Categorize detected items into storage categories.

        This uses a small heuristic mapping; extend with a database if needed.
        """
        mapping = {
            "Dry Storage": ["rice", "pasta", "olive oil", "salt", "canned tomatoes"],
            "Cold Storage": ["milk", "butter"],
            "Frozen": ["frozen peas"],
            "Fresh Produce": ["apple", "banana"],
        }
        categorized = {k: [] for k in mapping}
        for it in items:
            placed = False
            lower = it.lower()
            for cat, examples in mapping.items():
                if lower in examples or any(ex in lower for ex in examples):
                    categorized[cat].append(it)
                    placed = True
                    break
            if not placed:
                categorized.setdefault("Dry Storage", []).append(it)
        return categorized
