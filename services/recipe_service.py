from typing import List, Dict, Any
from collections import Counter

from config import settings
from utils.prompt_templates import RECIPE_PROMPT


class RecipeService:
    """Generates recipes given detected pantry items and categories.

    This implementation uses a deterministic mock generator. Swap in LLM calls
    in `generate_recipes` to use real AI models.
    """

    def __init__(self):
        pass

    def generate_recipes(self, detected: List[str], categorized: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Return 3 recipe dicts with required keys.

        Keys: title, ingredients (list), steps (list of 3), time_minutes (int), difficulty
        """
        base = [d for d in detected]
        recipes = []
        # Create three simple recipes using common pantry combos
        combos = [
            {"title": "Pasta with Tomato Sauce", "needs": ["pasta", "canned tomatoes", "olive oil"]},
            {"title": "Scrambled Eggs on Toast", "needs": ["eggs", "butter", "salt"]},
            {"title": "Fruit Yogurt Bowl", "needs": ["banana", "apple", "milk"]},
        ]

        for combo in combos:
            ingredients = []
            for need in combo["needs"]:
                # if present in detected, include; otherwise mark as missing (still include)
                if any(need.lower() in d.lower() for d in detected):
                    ingredients.append(need)
                else:
                    ingredients.append(need + " (missing)")

            steps = [
                f"Prepare ingredients: {', '.join(ingredients)}.",
                "Cook according to basic directions for each ingredient.",
                "Serve and enjoy."
            ]

            recipes.append({
                "title": combo["title"],
                "ingredients": ingredients,
                "steps": steps,
                "time_minutes": 15 if combo["title"] != "Fruit Yogurt Bowl" else 5,
                "difficulty": "Easy",
            })

        return recipes

    def identify_missing_ingredients(self, recipes: List[Dict[str, Any]], detected: List[str]) -> List[str]:
        """Return a unique list of missing ingredients across recipes."""
        missing = []
        detected_lower = [d.lower() for d in detected]
        for r in recipes:
            for ing in r.get("ingredients", []):
                if "(missing)" in ing:
                    missing.append(ing.replace(" (missing)", ""))
        # Deduplicate
        return list(dict.fromkeys(missing))
