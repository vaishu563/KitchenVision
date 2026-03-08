from typing import Tuple, Dict, Any


class HealthService:
    """Produces a simple health score for a recipe.

    This mock implementation scores recipes using heuristics on ingredients;
    replace with a nutrition API (e.g., USDA or nutritionix) for production.
    """

    def __init__(self):
        pass

    def score_recipe(self, recipe: Dict[str, Any]) -> Tuple[int, str]:
        """Return (score 1-10, explanation string)."""
        ingredients = [i.lower() for i in recipe.get("ingredients", [])]

        # Simple heuristics
        score = 6
        explanation_parts = []

        if any("vegetable" in i or "apple" in i or "banana" in i or "peas" in i for i in ingredients):
            score += 1
            explanation_parts.append("Includes produce → +1")

        if any("butter" in i or "olive oil" in i for i in ingredients):
            score += 0  # neutral
            explanation_parts.append("Contains fats → moderate impact")

        if any("sugar" in i or "honey" in i or "maple" in i for i in ingredients):
            score -= 1
            explanation_parts.append("Contains sweetener → -1")

        if any("eggs" in i for i in ingredients):
            score += 0
            explanation_parts.append("Protein source present")

        score = max(1, min(10, score))

        explanation = "; ".join(explanation_parts) or "Balanced—no strong positives/negatives detected."
        # Add a short structured nutrition-like summary (mock)
        explanation += f" | Estimated calories: ~{200 + 50 * len(ingredients)} kcal"

        return score, explanation
