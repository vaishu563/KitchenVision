DETECT_PROMPT = """
You are given a photo of an open pantry. List all visible food items and ingredients
as short nouns or noun phrases (e.g., 'pasta', 'canned tomatoes', 'olive oil').
Return a JSON array of strings only.
"""

RECIPE_PROMPT = """
You are given a list of available pantry items and categories. Produce three simple,
home-cooking recipes using these items. For each recipe return JSON with keys:
title, ingredients (list), steps (list of 3), time_minutes (int), difficulty.
Keep recipes short and achievable with basic pantry ingredients.
"""
