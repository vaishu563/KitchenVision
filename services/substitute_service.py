from typing import List, Dict


class SubstituteService:
    """Suggests substitutes for common pantry ingredients."""

    COMMON_SUBS = {
        "milk": ["plant-based milk", "water + milk powder"],
        "butter": ["margarine", "olive oil"],
        "eggs": ["flax egg", "chia egg"],
        "sugar": ["honey", "maple syrup"],
        "olive oil": ["vegetable oil", "canola oil"],
        "pasta": ["rice", "quinoa"],
    }

    def suggest_substitutes(self, missing: List[str]) -> Dict[str, List[str]]:
        """Return mapping of missing ingredient to list of possible substitutes."""
        out = {}
        for m in missing:
            key = m.lower()
            if key in self.COMMON_SUBS:
                out[m] = self.COMMON_SUBS[key]
            else:
                # Basic heuristic: split words and try base word
                base = key.split()[0]
                out[m] = self.COMMON_SUBS.get(base, ["No common substitute found"])
        return out
