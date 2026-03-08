import io
from typing import List, Dict, Any

import streamlit as st
from PIL import Image

from config import settings
from services.vision_service import VisionService
from services.recipe_service import RecipeService
try:
    from services.substitute_service import SubstituteService
except Exception:
    # Fallback lightweight substitute service if import fails
    class SubstituteService:
        COMMON_SUBS = {
            "milk": ["plant-based milk", "water + milk powder"],
            "butter": ["margarine", "olive oil"],
            "eggs": ["flax egg", "chia egg"],
            "sugar": ["honey", "maple syrup"],
            "olive oil": ["vegetable oil", "canola oil"],
            "pasta": ["rice", "quinoa"],
        }

        def suggest_substitutes(self, missing):
            out = {}
            for m in missing:
                key = m.lower()
                if key in self.COMMON_SUBS:
                    out[m] = self.COMMON_SUBS[key]
                else:
                    base = key.split()[0]
                    out[m] = self.COMMON_SUBS.get(base, ["No common substitute found"])
            return out

try:
    from services.health_service import HealthService
except Exception:
    class HealthService:
        def score_recipe(self, recipe):
            return 6, "Fallback health score — mock summary."

try:
    from utils.image_utils import enhance_image_bytes
except Exception:
    from io import BytesIO
    from PIL import Image as PILImage, ImageEnhance

    def enhance_image_bytes(img: PILImage.Image, max_size: int = 1200, quality: int = 85) -> bytes:
        w, h = img.size
        max_side = max(w, h)
        if max_side > max_size:
            scale = max_size / max_side
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, PILImage.LANCZOS)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

st.set_page_config(page_title="Kitchen Vision", layout="wide")

vision = VisionService()
recipe_svc = RecipeService()
sub_svc = SubstituteService()
health_svc = HealthService()


def sidebar_instructions() -> None:
    st.sidebar.title("Kitchen Vision")
    st.sidebar.markdown("Upload a clear photo of your open pantry (JPG/PNG).")
    st.sidebar.markdown("Set AI provider and API keys in environment variables or use mock mode for demo.")


def main() -> None:
    sidebar_instructions()

    st.title("Kitchen Vision — Pantry to Recipes")

    uploaded = st.file_uploader("Upload pantry image", type=["jpg", "jpeg", "png"]) 

    if uploaded is None:
        st.info("Upload an image to get started. See README for sample JSON output.")
        return

    try:
        img = Image.open(uploaded).convert("RGB")
        st.subheader("Image Preview")
        st.image(img, use_column_width=True)

        # Enhance image and get bytes
        img_bytes = enhance_image_bytes(img)

        with st.spinner("Detecting items in the pantry..."):
            detected = vision.detect_items(img_bytes)

        # Categorize items
        categorized = vision.categorize_items(detected)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.header("Detected Items")
            for item in detected:
                st.write(f"- {item}")

            st.markdown("---")
            st.header("Categories")
            for cat, items in categorized.items():
                st.write(f"**{cat}**: {', '.join(items) if items else '—'}")

        with col2:
            st.header("Recipes")
            with st.spinner("Generating recipes..."):
                recipes = recipe_svc.generate_recipes(detected, categorized)

            for i, r in enumerate(recipes, 1):
                st.subheader(f"Recipe {i}: {r['title']}")
                st.markdown(f"**Difficulty:** {r['difficulty']} — **Time:** {r['time_minutes']} min")
                st.markdown("**Ingredients**")
                for ing in r['ingredients']:
                    st.write(f"- {ing}")
                st.markdown("**Steps**")
                for step_n, step in enumerate(r['steps'], 1):
                    st.write(f"{step_n}. {step}")

                # Health score
                score, explanation = health_svc.score_recipe(r)
                st.markdown(f"**Health score:** {score}/10")
                st.write(explanation)
                st.markdown("---")

        with col3:
            st.header("Grocery List (Missing Items)")
            missing = recipe_svc.identify_missing_ingredients(recipes, detected)
            if missing:
                for m in missing:
                    st.checkbox(m, value=False)
            else:
                st.write("No missing ingredients detected — you have what you need!")

            st.markdown("---")
            st.header("Substitutions")
            subs = sub_svc.suggest_substitutes(missing)
            for k, v in subs.items():
                st.write(f"**{k}** → {', '.join(v)}")

        st.success("Done — see structured JSON output below")

        # Structured JSON output
        st.subheader("Structured Output (JSON)")
        out = {
            "detected_items": detected,
            "categories": categorized,
            "recipes": recipes,
            "missing": missing,
        }
        st.json(out)

    except Exception as e:
        st.error(f"Error processing image: {e}")


if __name__ == '__main__':
    main()
