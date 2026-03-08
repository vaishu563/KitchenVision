# Kitchen Vision

Kitchen Vision is a Streamlit app that accepts a pantry photo and generates:

- Detected pantry items and categories
- 3 simple recipe suggestions (3-step each)
- Health score for each recipe
- Grocery list of missing items and suggested substitutions

This repository includes a mock mode so you can run the app without API keys.

## Quickstart (local)

1. Copy `.env.example` to `.env` and set `KV_PROVIDER=mock` for demo mode.
2. Create a venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run "KitchenVision/app.py"
```

## Using a real AI provider

Set `KV_PROVIDER=openai` and add your `OPENAI_API_KEY` to `.env`. The provider hooks
are left as placeholders in `services/vision_service.py`; implement `_call_provider_detect`
and expand `recipe_service` to call the LLM for richer outputs.

Alternatively set `KV_PROVIDER=gemeai` and provide `GEMEAI_API_KEY` (and optional `GEMEAI_ENDPOINT`) in `.env` to use the included Gemeai adapter.

## Docker

Build and run with Docker (example):

```bash
docker build -t kitchen-vision .
docker run -p 8501:8501 --env-file .env kitchen-vision
```

## Deployment

- Streamlit Cloud: push this repo to GitHub and connect it in Streamlit Cloud; set secrets via the UI.
- Docker: use the `Dockerfile` included to containerize the app.

## Example output (JSON)

```json
{
  "detected_items": ["rice", "pasta", "canned tomatoes", "olive oil", "salt", "eggs"],
  "categories": {"Dry Storage": ["rice", "pasta", "canned tomatoes", "olive oil", "salt"], "Cold Storage": ["eggs"], "Frozen": [], "Fresh Produce": []},
  "recipes": [ {"title":"Pasta with Tomato Sauce","ingredients":["pasta","canned tomatoes","olive oil"],"steps":["Prepare ingredients: pasta, canned tomatoes, olive oil.","Cook according to basic directions for each ingredient.","Serve and enjoy."],"time_minutes":15,"difficulty":"Easy"}, ... ],
  "missing": []
}
```

## Notes

- This project is scaffolded to make provider integration straightforward and swap-able.
- Follow code comments to add real provider calls (OpenAI/Gemini/Azure).
