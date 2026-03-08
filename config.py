import os
from dataclasses import dataclass
from typing import Optional
import os
from dataclasses import dataclass
from typing import Optional

# Load .env when available so environment variables from .env are available during local runs
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed yet, environment variables may still come from the shell
    pass


@dataclass
class Settings:
    provider: str = os.getenv("KV_PROVIDER", "mock")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    azure_api_key: Optional[str] = os.getenv("AZURE_API_KEY")
    azure_endpoint: Optional[str] = os.getenv("AZURE_ENDPOINT")
    gemeai_api_key: Optional[str] = os.getenv("GEMEAI_API_KEY")
    gemeai_endpoint: Optional[str] = os.getenv("GEMEAI_ENDPOINT")
    cache_dir: str = os.getenv("KV_CACHE_DIR", ".cache")


settings = Settings()
