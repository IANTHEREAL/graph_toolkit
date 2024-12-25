import os
from dotenv import load_dotenv

load_dotenv()

# LLM settings
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "aya-expanse")
FAST_LLM_MODEL = os.environ.get("FAST_LLM_MODEL", None)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

if FAST_LLM_MODEL is None:
    FAST_LLM_MODEL = LLM_MODEL

# DB settings
DATABASE_URI = os.environ.get("DATABASE_URI")
SESSION_POOL_SIZE: int = os.environ.get("SESSION_POOL_SIZE", 40)
