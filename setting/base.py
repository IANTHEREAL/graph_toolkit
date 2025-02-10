import os
import json
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


# Model configurations
def parse_model_configs() -> dict:
    """Parse MODEL_CONFIGS from environment variable"""
    config_str = os.environ.get("MODEL_CONFIGS", "{}")
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        print(f"Warning: Invalid MODEL_CONFIGS format: {config_str}")
        return {}


MODEL_CONFIGS = parse_model_configs()
