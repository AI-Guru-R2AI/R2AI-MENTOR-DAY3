import os

from dotenv import load_dotenv

# Find workspace root relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)

dotenv_path = os.path.join(workspace_root, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "law_articles")

EMBEDDING_OPENAI_API_KEY = os.getenv("EMBEDDING_OPENAI_API_KEY", "not_needed")
EMBEDDING_OPENAI_BASE_URL = os.getenv("EMBEDDING_OPENAI_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_OPENAI_MODEL = os.getenv("EMBEDDING_OPENAI_MODEL", "bge-large-en-v1.5")

RERANKER_OPENAI_API_KEY = os.getenv("RERANKER_OPENAI_API_KEY", "not_needed")
RERANKER_OPENAI_BASE_URL = os.getenv("RERANKER_OPENAI_BASE_URL", "http://localhost:1234/v1")
RERANKER_OPENAI_MODEL = os.getenv("RERANKER_OPENAI_MODEL", "bge-reranker-v2-m3")

REASONING_API_KEY = os.getenv("REASONING_API_KEY")
REASONING_BASE_URL = os.getenv("REASONING_BASE_URL")
REASONING_MODEL = os.getenv("REASONING_MODEL", "gpt-4o-mini")
REASONING_MODEL_CONTEXT = int(os.getenv("REASONING_MODEL_CONTEXT", 16384))

NON_REASONING_API_KEY = os.getenv("NON_REASONING_API_KEY")
NON_REASONING_BASE_URL = os.getenv("NON_REASONING_BASE_URL")
NON_REASONING_MODEL = os.getenv("NON_REASONING_MODEL", "gpt-4o-mini")
NON_REASONING_MODEL_CONTEXT = int(os.getenv("NON_REASONING_MODEL_CONTEXT", 16384))

GUARDRAIL_OPENAI_API_KEY = os.getenv("GUARDRAIL_OPENAI_API_KEY", REASONING_API_KEY)
GUARDRAIL_OPENAI_BASE_URL = os.getenv("GUARDRAIL_OPENAI_BASE_URL", REASONING_BASE_URL)
GUARDRAIL_OPENAI_MODEL = os.getenv("GUARDRAIL_OPENAI_MODEL", "gpt-4o-mini")
