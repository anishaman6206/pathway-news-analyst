import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # Neon
    database_url: str
    db_poll_interval: int = 3
    db_batch_size: int = 50
    debug_emit: bool = False

    # REST server
    host: str = "0.0.0.0"
    port: int = 8000

    # Azure OpenAI (same resource for embeddings + chat)
    azure_api_key: str = ""
    azure_api_base: str = ""
    azure_api_version: str = ""

    # Azure deployment names (NOT model names)
    azure_embeddings_deployment: str = ""
    azure_chat_deployment: str = ""
    min_retrieval_score: float = 0.0


def get_settings() -> Settings:
    load_dotenv()

    s = Settings(
        database_url=os.environ.get("DATABASE_URL", "").strip(),
        db_poll_interval=int(os.environ.get("DB_POLL_INTERVAL", "3")),
        db_batch_size=int(os.environ.get("DB_BATCH_SIZE", "50")),
        debug_emit=os.environ.get("DEBUG_EMIT", "0") == "1",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        azure_api_key=os.environ.get("AZURE_API_KEY", "").strip(),
        azure_api_base=os.environ.get("AZURE_API_BASE", "").strip(),
        azure_api_version=os.environ.get("AZURE_API_VERSION", "").strip(),
        azure_embeddings_deployment=os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "").strip(),
        azure_chat_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip(),
        min_retrieval_score=float(os.environ.get("MIN_RETRIEVAL_SCORE", "0.0")),
    )

    if not s.database_url:
        raise ValueError("Missing DATABASE_URL in environment/.env")

    return s
