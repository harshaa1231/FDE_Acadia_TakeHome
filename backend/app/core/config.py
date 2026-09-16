from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider (Groq free-tier by default; interface is provider-agnostic)
    groq_api_key: str = ""
    sql_model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-20b"
    llm_request_timeout_s: float = 30.0

    # Token usage is always tracked and reported per job (real, measured).
    # A $/1k-token estimate is only computed if you set these to your
    # actual negotiated rate for each model - left at 0 (unset) by
    # default rather than guessing a price, since Groq's published rates
    # change and differ by tier.
    sql_model_cost_per_1k_tokens: float = 0.0
    fast_model_cost_per_1k_tokens: float = 0.0

    # Data / query limits
    data_dir: str = "./data_store"
    max_upload_mb: int = 200
    max_result_rows: int = 500
    sql_execution_timeout_s: float = 15.0
    sql_max_retries: int = 1

    # Async job engine
    job_worker_concurrency: int = 4
    job_queue_max_size: int = 100

    # Sampling used to build schema profile / concept map
    profile_sample_rows: int = 20
    profile_max_distinct_scan: int = 1000

    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


settings = Settings()
