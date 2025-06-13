from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="forbid")

    data_dir: str
    llm_model_name: str
    embedding_model_name: str
    google_api_key: str
    tavily_api_key: str

    @property
    def data_path(self) -> Path:
        return (BASE_DIR / self.data_dir).resolve()


settings = Settings()
