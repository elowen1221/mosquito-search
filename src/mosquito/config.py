"""应用配置：从环境变量 / .env 读取。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """GITHUB_TOKEN 为可选：存在时使用认证请求，缺失时匿名访问并告警。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    github_token: str | None = None
    github_search_per_page: int = 50
    embedding_model: str = "BAAI/bge-small-zh-v1.5"


settings = Settings()