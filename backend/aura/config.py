"""Aura Configuration System using Pydantic Settings"""
import os
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application configuration using Pydantic Settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # File constraints
    MAX_UPLOAD_MB: float = Field(default=150, ge=0)
    MAX_DURATION_MIN: float = Field(default=15, ge=0)

    # API Keys
    OPENAI_API_KEY: SecretStr = Field(default=SecretStr(""), description="OpenAI API key for AI agent")
    GEMINI_API_KEY: SecretStr = Field(default=SecretStr(""), description="Google Gemini API key for AI agent")
    llm_provider: str = Field("openai", env="LLM_PROVIDER")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # Audio processing
    DEFAULT_SR: int = Field(default=44100, ge=8000, le=192000)

    # Analysis settings
    SEGMENT_LENGTH_SEC: float = Field(default=4.0, gt=0.0, le=30.0)
    SEGMENT_OVERLAP_RATIO: float = Field(default=0.5, ge=0.0, lt=1.0)

    # Worker settings
    MAX_RAM_MB_WORKER: int = Field(default=1024, ge=128, le=8192, description="Maximum RAM in MB for worker processes")
    STATUS_DIR: str = Field(default="worker_status", description="Directory for worker status files")

    # Directory settings
    UPLOAD_DIR: str = Field(default="uploads", description="Directory for uploaded files")
    DOWNLOAD_DIR: str = Field(default="downloads", description="Directory for processed files")
    TEMP_DIR: str = Field(default="temp", description="Directory for temporary files")
    LOG_DIR: str = Field(default="logs", description="Directory for log files")


# Global config instance
app_config = AppConfig()

