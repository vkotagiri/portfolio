# app/config.py
from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_sqlite_url() -> str:
    db_path = (Path(__file__).resolve().parents[1] / "portfolio.db").as_posix()
    return f"sqlite:///{db_path}"


class Settings(BaseSettings):
    """
    Central app configuration.
    - Reads from .env
    - Accepts BOTH UPPERCASE and lowercase env names (AliasChoices)
    - Ignores unknown extras so new keys in .env won't crash
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Database ----
    database_url: str = Field(
        default_factory=_default_sqlite_url,
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
    )

    # ---- App basics ----
    port: int = Field(8000, validation_alias=AliasChoices("PORT", "port"))
    env: str = Field("dev", validation_alias=AliasChoices("ENV", "env"))
    timezone: str = Field(
        "America/New_York", validation_alias=AliasChoices("TIMEZONE", "timezone")
    )
    offline: bool = Field(False, validation_alias=AliasChoices("OFFLINE", "offline"))

    # ---- API keys (optional) ----
    tiingo_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("TIINGO_API_KEY", "tiingo_api_key")
    )
    alphavantage_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ALPHAVANTAGE_API_KEY", "alphavantage_api_key"),
    )
    fred_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("FRED_API_KEY", "fred_api_key")
    )
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key")
    )

    # ---- Tiingo rate-limit knobs ----
    tiingo_hourly: int = Field(
        50, validation_alias=AliasChoices("TIINGO_HOURLY", "tiingo_hourly")
    )
    tiingo_daily: int = Field(
        1000, validation_alias=AliasChoices("TIINGO_DAILY", "tiingo_daily")
    )
    tiingo_unique_month: int = Field(
        500, validation_alias=AliasChoices("TIINGO_UNIQUE_MONTH", "tiingo_unique_month")
    )
    tiingo_rl_mode: Literal["sleep", "strict", "off"] = Field(
        "sleep", validation_alias=AliasChoices("TIINGO_RL_MODE", "tiingo_rl_mode")
    )
    tiingo_max_sleep: int = Field(
        120, validation_alias=AliasChoices("TIINGO_MAX_SLEEP", "tiingo_max_sleep")
    )
    tiingo_max_total_sleep: int = Field(
        900,
        validation_alias=AliasChoices("TIINGO_MAX_TOTAL_SLEEP", "tiingo_max_total_sleep"),
    )
    tiingo_debug: bool = Field(
        False, validation_alias=AliasChoices("TIINGO_DEBUG", "tiingo_debug")
    )

    # ---- Yahoo Finance (yfinance) throttling ----
    yf_rpm: int = Field(
        240, validation_alias=AliasChoices("YF_RPM", "yf_rpm"),
        description="Soft throttle for yfinance requests/min."
    )
    yf_burst: int = Field(
        10, validation_alias=AliasChoices("YF_BURST", "yf_burst"),
        description="Approx. burst tokens before backoff."
    )
    yf_parallel: int = Field(
        5, validation_alias=AliasChoices("YF_PARALLEL", "yf_parallel"),
        description="Max concurrent yfinance requests."
    )
    yf_backoff_base: float = Field(
        1.5, validation_alias=AliasChoices("YF_BACKOFF_BASE", "yf_backoff_base"),
        description="Exponential backoff base for retries."
    )
    yf_debug: bool = Field(
        False, validation_alias=AliasChoices("YF_DEBUG", "yf_debug")
    )

    # ---- Alpha Vantage throttling (free-tier friendly) ----
    alphavantage_rpm: int = Field(
        5, validation_alias=AliasChoices("ALPHAVANTAGE_RPM", "av_rpm")
    )
    alphavantage_rpd: int = Field(
        25, validation_alias=AliasChoices("ALPHAVANTAGE_RPD", "av_daily")
    )
    alphavantage_burst: int = Field(
        5, validation_alias=AliasChoices("ALPHAVANTAGE_BURST", "av_burst")
    )
    av_debug: bool = Field(
        False, validation_alias=AliasChoices("ALPHAVANTAGE_DEBUG", "av_debug")
    )

    # ---- Email Notifications ----
    smtp_host: str = Field(
        "smtp.gmail.com", validation_alias=AliasChoices("SMTP_HOST", "smtp_host")
    )
    smtp_port: int = Field(
        587, validation_alias=AliasChoices("SMTP_PORT", "smtp_port")
    )
    smtp_user: str | None = Field(
        default=None, validation_alias=AliasChoices("SMTP_USER", "smtp_user")
    )
    smtp_password: str | None = Field(
        default=None, validation_alias=AliasChoices("SMTP_PASSWORD", "smtp_password")
    )
    email_to: str | None = Field(
        default=None, validation_alias=AliasChoices("EMAIL_TO", "email_to"),
        description="Recipient email for portfolio notifications"
    )
    email_from: str | None = Field(
        default=None, validation_alias=AliasChoices("EMAIL_FROM", "email_from"),
        description="Sender email (defaults to SMTP_USER)"
    )
    smtp_use_tls: bool = Field(
        True, validation_alias=AliasChoices("SMTP_USE_TLS", "smtp_use_tls"),
        description="Use STARTTLS (True) or SSL (False)"
    )
    report_base_url: str | None = Field(
        default=None, 
        validation_alias=AliasChoices("REPORT_BASE_URL", "report_base_url"),
        description="Base URL for report links in emails (e.g., http://localhost:8080)"
    )

settings = Settings()
