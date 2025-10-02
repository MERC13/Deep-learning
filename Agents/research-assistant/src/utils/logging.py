import logging
import os
from typing import Optional


def _level_from_env(default: str = "INFO") -> int:
    level = os.getenv("LOG_LEVEL", default).upper()
    return getattr(logging, level, logging.INFO)


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once.

    Env:
      - LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default INFO)
      - LOG_FORMAT: 'plain' (default) or 'json'
    """
    if getattr(configure_logging, "_configured", False):
        return
    fmt_plain = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    fmt_json = "{\"time\":\"%(asctime)s\",\"level\":\"%(levelname)s\",\"name\":\"%(name)s\",\"msg\":%(message)s}"
    use_json = os.getenv("LOG_FORMAT", "plain").lower() == "json"
    logging.basicConfig(level=_level_from_env(level or "INFO"), format=fmt_json if use_json else fmt_plain)
    configure_logging._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
