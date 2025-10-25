import logging
from typing import Optional

_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger with consistent formatting.

    If not yet configured, configures the root logger once.
    """
    _configure_root_logger()
    return logging.getLogger(name or "app")
