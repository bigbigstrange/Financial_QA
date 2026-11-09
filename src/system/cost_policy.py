"""One explicit switch for pausing project paid-model operations."""
import os
from pathlib import Path
from dotenv import dotenv_values


def paid_api_disabled():
    value = os.getenv("FINANCE_DISABLE_PAID_API")
    if value is None:
        value = dotenv_values(Path(__file__).resolve().parents[2] / ".env").get("FINANCE_DISABLE_PAID_API", "0")
    return str(value).strip().lower() in {"1", "true", "yes"}
