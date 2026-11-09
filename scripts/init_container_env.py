"""Generate separate local-container credentials once; never print secrets."""
import secrets
from pathlib import Path

target = Path(__file__).resolve().parents[1] / ".env.container"
with target.open("x", encoding="utf-8") as out:
    out.write("JWT_SECRET=" + secrets.token_hex(32) + "\n")
    out.write("BOOTSTRAP_ADMIN_PASSWORD=" + secrets.token_urlsafe(24) + "\n")
    out.write("POSTGRES_PASSWORD=" + secrets.token_hex(24) + "\n")
    out.write("FINANCE_DISABLE_PAID_API=1\nAPI_PORT=8002\n")
print("Created .env.container; credentials not printed. Existing files are never overwritten.")
