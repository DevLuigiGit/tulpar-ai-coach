"""`python -m tulpar_ai` — run the service (API + web + bot)."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("tulpar_ai.api.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8089")),
                log_level="info")
