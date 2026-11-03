"""HTTP API. The /score endpoint arrives in week 3."""

from fastapi import FastAPI

app = FastAPI(title="Fraud Risk Engine", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
