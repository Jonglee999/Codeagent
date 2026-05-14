"""FastAPI Main Application — entry point.

Current auth: Basic Auth via auth_middleware.authenticate_request.
"""

from fastapi import FastAPI

from backend.api.routes import router

app = FastAPI(title="Sample Full-Stack App", version="1.0.0")

app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to the Sample Full-Stack App"}
