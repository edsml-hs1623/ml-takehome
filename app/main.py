# FastAPI entrypoint
from fastapi import FastAPI
from app.routers import transcribe, summarise, match

app = FastAPI(title="ML Takehome API", version="1.0.0")

# Include routers
app.include_router(transcribe.router, prefix="/transcribe", tags=["transcribe"])
app.include_router(summarise.router, prefix="/summarise", tags=["summarise"])
app.include_router(match.router, prefix="/match", tags=["match"])

@app.get("/")
async def root():
    return {"message": "ML Takehome API is running"}
