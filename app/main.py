# FastAPI entrypoint
from fastapi import FastAPI
from app.routers import transcribe, summarise, match, topic_extraction_comparison, transcribe_summarise, health_check

app = FastAPI(title="ML Takehome API", version="1.0.0")

# Include routers
app.include_router(transcribe.router, prefix="/transcribe", tags=["transcribe"])
app.include_router(summarise.router, prefix="/summarise", tags=["summarise"])
app.include_router(transcribe_summarise.router, prefix="/transcribe-summarise", tags=["transcribe-summarise"])
app.include_router(match.router, prefix="/match", tags=["match"])
app.include_router(topic_extraction_comparison.router, prefix="/topic-extraction-comparison", tags=["topic-extraction-comparison"])
app.include_router(health_check.router, prefix="/health", tags=["health"])

@app.get("/")
async def root():
    return {"message": "ML Takehome API is running"}
