# POST /summarise
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.topic_extraction import extract_topics, generate_summary

router = APIRouter()

class SummariseRequest(BaseModel):
    transcript: str

class SummariseResponse(BaseModel):
    topics: list[str]
    summary: str

@router.post("/", response_model=SummariseResponse)
async def summarise(request: SummariseRequest):
    """
    Extract topics and generate summary from transcript
    """
    topics = extract_topics(request.transcript)
    summary = generate_summary(request.transcript, topics)
    return {"topics": topics, "summary": summary}

