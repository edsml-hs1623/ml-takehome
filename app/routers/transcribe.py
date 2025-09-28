# POST /transcribe
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.services.transcription import transcribe_audio
from app.services.topic_extraction import extract_topics, generate_summary

router = APIRouter()

class TranscribeResponse(BaseModel):
    transcript: str

class TranscribeAndSummarizeResponse(BaseModel):
    transcript: str
    topics: list[str]
    summary: str

@router.post("/", response_model=TranscribeResponse)
async def transcribe(audio_file: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper
    """
    transcript = transcribe_audio(audio_file)
    return {"transcript": transcript}
