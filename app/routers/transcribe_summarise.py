from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.transcription import transcribe_audio
from app.services.topic_extraction import extract_topics, generate_summary

router = APIRouter()

class TranscribeSummariseResponse(BaseModel):
    transcript: str
    topics: list[str]
    summary: str

@router.post("/", response_model=TranscribeSummariseResponse)
async def transcribe_and_summarise(audio_file: UploadFile = File(...)):
    """
    Transcribe audio file to text and generate summary with topics
    """
    try:
        # Transcribe audio
        transcript = transcribe_audio(audio_file)
        
        # Extract topics
        topics = extract_topics(transcript)
        
        # Generate summary
        summary = generate_summary(transcript, topics)
        
        return {
            "transcript": transcript,
            "topics": topics,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
