# POST /match
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.user_matching import compute_compatibility
from app.utils.io_utils import load_users

router = APIRouter()

class MatchRequest(BaseModel):
    user1_id: str
    user2_id: str
    topics: Optional[list[str]] = None
    topic_weight: Optional[float] = 0.5
    psych_weight: Optional[float] = 1.0

class MatchResponse(BaseModel):
    score: float
    interpretation: str

@router.post("/", response_model=MatchResponse)
async def match(request: MatchRequest):
    """
    Compute compatibility score between two users
    Optionally include topics for enhanced matching with weighted combination
    """
    try:
        users = load_users("sample_data/synthetic_users.json")
        
        # Edge case: Validate user IDs exist
        if request.user1_id not in users:
            raise ValueError(f"User {request.user1_id} not found")
        if request.user2_id not in users:
            raise ValueError(f"User {request.user2_id} not found")
        
        # Edge case: Validate weights are reasonable
        if request.topic_weight > 10 or request.psych_weight > 10:
            raise ValueError("Weights too high (max 10)")
        
        score, interpretation = compute_compatibility(
            users[request.user1_id], 
            users[request.user2_id],
            topics=request.topics,
            topic_weight=request.topic_weight,
            psych_weight=request.psych_weight
        )
        
        return {"score": score, "interpretation": interpretation}
        
    except ValueError as e:
        # Re-raise validation errors with proper HTTP status
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected errors
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

