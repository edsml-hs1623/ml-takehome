# Integration tests for complete workflows
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_complete_workflow_transcript_to_matching():
    """Should complete workflow: transcript -> topics -> summary -> user matching"""
    # Step 1: Summarize transcript
    transcript = "We discussed artificial intelligence and machine learning technologies for data analysis."
    summarise_response = client.post("/summarise/", json={"transcript": transcript})
    
    assert summarise_response.status_code == 200
    summarise_data = summarise_response.json()
    topics = summarise_data["topics"]
    summary = summarise_data["summary"]
    
    assert isinstance(topics, list)
    assert isinstance(summary, str)
    assert len(topics) > 0
    assert len(summary) > 0
    
    # Step 2: Use extracted topics for user matching
    match_response = client.post("/match/", json={
        "user1_id": "user_1",
        "user2_id": "user_2",
        "topics": topics
    })
    
    assert match_response.status_code == 200
    match_data = match_response.json()
    assert "score" in match_data
    assert "interpretation" in match_data
    assert 0.0 <= match_data["score"] <= 1.0

