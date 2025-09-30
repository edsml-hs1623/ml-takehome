# Test user matching functionality
import pytest
from app.services.user_matching import compute_compatibility, interpret_score

def test_same_user_compatibility():
    """Same user should have perfect compatibility"""
    user_data = {
        "id": "user_1",
        "psychometrics": [0.5, 0.6, 0.7, 0.8, 0.9]
    }
    
    score, interpretation = compute_compatibility(user_data, user_data)
    assert score == 1.0
    assert "Identical users" in interpretation

def test_different_users_compatibility():
    """Different users should have some compatibility score"""
    user1_data = {
        "id": "user_1",
        "psychometrics": [0.5, 0.6, 0.7, 0.8, 0.9]
    }
    user2_data = {
        "id": "user_2", 
        "psychometrics": [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    
    score, interpretation = compute_compatibility(user1_data, user2_data)
    assert 0.0 <= score <= 1.0
    assert isinstance(interpretation, str)

def test_compatibility_with_topics():
    """Compatibility should work with topics"""
    user1_data = {
        "id": "user_1",
        "psychometrics": [0.5, 0.6, 0.7, 0.8, 0.9]
    }
    user2_data = {
        "id": "user_2",
        "psychometrics": [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    topics = ["technology", "science", "innovation"]
    
    score, interpretation = compute_compatibility(
        user1_data, user2_data, topics=topics
    )
    assert 0.0 <= score <= 1.0
    assert isinstance(interpretation, str)

def test_missing_psychometrics_error():
    """Should raise error when psychometrics are missing"""
    user1_data = {"id": "user_1"}
    user2_data = {"id": "user_2", "psychometrics": [0.5, 0.6, 0.7]}
    
    with pytest.raises(ValueError):
        compute_compatibility(user1_data, user2_data)

def test_empty_psychometrics():
    """Should handle empty psychometric data"""
    user1_data = {"id": "user_1", "psychometrics": []}
    user2_data = {"id": "user_2", "psychometrics": []}
    
    score, interpretation = compute_compatibility(user1_data, user2_data)
    assert score == 0.0
    assert "No psychometric data available" in interpretation

def test_interpret_score():
    """Score interpretation should work for different ranges"""
    assert "Exceptionally compatible" in interpret_score(0.95)
    assert "Highly compatible" in interpret_score(0.85)
    assert "Very compatible" in interpret_score(0.75)
    assert "Moderately compatible" in interpret_score(0.65)
    assert "Somewhat compatible" in interpret_score(0.45)
    assert "Low compatibility" in interpret_score(0.25)
    assert "Very low compatibility" in interpret_score(0.05)

