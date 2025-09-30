# Test topic extraction functionality
import pytest
from app.services.topic_extraction import extract_topics, generate_summary

def test_extract_topics():
    """Should extract topics from text"""
    text = "Michael and Hanson are discussing a great topic on artificial intelligence, machine learning, and deep learning technologies."
    topics = extract_topics(text, top_n=3)
    
    assert isinstance(topics, list)
    assert len(topics) <= 3
    assert all(isinstance(topic, str) for topic in topics)

def test_extract_topics_edge_cases():
    """Should handle edge cases like empty/short text"""
    # Empty text
    topics = extract_topics("")
    assert isinstance(topics, list)
    
    # Short text
    topics = extract_topics("AI")
    assert isinstance(topics, list)

def test_generate_summary():
    """Should generate a summary"""
    transcript = """
    So let's talk about Mars missions. We have reusable ships and we're planning to launch several Starships to Mars at the end of next year. 
    The orbital synchronization happens every 26 months, so we need to launch in November or December. 
    We're going to try to land on Mars and see if we succeed.
    """
    topics = ["mars", "starships", "landing", "missions"]
    
    summary = generate_summary(transcript, topics)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(transcript)

def test_generate_summary_with_topics():
    """Should generate summary using topics"""
    transcript = """
    We discussed the Mars mission planning. The Starships will be launched next year. 
    Landing on Mars is the main challenge. We need to ensure safe landing procedures.
    """
    topics = ["mars", "starships", "landing"]
    
    summary = generate_summary(transcript, topics)
    
    # Summary should contain some of the topics
    summary_lower = summary.lower()
    topic_found = any(topic in summary_lower for topic in topics)
    assert topic_found

def test_generate_summary_edge_cases():
    """Should handle edge cases like empty/short text and no topics"""
    # Empty transcript
    summary = generate_summary("", [])
    assert summary == ""
    
    # Short transcript
    transcript = "Short text."
    topics = ["test"]
    summary = generate_summary(transcript, topics)
    assert summary == transcript
    
    # No topics
    transcript = "This is a longer conversation about various topics."
    topics = []
    summary = generate_summary(transcript, topics)
    assert isinstance(summary, str)
    assert len(summary) > 0

