# Test vectorization functionality
import pytest
import numpy as np
from app.services.vectorization import vectorize_topics, vectorize_psychometrics, fuse_vectors

def test_vectorize_topics():
    """Should vectorize topics"""
    topics = ["artificial intelligence", "machine learning", "deep learning"]
    vector = vectorize_topics(topics)
    
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

def test_vectorize_psychometrics():
    """Should vectorize psychometric data"""
    psychometric_data = [0.5, 0.6, 0.7, 0.8, 0.9]
    vector = vectorize_psychometrics(psychometric_data)
    
    assert isinstance(vector, np.ndarray)
    assert len(vector) == len(psychometric_data)
    assert np.array_equal(vector, np.array(psychometric_data))

def test_fuse_vectors():
    """Should fuse topic and psychometric vectors"""
    topic_vector = np.array([0.1, 0.2, 0.3])
    psychometric_vector = np.array([0.4, 0.5, 0.6])
    
    fused = fuse_vectors(topic_vector, psychometric_vector)
    
    assert isinstance(fused, np.ndarray)
    assert len(fused) == len(topic_vector) + len(psychometric_vector)
    assert np.array_equal(fused[:3], topic_vector)
    assert np.array_equal(fused[3:], psychometric_vector)

def test_vectorization_edge_cases():
    """Should handle edge cases"""
    # Single topic
    topics = ["technology"]
    vector = vectorize_topics(topics)
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    
    # Empty psychometric data
    psychometric_data = []
    vector = vectorize_psychometrics(psychometric_data)
    assert isinstance(vector, np.ndarray)
    assert len(vector) == 0
    
    # Different length vectors
    topic_vector = np.array([0.1, 0.2])
    psychometric_vector = np.array([0.3, 0.4, 0.5, 0.6])
    fused = fuse_vectors(topic_vector, psychometric_vector)
    assert isinstance(fused, np.ndarray)
    assert len(fused) == 6