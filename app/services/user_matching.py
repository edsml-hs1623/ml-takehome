# Cosine similarity + interpretation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def compute_compatibility(user1_data: dict, user2_data: dict, topics: list[str] = None, 
                         topic_weight: float = 0.5, psych_weight: float = 1.0) -> tuple[float, str]:
    """
    Compute compatibility score using cosine similarity between combined topic and psychometric profiles
    with comprehensive edge-case handling
    """
    # Edge case: Same user
    if user1_data["id"] == user2_data["id"]:
        return 1.0, "Identical users (perfect match)"
    
    # Edge case: Validate psychometric data
    try:
        user1_psych_raw = user1_data["psychometrics"]
        user2_psych_raw = user2_data["psychometrics"]
    except KeyError as e:
        raise ValueError(f"Missing psychometric data: {e}")
    
    # Edge case: Handle empty psychometric data
    if not user1_psych_raw or not user2_psych_raw:
        return 0.0, "No psychometric data available"
    
    # Normalize and resample psychometric data
    user1_psych = normalize_psychometrics(user1_psych_raw)
    user2_psych = normalize_psychometrics(user2_psych_raw)
    
    # Resample to ensure same dimensions
    target_length = max(len(user1_psych), len(user2_psych), 5)  # At least 5 dimensions
    user1_psych = resample_psychometrics(user1_psych, target_length)
    user2_psych = resample_psychometrics(user2_psych, target_length)
    
    # Edge case: Check for all-zero vectors (would cause division by zero in cosine similarity)
    if np.all(user1_psych == 0) or np.all(user2_psych == 0):
        return 0.0, "No psychometric data available"
    
    # Edge case: Validate weights
    if topic_weight < 0 or psych_weight < 0:
        raise ValueError("Weights must be non-negative")
    
    # If topics are provided, vectorize them and combine with psychometrics
    if topics and len(topics) > 0:
        # Edge case: Filter out empty topics
        topics = [topic.strip() for topic in topics if topic.strip()]
        
        if not topics:
            # Fallback to psychometric-only if no valid topics
            score = cosine_similarity([user1_psych], [user2_psych])[0][0]
        else:
            # Vectorize topics using TF-IDF
            topic_vector = vectorize_topics(topics)
            
            # Edge case: Check if topic vector is all zeros
            if np.all(topic_vector == 0):
                # Fallback to psychometric-only if topics don't vectorize properly
                score = cosine_similarity([user1_psych], [user2_psych])[0][0]
            else:
                # Combine topic and psychometric vectors with weights
                user1_combined = combine_vectors(topic_vector, user1_psych, topic_weight, psych_weight)
                user2_combined = combine_vectors(topic_vector, user2_psych, topic_weight, psych_weight)
                
                # Compute cosine similarity on combined vectors
                score = cosine_similarity([user1_combined], [user2_combined])[0][0]
    else:
        # Fallback to psychometric-only comparison
        score = cosine_similarity([user1_psych], [user2_psych])[0][0]
    
    # Edge case: Handle NaN or infinite scores
    if np.isnan(score) or np.isinf(score):
        return 0.0, "Unable to compute compatibility score"
    
    # Clamp score to valid range [0, 1]
    score = np.clip(score, 0.0, 1.0)
    
    # Generate interpretation with enhanced thresholds
    interpretation = interpret_score(score)
    
    return float(score), interpretation

def vectorize_topics(topics: list[str]) -> np.ndarray:
    """
    Vectorize topics using TF-IDF
    """
    text = " ".join(topics)
    vectorizer = TfidfVectorizer()
    vec = vectorizer.fit_transform([text])
    return vec.toarray()[0]

def combine_vectors(topic_vec: np.ndarray, psych_vec: np.ndarray, 
                   topic_weight: float = 1.0, psych_weight: float = 1.0) -> np.ndarray:
    """
    Combine topic and psychometric vectors with specified weights
    """
    topic_vec = np.array(topic_vec)
    psych_vec = np.array(psych_vec)
    
    # Scale vectors with weights and concatenate
    combined = np.concatenate([topic_weight * topic_vec, psych_weight * psych_vec])
    return combined

def interpret_score(score: float) -> str:
    """
    Generate natural language interpretation of compatibility score with enhanced thresholds
    """
    if score >= 0.9:
        return "Exceptionally compatible - Perfect match"
    elif score >= 0.8:
        return "Highly compatible - Strong match"
    elif score >= 0.7:
        return "Very compatible - Good match"
    elif score >= 0.6:
        return "Moderately compatible - Decent match"
    elif score >= 0.4:
        return "Somewhat compatible - Weak match"
    elif score >= 0.2:
        return "Low compatibility - Poor match"
    else:
        return "Very low compatibility - Minimal match"

def resample_psychometrics(psychometric_data: list[float], target_length: int = 5) -> np.ndarray:
    """
    Resample psychometric data to target length using interpolation
    Handles cases where psychometric vectors have different lengths
    """
    psych_array = np.array(psychometric_data)
    current_length = len(psych_array)
    
    if current_length == target_length:
        return psych_array
    
    if current_length == 0:
        # Return neutral values if no data
        return np.full(target_length, 0.5)
    
    if current_length == 1:
        # Replicate single value
        return np.full(target_length, psych_array[0])
    
    # Use linear interpolation to resample
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    resampled = np.interp(x_new, x_old, psych_array)
    return resampled

def normalize_psychometrics(psychometric_data: list[float]) -> np.ndarray:
    """
    Normalize psychometric data to [0, 1] range
    Handles cases where data might be outside expected range
    """
    psych_array = np.array(psychometric_data)
    
    # Handle all same values
    if np.all(psych_array == psych_array[0]):
        return np.full_like(psych_array, 0.5)
    
    # Min-max normalization to [0, 1]
    min_val = np.min(psych_array)
    max_val = np.max(psych_array)
    
    if max_val == min_val:
        return np.full_like(psych_array, 0.5)
    
    normalized = (psych_array - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)

