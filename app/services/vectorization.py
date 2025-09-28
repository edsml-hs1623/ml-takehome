# TF-IDF / embeddings / one-hot
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def vectorize_topics(topics: list[str]) -> np.ndarray:
    """
    Vectorize topics using TF-IDF
    """
    text = " ".join(topics)
    vectorizer = TfidfVectorizer()
    vec = vectorizer.fit_transform([text])
    return vec.toarray()[0]

def vectorize_psychometrics(psychometric_data: dict) -> np.ndarray:
    """
    Vectorize psychometric profile data
    """
    return np.array(psychometric_data)

def fuse_vectors(topic_vector: np.ndarray, psychometric_vector: np.ndarray) -> np.ndarray:
    """
    Fuse topic and psychometric vectors
    """
    return np.concatenate([topic_vector, psychometric_vector])

