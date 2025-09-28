from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import psutil
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import spacy
from collections import Counter

router = APIRouter()

class TopicComparisonRequest(BaseModel):
    transcript: str
    top_n: int = 5

class MethodResult(BaseModel):
    method: str
    topics: List[str]
    time_sec: float
    memory_mb: float

class TopicComparisonResponse(BaseModel):
    results: List[MethodResult]
    summary: Dict[str, Any]

def get_memory_usage_mb():
    """Return current memory usage (MB) of this process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def extract_topics_tfidf(text: str, top_n: int = 5) -> List[str]:
    """Extract topics using TF-IDF"""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform([text])
    scores = X.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    term_score_pairs = list(zip(terms, scores))
    term_score_pairs.sort(key=lambda x: x[1], reverse=True)
    top_terms = [t for t, s in term_score_pairs[:top_n]]
    return top_terms

def extract_topics_spacy(text: str, top_n: int = 5) -> List[str]:
    """Extract topics using spaCy"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise HTTPException(status_code=500, detail="spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
    
    doc = nlp(text.lower())
    # Use nouns and proper nouns only
    candidates = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
    # Count frequency
    counts = Counter(candidates)
    top_keywords = [word for word, _ in counts.most_common(top_n)]
    return top_keywords

def extract_topics_keybert(text: str, top_n: int = 5, model_name: str = "all-MiniLM-L6-v2") -> List[str]:
    """Extract topics using KeyBERT"""
    kw_model = KeyBERT(model_name)
    # stop_words='english' removes common English stop words
    # use_mmr=True (Maximal Marginal Relevance) reduces repeated/similar keywords
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english', use_mmr=True)
    return [kw for kw, score in keywords]

def extract_topics_distilbert(text: str, top_n: int = 5, model_name: str = "distilbert-base-nli-stsb-mean-tokens") -> List[str]:
    """Extract topics using DistilBERT + Clustering"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    
    n_clusters = min(top_n, len(sentences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    
    labels = kmeans.labels_
    topics = []
    for i in range(n_clusters):
        cluster_sentences = [sentences[j] for j in range(len(sentences)) if labels[j]==i]
        topics.append(max(cluster_sentences, key=len))  # pick longest sentence
    return topics

def extract_topics_st_clustering(text: str, top_n: int = 5, model_name: str = "all-MiniLM-L6-v2") -> List[str]:
    """Extract topics using SentenceTransformer + Clustering"""
    # Split transcript into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    
    n_clusters = min(top_n, len(sentences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    topics = []
    for i in range(n_clusters):
        cluster_sentences = [sentences[j] for j in range(len(sentences)) if labels[j]==i]
        # pick the longest sentence as representative topic
        topics.append(max(cluster_sentences, key=len))
    return topics

@router.post("/", response_model=TopicComparisonResponse)
async def compare_topic_extraction_methods(request: TopicComparisonRequest):
    """
    Compare different topic extraction methods on the same transcript
    """
    try:
        results = []
        
        # TF-IDF
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        tfidf_topics = extract_topics_tfidf(request.transcript, request.top_n)
        tfidf_time = time.time() - start_time
        tfidf_mem = get_memory_usage_mb() - start_mem
        
        results.append(MethodResult(
            method="TF-IDF",
            topics=tfidf_topics,
            time_sec=round(tfidf_time, 3),
            memory_mb=round(tfidf_mem, 1)
        ))
        
        # spaCy
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        spacy_topics = extract_topics_spacy(request.transcript, request.top_n)
        spacy_time = time.time() - start_time
        spacy_mem = get_memory_usage_mb() - start_mem
        
        results.append(MethodResult(
            method="spaCy",
            topics=spacy_topics,
            time_sec=round(spacy_time, 3),
            memory_mb=round(spacy_mem, 1)
        ))
        
        # KeyBERT
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        keybert_topics = extract_topics_keybert(request.transcript, request.top_n)
        keybert_time = time.time() - start_time
        keybert_mem = get_memory_usage_mb() - start_mem
        
        results.append(MethodResult(
            method="KeyBERT",
            topics=keybert_topics,
            time_sec=round(keybert_time, 3),
            memory_mb=round(keybert_mem, 1)
        ))
        
        # DistilBERT + Clustering
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        distilbert_topics = extract_topics_distilbert(request.transcript, request.top_n)
        distilbert_time = time.time() - start_time
        distilbert_mem = get_memory_usage_mb() - start_mem
        
        results.append(MethodResult(
            method="DistilBERT+Clustering",
            topics=distilbert_topics,
            time_sec=round(distilbert_time, 3),
            memory_mb=round(distilbert_mem, 1)
        ))
        
        # SentenceTransformer + Clustering
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        st_topics = extract_topics_st_clustering(request.transcript, request.top_n)
        st_time = time.time() - start_time
        st_mem = get_memory_usage_mb() - start_mem
        
        results.append(MethodResult(
            method="ST+Clustering",
            topics=st_topics,
            time_sec=round(st_time, 3),
            memory_mb=round(st_mem, 1)
        ))
        
        # Generate summary
        total_time = sum(r.time_sec for r in results)
        total_memory = sum(r.memory_mb for r in results)
        
        summary = {
            "total_methods": len(results),
            "total_time_sec": round(total_time, 3),
            "total_memory_mb": round(total_memory, 1),
            "fastest_method": min(results, key=lambda x: x.time_sec).method,
            "most_memory_efficient": min(results, key=lambda x: x.memory_mb).method,
            "transcript_length": len(request.transcript),
            "top_n": request.top_n
        }
        
        return TopicComparisonResponse(results=results, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing topic methods: {str(e)}")

@router.get("/methods")
async def get_available_methods():
    """
    Get list of available topic extraction methods
    """
    return {
        "methods": [
            {
                "name": "TF-IDF",
                "description": "Fast statistical method using term frequency-inverse document frequency",
                "output_type": "Keywords (single words or short n-grams)",
                "pros": ["Very fast", "Low memory usage", "No external dependencies"],
                "cons": ["May select trivial words", "Ignores semantic context"]
            },
            {
                "name": "spaCy",
                "description": "Linguistic analysis using POS tagging and frequency counting",
                "output_type": "Nouns and proper nouns",
                "pros": ["Better than TF-IDF for conversational text", "Filters stop words well"],
                "cons": ["Still ignores semantic context", "Requires spaCy model"]
            },
            {
                "name": "KeyBERT",
                "description": "Semantic keyword extraction using BERT embeddings",
                "output_type": "Keywords and multi-word phrases",
                "pros": ["Captures semantic meaning", "Good balance of quality and speed"],
                "cons": ["Moderate memory usage", "Slower than statistical methods"]
            },
            {
                "name": "DistilBERT+Clustering",
                "description": "Semantic clustering using DistilBERT embeddings",
                "output_type": "Representative sentences from clusters",
                "pros": ["Excellent semantic clustering", "Coherent topic representation"],
                "cons": ["Sentence output not ideal for vectorization", "Higher memory usage"]
            },
            {
                "name": "SentenceTransformer+Clustering",
                "description": "Semantic clustering using SentenceTransformer embeddings",
                "output_type": "Representative sentences from clusters",
                "pros": ["Good semantic clustering", "Faster than DistilBERT"],
                "cons": ["Sentence output not ideal for vectorization", "Moderate memory usage"]
            }
        ]
    }
