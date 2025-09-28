# KeyBERT / embeddings
from keybert import KeyBERT

def extract_topics(transcript: str, top_n=5, model="all-MiniLM-L6-v2") -> list[str]:
    """
    Extract topics from text using KeyBERT
    """
    kw_model = KeyBERT(model)
    topics = kw_model.extract_keywords(transcript, top_n=top_n, stop_words="english")
    return [t[0] for t in topics]

def generate_summary(transcript: str, topics: list[str]) -> str:
    """
    Generate summary based on extracted topics using topic-guided extractive summarization
    """
    sentences = [s.strip() for s in transcript.split('.') if s.strip()]
    
    if len(sentences) <= 2:
        return transcript
    
    # Score sentences based on topic relevance and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        
        # Topic relevance score (higher if sentence contains more topics)
        topic_matches = sum(1 for topic in topics if topic.lower() in sentence.lower())
        topic_score = topic_matches / len(topics) if topics else 0
        
        # Position score (slight bias toward beginning and end)
        if i == 0:  # First sentence
            position_score = 0.3
        elif i == len(sentences) - 1:  # Last sentence
            position_score = 0.2
        else:
            position_score = 0.1
        
        # Length penalty (prefer medium-length sentences)
        length_penalty = 0.1 if len(sentence) < 20 or len(sentence) > 200 else 0
        
        # Combined score
        score = topic_score + position_score - length_penalty
        scored_sentences.append((score, sentence, i))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Take 2-3 best sentences, maintaining original order
    selected_indices = sorted([item[2] for item in scored_sentences[:3]])
    summary_sentences = [sentences[i] for i in selected_indices]
    
    # Join and clean up
    summary = '. '.join(summary_sentences)
    if not summary.endswith('.'):
        summary += '.'
    
    return summary
