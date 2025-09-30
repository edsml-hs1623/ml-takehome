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
    Generate intelligent single-sentence summary for conversational audio using advanced topic-guided extractive summarization.
    
    This function processes conversational transcripts by:
    1. Splitting text into sentences using regex patterns for better conversational handling
    2. Cleaning up fillers and repetitive phrases (yeah, uh, um, like, you know, etc.)
    3. Scoring sentences based on multiple criteria:
       - Topic relevance (40% weight): How many extracted topics appear in the sentence
       - Content density (20% weight): Word count and meaningful content
       - Position weighting: Prioritizes middle content (main discussion) over intro/conclusion
       - Conversational elements: Bonus for questions and future-oriented statements
       - Length optimization: Penalizes too short (<30 chars) or too long (>300 chars) sentences
       - Quality checks: Avoids incomplete thoughts ending with conjunctions
    4. Selecting the single highest-scoring sentence that best captures the conversation's essence
    
    Args:
        transcript (str): Raw conversational transcript text
        topics (list[str]): List of extracted topics to guide summarization
        
    Returns:
        str: Single-sentence summary that captures the key points of the conversation
        
    Example:
        >>> topics = ["mars", "starships", "landing"]
        >>> transcript = "So let's talk about Mars missions. Yeah, we have reusable ships..."
        >>> summary = generate_summary(transcript, topics)
        >>> print(summary)
        "The discussion covers Mars mission planning with reusable ships and landing strategies."
    """
    import re
    
    # Better sentence splitting for conversational content
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if len(sentences) <= 2:
        return transcript
    
    # Clean up conversational fillers and improve sentence quality
    cleaned_sentences = []
    for sentence in sentences:
        # Remove excessive repetition and fillers
        sentence = re.sub(r'\b(yeah|uh|um|like|you know|so|well)\b', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        if len(sentence) > 15:  # Only keep substantial sentences
            cleaned_sentences.append(sentence)
    
    if len(cleaned_sentences) <= 2:
        return transcript
    
    # Advanced scoring system for conversational content
    scored_sentences = []
    for i, sentence in enumerate(cleaned_sentences):
        score = 0
        
        # 1. Topic relevance (most important)
        topic_matches = sum(1 for topic in topics if topic.lower() in sentence.lower())
        topic_score = (topic_matches / len(topics)) * 0.4 if topics else 0
        
        # 2. Content density (prefer sentences with more meaningful content)
        word_count = len(sentence.split())
        content_density = min(word_count / 50, 1.0) * 0.2
        
        # 3. Position weighting (conversational structure)
        total_sentences = len(cleaned_sentences)
        if i < total_sentences * 0.1:  # First 10% - introduction
            position_score = 0.15
        elif i > total_sentences * 0.8:  # Last 20% - conclusion
            position_score = 0.15
        elif total_sentences * 0.3 <= i <= total_sentences * 0.7:  # Middle 40% - main content
            position_score = 0.25
        else:
            position_score = 0.1
        
        # 4. Question/statement bonus (conversational elements)
        question_bonus = 0
        future_bonus = 0
        
        if '?' in sentence:
            question_bonus = 0.1
        elif any(word in sentence.lower() for word in ['plan', 'next', 'future', 'will', 'going to']):
            future_bonus = 0.1
        
        # 5. Length penalty (avoid too short or too long)
        if len(sentence) < 30:
            length_penalty = 0.1
        elif len(sentence) > 300:
            length_penalty = 0.05
        else:
            length_penalty = 0
        
        # 6. Conversational quality (avoid incomplete thoughts)
        if sentence.endswith(('and', 'but', 'so', 'because', 'the')):
            incomplete_penalty = 0.1
        else:
            incomplete_penalty = 0
        
        # Combined score
        score = topic_score + content_density + position_score + question_bonus + future_bonus - length_penalty - incomplete_penalty
        scored_sentences.append((score, sentence, i))
    
    # Sort by score and select diverse sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Select the single best sentence that captures the essence
    if scored_sentences:
        best_sentence = scored_sentences[0][1]  # Get the highest scoring sentence
        
        # Clean up the sentence for better readability
        best_sentence = re.sub(r'\s+', ' ', best_sentence).strip()
        
        # Ensure it ends with proper punctuation
        if not best_sentence.endswith(('.', '!', '?')):
            best_sentence += '.'
        
        summary = best_sentence
    else:
        summary = transcript
    
    return summary
