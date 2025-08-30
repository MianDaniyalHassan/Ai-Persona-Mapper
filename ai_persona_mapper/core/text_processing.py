# ai_persona_mapper/core/text_processing.py
"""
Text processing utilities for AI Persona Mapper.
Handles tokenization, cleaning, and basic NLP without external libraries.
"""

import re
import string
from typing import List, Tuple, Dict, Optional
from collections import Counter
import pandas as pd


def load_stopwords(filepath: str = "assets/stopwords.txt") -> set:
    """
    Load stopwords from file. Falls back to default list if file not found.
    
    Args:
        filepath: Path to stopwords file
        
    Returns:
        Set of stopword strings
    """
    default_stopwords = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
        'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
        'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
        'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
        'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
        'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
        'were', 'said', 'did', 'getting', 'made', 'find', 'where', 'much', 'too',
        'very', 'still', 'being', 'going', 'why', 'before', 'never', 'here', 'more'
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = set(line.strip().lower() for line in f if line.strip())
            return loaded if loaded else default_stopwords
    except FileNotFoundError:
        return default_stopwords


def clean_text(text: str) -> str:
    """
    Clean text while preserving emojis and meaningful punctuation.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Preserve emojis by temporarily replacing them
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    emojis = emoji_pattern.findall(text)
    text = emoji_pattern.sub(' EMOJI ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.!?,\'-]', ' ', text)
    
    # Restore emojis
    for emoji in emojis:
        text = text.replace(' EMOJI ', f' {emoji} ', 1)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize(text: str, exclude_stopwords: bool = True) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Cleaned text string
        exclude_stopwords: Whether to filter out stopwords
        
    Returns:
        List of token strings
    """
    # Basic word tokenization
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    
    # Filter tokens
    tokens = [t for t in tokens if len(t) > 1 or t in '!?.,']
    
    if exclude_stopwords:
        stopwords = load_stopwords()
        tokens = [t for t in tokens if t not in stopwords]
    
    return tokens


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic punctuation rules.
    
    Args:
        text: Input text
        
    Returns:
        List of sentence strings
    """
    # Simple sentence splitting by punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def compute_lexical_stats(tokens: List[str], original_text: str) -> Dict:
    """
    Compute lexical statistics from tokens and text.
    
    Args:
        tokens: List of tokens
        original_text: Original input text
        
    Returns:
        Dictionary of lexical statistics
    """
    # Basic stats
    vocab_size = len(set(tokens))
    total_tokens = len(tokens)
    type_token_ratio = vocab_size / total_tokens if total_tokens > 0 else 0
    
    # Word lengths (excluding punctuation)
    word_tokens = [t for t in tokens if t.isalnum()]
    avg_word_length = sum(len(t) for t in word_tokens) / len(word_tokens) if word_tokens else 0
    
    # Count exclamations and questions
    exclamation_count = original_text.count('!')
    question_count = original_text.count('?')
    
    # Sentence stats
    sentences = split_sentences(original_text)
    avg_sentence_length = len(tokens) / len(sentences) if sentences else 0
    
    return {
        'vocab_size': vocab_size,
        'total_tokens': total_tokens,
        'type_token_ratio': type_token_ratio,
        'avg_word_length': avg_word_length,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'sentence_count': len(sentences),
        'avg_sentence_length': avg_sentence_length
    }


def extract_bigrams(tokens: List[str], top_n: int = 8) -> pd.DataFrame:
    """
    Extract and count bigrams from token list.
    
    Args:
        tokens: List of tokens
        top_n: Number of top bigrams to return
        
    Returns:
        DataFrame with bigram and count columns
    """
    if len(tokens) < 2:
        return pd.DataFrame(columns=['bigram', 'count'])
    
    # Generate bigrams
    bigrams = []
    for i in range(len(tokens) - 1):
        # Skip if either token is punctuation
        if tokens[i].isalnum() and tokens[i+1].isalnum():
            bigrams.append(f"{tokens[i]} {tokens[i+1]}")
    
    # Count and sort
    bigram_counts = Counter(bigrams)
    top_bigrams = bigram_counts.most_common(top_n)
    
    # Create DataFrame
    df = pd.DataFrame(top_bigrams, columns=['bigram', 'count'])
    return df


def process_text(text: str, exclude_stopwords: bool = True) -> Tuple[List[str], List[str], Dict]:
    """
    Main text processing pipeline.
    
    Args:
        text: Raw input text
        exclude_stopwords: Whether to exclude stopwords
        
    Returns:
        Tuple of (tokens, sentences, lexical_stats)
    """
    # Clean the text
    cleaned = clean_text(text)
    
    # Tokenize
    tokens = tokenize(cleaned, exclude_stopwords)
    
    # Split sentences
    sentences = split_sentences(text)
    
    # Compute stats
    lexical_stats = compute_lexical_stats(tokens, text)
    
    return tokens, sentences, lexical_stats


# Basic sanity test
if __name__ == "__main__":
    test_text = "Hello world! This is a test. I love coding and coffee! 😊"
    tokens, sentences, stats = process_text(test_text)
    print(f"Tokens: {tokens}")
    print(f"Sentences: {sentences}")
    print(f"Stats: {stats}")