# ai_persona_mapper/core/patterns.py
"""
Pattern analysis and insight generation for AI Persona Mapper.
"""

from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from collections import Counter


def compute_frequency_dataframe(tokens: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Compute token frequency dataframe.
    
    Args:
        tokens: List of tokens
        top_n: Number of top tokens to return
        
    Returns:
        DataFrame with token and frequency columns
    """
    # Count tokens
    token_counts = Counter(tokens)
    
    # Get top N
    top_tokens = token_counts.most_common(top_n)
    
    # Create DataFrame
    df = pd.DataFrame(top_tokens, columns=['token', 'count'])
    
    # Add percentage column
    total = sum(token_counts.values())
    df['percentage'] = (df['count'] / total * 100).round(1)
    
    return df


def compute_cooccurrence_matrix(tokens: List[str], top_k: int = 12, window_size: int = 4) -> Tuple[np.ndarray, List[str]]:
    """
    Compute co-occurrence matrix for top K tokens.
    
    Args:
        tokens: List of tokens
        top_k: Number of top tokens to consider
        window_size: Context window size
        
    Returns:
        Tuple of (normalized matrix, labels)
    """
    # Get top K tokens
    token_counts = Counter(tokens)
    top_tokens = [token for token, _ in token_counts.most_common(top_k)]
    
    # Create token to index mapping
    token_to_idx = {token: i for i, token in enumerate(top_tokens)}
    
    # Initialize co-occurrence matrix
    matrix = np.zeros((top_k, top_k))
    
    # Compute co-occurrences
    for i in range(len(tokens)):
        if tokens[i] in token_to_idx:
            idx1 = token_to_idx[tokens[i]]
            
            # Look within window
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            
            for j in range(start, end):
                if i != j and tokens[j] in token_to_idx:
                    idx2 = token_to_idx[tokens[j]]
                    matrix[idx1][idx2] += 1
    
    # Normalize matrix to [0, 1]
    if matrix.max() > 0:
        matrix = matrix / matrix.max()
    
    return matrix, top_tokens


def calculate_burstiness(tokens: List[str]) -> float:
    """
    Calculate burstiness score for token distribution.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Burstiness score (0-1)
    """
    if len(tokens) < 10:
        return 0.0
    
    # Get token counts
    counts = Counter(tokens)
    
    # Calculate variance/mean ratio (index of dispersion)
    frequencies = list(counts.values())
    mean_freq = np.mean(frequencies)
    var_freq = np.var(frequencies)
    
    if mean_freq > 0:
        burstiness = var_freq / mean_freq
        # Normalize to 0-1 range
        burstiness = min(1.0, burstiness / 10)
    else:
        burstiness = 0.0
    
    return float(burstiness)


def calculate_repetition_score(tokens: List[str]) -> float:
    """
    Calculate how repetitive the text is.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Repetition score (0-1)
    """
    if len(tokens) == 0:
        return 0.0
    
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    
    # Inverse of type-token ratio
    repetition = 1 - (unique_tokens / total_tokens)
    
    return float(repetition)


def detect_emotional_intensity(tokens: List[str], lexical_stats: Dict) -> float:
    """
    Detect emotional intensity based on various signals.
    
    Args:
        tokens: List of tokens
        lexical_stats: Dictionary of lexical statistics
        
    Returns:
        Emotional intensity score (0-1)
    """
    intensity_markers = {
        'very', 'really', 'so', 'extremely', 'absolutely', 'totally',
        'completely', 'definitely', 'seriously', 'literally', 'actually',
        'honestly', 'truly', 'super', 'ultra', 'mega', 'incredibly'
    }
    
    # Count intensity markers
    intensity_count = sum(1 for t in tokens if t.lower() in intensity_markers)
    
    # Factor in exclamation marks
    exclamation_score = min(1.0, lexical_stats.get('exclamation_count', 0) / 10)
    
    # Factor in capitalized words (if we had original casing)
    # For now, use intensity markers and exclamations
    
    intensity = min(1.0, (intensity_count / len(tokens) * 20) + exclamation_score * 0.5)
    
    return float(intensity)


def generate_insights(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> pd.DataFrame:
    """
    Generate human-readable insights from analysis.
    
    Args:
        tokens: List of tokens
        lexical_stats: Dictionary of lexical statistics
        themes: Series of theme scores
        
    Returns:
        DataFrame with Insight and Value/Note columns
    """
    insights = []
    
    # Vocabulary richness
    ttr = lexical_stats.get('type_token_ratio', 0)
    if ttr > 0.7:
        insights.append(("Vocabulary Richness", "Very diverse vocabulary (>70% unique)"))
    elif ttr > 0.5:
        insights.append(("Vocabulary Richness", f"Moderate diversity ({ttr*100:.0f}% unique)"))
    else:
        insights.append(("Vocabulary Richness", f"Repetitive language ({ttr*100:.0f}% unique)"))
    
    # Dominant theme
    if len(themes) > 0 and themes.iloc[0] > 0:
        top_theme = themes.index[0]
        insights.append(("Dominant Theme", f"{top_theme.title()} ({themes.iloc[0]*100:.0f}%)"))
    
    # Emotional intensity
    intensity = detect_emotional_intensity(tokens, lexical_stats)
    if intensity > 0.7:
        insights.append(("Emotional Intensity", "Very high - lots of emphasis"))
    elif intensity > 0.4:
        insights.append(("Emotional Intensity", "Moderate emotional expression"))
    else:
        insights.append(("Emotional Intensity", "Calm and measured tone"))
    
    # Sentence complexity
    avg_sent_len = lexical_stats.get('avg_sentence_length', 0)
    if avg_sent_len > 20:
        insights.append(("Writing Style", "Complex, detailed sentences"))
    elif avg_sent_len > 10:
        insights.append(("Writing Style", "Balanced sentence structure"))
    else:
        insights.append(("Writing Style", "Short, concise sentences"))
    
    # Question frequency
    questions = lexical_stats.get('question_count', 0)
    if questions > 5:
        insights.append(("Questioning", f"Very inquisitive ({questions} questions)"))
    elif questions > 0:
        insights.append(("Questioning", f"Some curiosity ({questions} questions)"))
    
    # Burstiness
    burst = calculate_burstiness(tokens)
    if burst > 0.7:
        insights.append(("Topic Focus", "Highly focused on specific topics"))
    elif burst > 0.3:
        insights.append(("Topic Focus", "Moderate topic variation"))
    else:
        insights.append(("Topic Focus", "Evenly distributed topics"))
    
    # Theme diversity
    active_themes = themes[themes > 0.05]  # Themes with >5% presence
    if len(active_themes) > 4:
        insights.append(("Theme Diversity", f"Very diverse ({len(active_themes)} themes)"))
    elif len(active_themes) > 2:
        insights.append(("Theme Diversity", f"Moderate ({len(active_themes)} themes)"))
    else:
        insights.append(("Theme Diversity", "Focused on few themes"))
    
    # Word length preference
    avg_word_len = lexical_stats.get('avg_word_length', 0)
    if avg_word_len > 6:
        insights.append(("Word Choice", "Prefers sophisticated vocabulary"))
    elif avg_word_len > 4:
        insights.append(("Word Choice", "Standard vocabulary complexity"))
    else:
        insights.append(("Word Choice", "Simple, accessible language"))
    
    # Create DataFrame
    df = pd.DataFrame(insights, columns=['Insight', 'Value/Note'])
    
    return df