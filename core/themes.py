# ai_persona_mapper/core/themes.py
"""
Theme detection and keyword mapping for AI Persona Mapper.
"""

from typing import List, Dict
import pandas as pd


def get_theme_keywords() -> Dict[str, set]:
    """
    Get curated theme-keyword mappings.
    
    Returns:
        Dictionary mapping theme names to sets of keywords
    """
    return {
        'food': {
            'eat', 'food', 'meal', 'breakfast', 'lunch', 'dinner', 'snack',
            'hungry', 'cook', 'recipe', 'restaurant', 'coffee', 'tea', 'drink',
            'taste', 'delicious', 'pizza', 'burger', 'salad', 'fruit', 'vegetable'
        },
        'work': {
            'work', 'job', 'office', 'meeting', 'project', 'deadline', 'boss',
            'colleague', 'team', 'task', 'email', 'report', 'presentation',
            'career', 'professional', 'business', 'client', 'company', 'manager'
        },
        'sleep': {
            'sleep', 'bed', 'tired', 'rest', 'nap', 'dream', 'wake', 'morning',
            'night', 'insomnia', 'pillow', 'blanket', 'snore', 'alarm', 'drowsy',
            'yawn', 'bedtime', 'exhausted', 'fatigue'
        },
        'social': {
            'friend', 'family', 'party', 'meet', 'talk', 'chat', 'social',
            'people', 'group', 'hang', 'visit', 'together', 'conversation',
            'relationship', 'date', 'love', 'community', 'gather', 'connect'
        },
        'tech': {
            'computer', 'phone', 'app', 'website', 'code', 'program', 'software',
            'internet', 'online', 'digital', 'tech', 'device', 'screen', 'laptop',
            'data', 'algorithm', 'ai', 'machine', 'system', 'update'
        },
        'study': {
            'study', 'learn', 'read', 'book', 'class', 'exam', 'test', 'homework',
            'assignment', 'lecture', 'notes', 'research', 'library', 'knowledge',
            'education', 'school', 'university', 'course', 'subject', 'practice'
        },
        'exercise': {
            'exercise', 'workout', 'gym', 'run', 'walk', 'fitness', 'sport',
            'yoga', 'stretch', 'muscle', 'cardio', 'weight', 'training', 'jog',
            'bike', 'swim', 'health', 'active', 'sweat', 'strong'
        },
        'fun': {
            'fun', 'play', 'game', 'movie', 'music', 'dance', 'laugh', 'enjoy',
            'entertainment', 'hobby', 'relax', 'chill', 'watch', 'listen',
            'concert', 'show', 'art', 'creative', 'adventure', 'explore'
        },
        'stress': {
            'stress', 'anxiety', 'worry', 'nervous', 'pressure', 'overwhelm',
            'panic', 'tense', 'frustrated', 'angry', 'upset', 'difficult',
            'problem', 'issue', 'challenge', 'struggle', 'hard', 'tough', 'crisis'
        },
        'night': {
            'night', 'evening', 'midnight', 'late', 'dark', 'moon', 'star',
            'pm', 'overnight', 'nocturnal', 'dusk', 'twilight', 'nighttime',
            'afterdark', 'nightlife', 'insomnia', 'nightowl'
        },
        'nature': {
            'nature', 'tree', 'forest', 'mountain', 'ocean', 'beach', 'sun',
            'rain', 'weather', 'outdoor', 'hike', 'camp', 'park', 'garden',
            'flower', 'animal', 'bird', 'sky', 'cloud', 'fresh'
        },
        'productivity': {
            'productive', 'accomplish', 'complete', 'finish', 'achieve', 'goal',
            'plan', 'organize', 'schedule', 'efficient', 'focus', 'concentrate',
            'priority', 'todo', 'list', 'manage', 'optimize', 'progress'
        }
    }


def detect_themes(tokens: List[str], normalize: bool = True) -> pd.Series:
    """
    Detect themes present in tokens based on keyword matching.
    
    Args:
        tokens: List of text tokens
        normalize: Whether to normalize scores to sum to 1
        
    Returns:
        Pandas Series with theme names as index and scores as values
    """
    theme_keywords = get_theme_keywords()
    theme_scores = {}
    
    # Convert tokens to set for faster lookup
    token_set = set(t.lower() for t in tokens)
    
    # Count keyword matches for each theme
    for theme, keywords in theme_keywords.items():
        matches = len(token_set.intersection(keywords))
        theme_scores[theme] = matches
    
    # Create Series
    theme_series = pd.Series(theme_scores)
    
    # Normalize if requested
    if normalize and theme_series.sum() > 0:
        theme_series = theme_series / theme_series.sum()
    
    # Sort by score
    theme_series = theme_series.sort_values(ascending=False)
    
    return theme_series


def get_dominant_theme(tokens: List[str]) -> str:
    """
    Get the most dominant theme from tokens.
    
    Args:
        tokens: List of text tokens
        
    Returns:
        Name of dominant theme
    """
    themes = detect_themes(tokens, normalize=False)
    
    if themes.sum() == 0:
        return 'general'  # Default theme if no matches
    
    return themes.idxmax()


def get_theme_diversity(tokens: List[str]) -> float:
    """
    Calculate theme diversity score (0-1).
    Higher score means more diverse themes.
    
    Args:
        tokens: List of text tokens
        
    Returns:
        Diversity score between 0 and 1
    """
    themes = detect_themes(tokens, normalize=True)
    
    # Filter out zero scores
    active_themes = themes[themes > 0]
    
    if len(active_themes) <= 1:
        return 0.0
    
    # Calculate entropy-based diversity
    # Maximum entropy when all themes equal
    entropy = -(active_themes * active_themes.apply(lambda x: pd.np.log(x) if x > 0 else 0)).sum()
    max_entropy = pd.np.log(len(active_themes))
    
    diversity = entropy / max_entropy if max_entropy > 0 else 0
    
    return float(diversity)