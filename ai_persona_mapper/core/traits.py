# ai_persona_mapper/core/traits.py
"""
Personality trait computation for AI Persona Mapper.
Rule-based trait extraction with transparent weights.
"""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


def get_trait_descriptions() -> Dict[str, str]:
    """
    Get descriptions for each personality trait.
    
    Returns:
        Dictionary mapping trait names to descriptions
    """
    return {
        'Focus': 'Ability to concentrate and maintain attention',
        'Chaos': 'Tendency toward randomness and unpredictability',
        'Energy': 'Overall enthusiasm and vigor level',
        'Discipline': 'Structure and self-control in approach',
        'Openness': 'Receptiveness to new ideas and experiences',
        'Socialness': 'Inclination toward social interaction',
        'NightOwlness': 'Preference for nighttime activity'
    }


def compute_focus_score(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[float, Dict]:
    """
    Compute Focus trait score (0-100).
    
    High focus: productivity keywords, low question marks, high discipline themes.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (score, feature_contributions)
    """
    contributions = {}
    
    # Keywords indicating focus (weight: 30%)
    focus_keywords = {'focus', 'concentrate', 'attention', 'deep', 'flow', 
                     'zone', 'productivity', 'work', 'study', 'complete', 'finish'}
    focus_count = sum(1 for t in tokens if t in focus_keywords)
    focus_keyword_score = min(100, focus_count * 10)
    contributions['focus_keywords'] = focus_keyword_score * 0.3
    
    # Low question marks indicate certainty (weight: 20%)
    questions = lexical_stats.get('question_count', 0)
    question_score = max(0, 100 - questions * 20)
    contributions['low_questions'] = question_score * 0.2
    
    # Productivity/study themes (weight: 30%)
    productivity_score = themes.get('productivity', 0) * 100
    study_score = themes.get('study', 0) * 100
    theme_score = (productivity_score + study_score) / 2
    contributions['productive_themes'] = theme_score * 0.3
    
    # Sentence structure - longer sentences = more focus (weight: 20%)
    avg_sent_len = lexical_stats.get('avg_sentence_length', 10)
    structure_score = min(100, avg_sent_len * 5)
    contributions['sentence_structure'] = structure_score * 0.2
    
    # Calculate final score
    final_score = sum(contributions.values())
    final_score = np.clip(final_score, 0, 100)
    
    return float(final_score), contributions


def compute_chaos_score(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[float, Dict]:
    """
    Compute Chaos trait score (0-100).
    
    High chaos: high vocabulary diversity, many exclamations, topic jumping.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (score, feature_contributions)
    """
    contributions = {}
    
    # High type-token ratio = more chaotic (weight: 25%)
    ttr = lexical_stats.get('type_token_ratio', 0.5)
    ttr_score = ttr * 100
    contributions['vocabulary_diversity'] = ttr_score * 0.25
    
    # Exclamation marks (weight: 25%)
    exclamations = lexical_stats.get('exclamation_count', 0)
    exclamation_score = min(100, exclamations * 15)
    contributions['exclamations'] = exclamation_score * 0.25
    
    # Chaos keywords (weight: 20%)
    chaos_keywords = {'random', 'crazy', 'wild', 'chaos', 'mess', 'confused',
                     'scattered', 'everywhere', 'sudden', 'unexpected', 'weird'}
    chaos_count = sum(1 for t in tokens if t in chaos_keywords)
    chaos_keyword_score = min(100, chaos_count * 15)
    contributions['chaos_keywords'] = chaos_keyword_score * 0.2
    
    # Theme diversity - more themes = more chaotic (weight: 30%)
    active_themes = themes[themes > 0.05]
    diversity_score = min(100, len(active_themes) * 20)
    contributions['theme_diversity'] = diversity_score * 0.3
    
    # Calculate final score
    final_score = sum(contributions.values())
    final_score = np.clip(final_score, 0, 100)
    
    return float(final_score), contributions


def compute_energy_score(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[float, Dict]:
    """
    Compute Energy trait score (0-100).
    
    High energy: exercise theme, energy keywords, exclamations.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (score, feature_contributions)
    """
    contributions = {}
    
    # Energy keywords (weight: 35%)
    energy_keywords = {'energy', 'excited', 'pumped', 'active', 'dynamic',
                      'enthusiastic', 'motivated', 'passionate', 'alive', 'vibrant',
                      'go', 'run', 'jump', 'move', 'action'}
    energy_count = sum(1 for t in tokens if t in energy_keywords)
    energy_keyword_score = min(100, energy_count * 12)
    contributions['energy_keywords'] = energy_keyword_score * 0.35
    
    # Exercise and fun themes (weight: 30%)
    exercise_score = themes.get('exercise', 0) * 100
    fun_score = themes.get('fun', 0) * 100
    theme_score = (exercise_score * 0.6 + fun_score * 0.4)
    contributions['active_themes'] = theme_score * 0.3
    
    # Exclamation marks indicate excitement (weight: 20%)
    exclamations = lexical_stats.get('exclamation_count', 0)
    exclamation_score = min(100, exclamations * 12)
    contributions['excitement'] = exclamation_score * 0.2
    
    # Short, punchy sentences (weight: 15%)
    avg_sent_len = lexical_stats.get('avg_sentence_length', 15)
    if avg_sent_len < 10:
        sentence_score = 80
    elif avg_sent_len < 15:
        sentence_score = 50
    else:
        sentence_score = 20
    contributions['sentence_energy'] = sentence_score * 0.15
    
    # Calculate final score
    final_score = sum(contributions.values())
    final_score = np.clip(final_score, 0, 100)
    
    return float(final_score), contributions


def compute_discipline_score(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[float, Dict]:
    """
    Compute Discipline trait score (0-100).
    
    High discipline: planning keywords, productivity theme, structured writing.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (score, feature_contributions)
    """
    contributions = {}
    
    # Discipline keywords (weight: 35%)
    discipline_keywords = {'plan', 'schedule', 'organize', 'routine', 'habit',
                          'discipline', 'control', 'structure', 'systematic',
                          'regular', 'consistent', 'daily', 'practice', 'manage'}
    discipline_count = sum(1 for t in tokens if t in discipline_keywords)
    discipline_keyword_score = min(100, discipline_count * 12)
    contributions['discipline_keywords'] = discipline_keyword_score * 0.35
    
    # Productivity and work themes (weight: 30%)
    productivity_score = themes.get('productivity', 0) * 100
    work_score = themes.get('work', 0) * 100
    theme_score = (productivity_score * 0.6 + work_score * 0.4)
    contributions['structured_themes'] = theme_score * 0.3
    
    # Low chaos indicates discipline (weight: 20%)
    # We'll estimate this simply
    ttr = lexical_stats.get('type_token_ratio', 0.5)
    structure_score = (1 - ttr) * 100  # Lower diversity = more structured
    contributions['structured_language'] = structure_score * 0.2
    
    # Consistent sentence length (weight: 15%)
    # For simplicity, use average sentence length as proxy
    avg_sent_len = lexical_stats.get('avg_sentence_length', 15)
    if 10 <= avg_sent_len <= 20:
        consistency_score = 80
    else:
        consistency_score = 40
    contributions['consistency'] = consistency_score * 0.15
    
    # Calculate final score
    final_score = sum(contributions.values())
    final_score = np.clip(final_score, 0, 100)
    
    return float(final_score), contributions


def compute_openness_score(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[float, Dict]:
    """
    Compute Openness trait score (0-100).
    
    High openness: curiosity keywords, questions, diverse themes.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (score, feature_contributions)
    """
    contributions = {}
    
    # Openness keywords (weight: 30%)
    openness_keywords = {'new', 'explore', 'discover', 'curious', 'wonder',
                        'imagine', 'creative', 'idea', 'interesting', 'learn',
                        'experience', 'adventure', 'try', 'experiment', 'maybe'}
    openness_count = sum(1 for t in tokens if t in openness_keywords)
    openness_keyword_score = min(100, openness_count * 10)
    contributions['openness_keywords'] = openness_keyword_score * 0.3
    
    # Questions indicate curiosity (weight: 25%)
    questions = lexical_stats.get('question_count', 0)
    question_score = min(100, questions * 20)
    contributions['curiosity'] = question_score * 0.25
    
    # Theme diversity (weight: 25%)
    active_themes = themes[themes > 0.05]
    diversity_score = min(100, len(active_themes) * 18)
    contributions['theme_exploration'] = diversity_score * 0.25
    
    # Vocabulary richness (weight: 20%)
    ttr = lexical_stats.get('type_token_ratio', 0.5)
    vocab_score = ttr * 100
    contributions['vocabulary_richness'] = vocab_score * 0.2
    
    # Calculate final score
    final_score = sum(contributions.values())
    final_score = np.clip(final_score, 0, 100)
    
    return float(final_score), contributions


def compute_socialness_score(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[float, Dict]:
    """
    Compute Socialness trait score (0-100).
    
    High socialness: social theme, people keywords, conversation markers.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (score, feature_contributions)
    """
    contributions = {}
    
    # Social keywords (weight: 35%)
    social_keywords = {'friend', 'people', 'talk', 'chat', 'meet', 'party',
                       'together', 'group', 'team', 'social', 'share', 'connect',
                       'community', 'conversation', 'we', 'us', 'our', 'they'}
    social_count = sum(1 for t in tokens if t in social_keywords)
    social_keyword_score = min(100, social_count * 8)
    contributions['social_keywords'] = social_keyword_score * 0.35
    
    # Social theme (weight: 35%)
    social_theme_score = themes.get('social', 0) * 100
    contributions['social_theme'] = social_theme_score * 0.35
    
    # Questions suggest engagement (weight: 15%)
    questions = lexical_stats.get('question_count', 0)
    engagement_score = min(100, questions * 15)
    contributions['engagement'] = engagement_score * 0.15
    
    # Fun theme also indicates socialness (weight: 15%)
    fun_score = themes.get('fun', 0) * 100
    contributions['fun_activities'] = fun_score * 0.15
    
    # Calculate final score
    final_score = sum(contributions.values())
    final_score = np.clip(final_score, 0, 100)
    
    return float(final_score), contributions


def compute_nightowlness_score(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[float, Dict]:
    """
    Compute NightOwlness trait score (0-100).
    
    High nightowlness: night theme, night keywords, late-night activities.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (score, feature_contributions)
    """
    contributions = {}
    
    # Night keywords (weight: 40%)
    night_keywords = {'night', 'late', 'midnight', 'evening', 'dark', 'pm',
                     'overnight', 'nocturnal', 'insomnia', 'nighttime', 'moon',
                     'star', 'afterdark', 'nightowl', '2am', '3am', 'dawn'}
    night_count = sum(1 for t in tokens if t in night_keywords)
    night_keyword_score = min(100, night_count * 15)
    contributions['night_keywords'] = night_keyword_score * 0.4
    
    # Night theme (weight: 35%)
    night_theme_score = themes.get('night', 0) * 100
    contributions['night_theme'] = night_theme_score * 0.35
    
    # Sleep issues (weight: 15%)
    sleep_theme = themes.get('sleep', 0) * 100
    # High sleep theme might indicate sleep issues
    contributions['sleep_patterns'] = sleep_theme * 0.15
    
    # Tech theme (often associated with late night) (weight: 10%)
    tech_score = themes.get('tech', 0) * 100
    contributions['late_tech_use'] = tech_score * 0.1
    
    # Calculate final score
    final_score = sum(contributions.values())
    final_score = np.clip(final_score, 0, 100)
    
    return float(final_score), contributions


def compute_traits(tokens: List[str], lexical_stats: Dict, themes: pd.Series) -> Tuple[Dict[str, float], Dict]:
    """
    Compute all personality traits.
    
    Args:
        tokens: List of tokens
        lexical_stats: Lexical statistics
        themes: Theme scores
        
    Returns:
        Tuple of (trait_scores, feature_contributions)
    """
    traits = {}
    all_contributions = {}
    
    # Compute each trait
    traits['Focus'], all_contributions['Focus'] = compute_focus_score(tokens, lexical_stats, themes)
    traits['Chaos'], all_contributions['Chaos'] = compute_chaos_score(tokens, lexical_stats, themes)
    traits['Energy'], all_contributions['Energy'] = compute_energy_score(tokens, lexical_stats, themes)
    traits['Discipline'], all_contributions['Discipline'] = compute_discipline_score(tokens, lexical_stats, themes)
    traits['Openness'], all_contributions['Openness'] = compute_openness_score(tokens, lexical_stats, themes)
    traits['Socialness'], all_contributions['Socialness'] = compute_socialness_score(tokens, lexical_stats, themes)
    traits['NightOwlness'], all_contributions['NightOwlness'] = compute_nightowlness_score(tokens, lexical_stats, themes)
    
    return traits, all_contributions