# ai_persona_mapper/core/habit.py
"""
Habit curve simulation and projection for AI Persona Mapper.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def logistic_curve(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    """
    Generate logistic growth curve.
    
    Args:
        t: Time points
        L: Maximum value (carrying capacity)
        k: Growth rate
        t0: Midpoint
        
    Returns:
        Array of curve values
    """
    return L / (1 + np.exp(-k * (t - t0)))


def simulate_single_habit_curve(
    traits: Dict[str, float],
    horizon_days: int,
    add_noise: bool = True
) -> np.ndarray:
    """
    Simulate a single habit formation curve based on traits.
    
    Args:
        traits: Dictionary of personality traits
        horizon_days: Number of days to simulate
        add_noise: Whether to add random variations
        
    Returns:
        Array of habit consistency values (0-1)
    """
    # Extract relevant traits
    focus = traits.get('Focus', 50) / 100
    chaos = traits.get('Chaos', 50) / 100
    energy = traits.get('Energy', 50) / 100
    discipline = traits.get('Discipline', 50) / 100
    
    # Determine logistic parameters based on traits
    # L (max consistency): Higher discipline = higher ceiling
    L = 0.85 + (discipline - 0.5) * 0.2  # Range: 0.75 to 0.95
    L = np.clip(L, 0.75, 0.95)
    
    # k (growth rate): Higher focus and energy = faster growth
    k = 0.08 + (focus * 0.04) + (energy * 0.03)  # Range: 0.08 to 0.15
    k = np.clip(k, 0.08, 0.15)
    
    # t0 (midpoint): Higher discipline = earlier midpoint
    t0 = 18 - (discipline * 6)  # Range: 12 to 18 days
    t0 = np.clip(t0, 12, 18)
    
    # Generate base curve
    days = np.arange(horizon_days)
    base_curve = logistic_curve(days, L, k, t0)
    
    if add_noise:
        # Add Gaussian noise scaled by chaos
        noise_scale = chaos * 0.05
        noise = np.random.normal(0, noise_scale, horizon_days)
        
        # Add periodic fatigue dips
        fatigue_period = 7  # Weekly dips
        fatigue_amplitude = chaos * 0.1
        fatigue = fatigue_amplitude * np.sin(2 * np.pi * days / fatigue_period)
        
        # Combine
        curve = base_curve + noise + fatigue
        
        # Add random "bad days" for high chaos
        if chaos > 0.6:
            n_bad_days = int(horizon_days * chaos * 0.1)
            bad_days = np.random.choice(horizon_days, n_bad_days, replace=False)
            curve[bad_days] *= (0.5 + np.random.random(n_bad_days) * 0.3)
    else:
        curve = base_curve
    
    # Clamp to valid range
    curve = np.clip(curve, 0, 1)
    
    return curve


def simulate_habit_curve(
    traits: Dict[str, float],
    horizon_days: int,
    monte_carlo_runs: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate habit formation with Monte Carlo uncertainty bands.
    
    Args:
        traits: Dictionary of personality traits
        horizon_days: Number of days to simulate
        monte_carlo_runs: Number of simulation runs
        
    Returns:
        Tuple of (days, median_curve, p10_curve, p90_curve, all_curves)
    """
    # Run multiple simulations
    all_curves = []
    for _ in range(monte_carlo_runs):
        curve = simulate_single_habit_curve(traits, horizon_days, add_noise=True)
        all_curves.append(curve)
    
    # Convert to numpy array for easier manipulation
    all_curves = np.array(all_curves)
    
    # Calculate statistics
    days = np.arange(horizon_days)
    median_curve = np.median(all_curves, axis=0)
    p10_curve = np.percentile(all_curves, 10, axis=0)
    p90_curve = np.percentile(all_curves, 90, axis=0)
    
    return days, median_curve, p10_curve, p90_curve, all_curves


def detect_habit_events(habit_curve: np.ndarray) -> List[Dict]:
    """
    Detect significant events in habit formation curve.
    
    Args:
        habit_curve: Array of habit consistency values
        
    Returns:
        List of event dictionaries
    """
    events = []
    
    # Detect plateau (7-day window with minimal change)
    if len(habit_curve) >= 14:
        for i in range(7, len(habit_curve) - 7):
            window = habit_curve[i:i+7]
            if np.std(window) < 0.02 and np.mean(window) > 0.5:
                events.append({
                    'Event': 'Plateau Reached',
                    'Day': i,
                    'Consistency': f"{np.mean(window)*100:.1f}%",
                    'Note': 'Habit stabilized'
                })
                break  # Only report first plateau
    
    # Detect slump (3-day cumulative drop)
    for i in range(3, len(habit_curve)):
        window = habit_curve[i-3:i]
        if len(window) == 3:
            drop = habit_curve[i-3] - habit_curve[i-1]
            if drop > 0.15:  # Significant drop
                events.append({
                    'Event': 'Slump Detected',
                    'Day': i-1,
                    'Consistency': f"{habit_curve[i-1]*100:.1f}%",
                    'Note': f"Lost {drop*100:.0f}% over 3 days"
                })
    
    # Detect milestone crossings
    milestones = [0.25, 0.5, 0.7, 0.9]
    milestone_names = ['Started', 'Halfway', 'Strong', 'Mastery']
    
    for milestone, name in zip(milestones, milestone_names):
        # Find first crossing
        crossing_idx = np.where(habit_curve >= milestone)[0]
        if len(crossing_idx) > 0:
            day = crossing_idx[0]
            events.append({
                'Event': f'{name} Milestone',
                'Day': day,
                'Consistency': f"{milestone*100:.0f}%",
                'Note': f"Reached {name.lower()} level"
            })
    
    # Sort events by day
    events = sorted(events, key=lambda x: x['Day'])
    
    # Limit to most significant events
    if len(events) > 5:
        # Prioritize milestones and significant events
        priority_events = []
        for event in events:
            if 'Milestone' in event['Event'] or 'Plateau' in event['Event']:
                priority_events.append(event)
        
        # Add any slumps if space remains
        for event in events:
            if 'Slump' in event['Event'] and len(priority_events) < 5:
                priority_events.append(event)
        
        events = sorted(priority_events, key=lambda x: x['Day'])[:5]
    
    return events


def get_habit_recommendations(traits: Dict[str, float], habit_keyword: str) -> List[str]:
    """
    Generate personalized habit recommendations based on traits.
    
    Args:
        traits: Dictionary of personality traits
        habit_keyword: The habit being tracked
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Focus-based recommendations
    if traits['Focus'] < 40:
        recommendations.append(f"Start with shorter {habit_keyword} sessions to build focus gradually")
    elif traits['Focus'] > 70:
        recommendations.append(f"Your high focus allows for intensive {habit_keyword} sessions")
    
    # Discipline-based recommendations
    if traits['Discipline'] < 40:
        recommendations.append("Use external accountability (apps, partners) to maintain consistency")
    elif traits['Discipline'] > 70:
        recommendations.append("Your strong discipline suggests you can handle aggressive goals")
    
    # Chaos-based recommendations
    if traits['Chaos'] > 60:
        recommendations.append("Build flexibility into your routine to accommodate your spontaneous nature")
    
    # Energy-based recommendations
    if traits['Energy'] < 40:
        recommendations.append(f"Schedule {habit_keyword} during your peak energy times")
    elif traits['Energy'] > 70:
        recommendations.append(f"Your high energy supports multiple daily {habit_keyword} sessions")
    
    # NightOwlness-based recommendations
    if traits['NightOwlness'] > 60:
        recommendations.append(f"Consider evening/night {habit_keyword} sessions for better alignment")
    
    return recommendations[:3]  # Return top 3 recommendations