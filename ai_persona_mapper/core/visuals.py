# ai_persona_mapper/core/visuals.py
"""
Visualization functions for AI Persona Mapper.
All plotting using Matplotlib only.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import io
import base64


# Set default style
plt.style.use('default')


def set_plot_style():
    """Set consistent plot styling."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def plot_top_tokens(df: pd.DataFrame) -> plt.Figure:
    """
    Plot top tokens as horizontal bar chart.
    
    Args:
        df: DataFrame with token, count, and percentage columns
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create horizontal bars
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['count'], color='#4A90E2')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['token'])
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title('Top Tokens', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    for i, (count, pct) in enumerate(zip(df['count'], df['percentage'])):
        ax.text(count + 0.5, i, f'{pct}%', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    return fig


def plot_top_bigrams(df: pd.DataFrame) -> plt.Figure:
    """
    Plot top bigrams as horizontal bar chart.
    
    Args:
        df: DataFrame with bigram and count columns
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No bigrams found', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Create horizontal bars
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['count'], color='#E94B3C')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['bigram'])
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title('Top Bigrams', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_theme_breakdown(theme_series: pd.Series) -> plt.Figure:
    """
    Plot theme breakdown as pie chart or bar chart.
    
    Args:
        theme_series: Series with theme names as index and scores as values
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter themes with >0 score
    active_themes = theme_series[theme_series > 0][:8]  # Top 8 themes
    
    if len(active_themes) == 0:
        ax.text(0.5, 0.5, 'No themes detected', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Use bar chart for better readability
    colors = plt.cm.Set3(np.linspace(0, 1, len(active_themes)))
    bars = ax.bar(range(len(active_themes)), active_themes.values, color=colors)
    
    # Customize
    ax.set_xticks(range(len(active_themes)))
    ax.set_xticklabels(active_themes.index, rotation=45, ha='right')
    ax.set_ylabel('Theme Score')
    ax.set_title('Theme Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(active_themes.values) * 1.1)
    
    # Add value labels
    for bar, val in zip(bars, active_themes.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_cooccurrence_heatmap(matrix: np.ndarray, labels: List[str]) -> plt.Figure:
    """
    Plot co-occurrence matrix as heatmap.
    
    Args:
        matrix: Co-occurrence matrix
        labels: Token labels
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Co-occurrence Strength', rotation=270, labelpad=20)
    
    # Add title
    ax.set_title('Token Co-occurrence Matrix', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    
    plt.tight_layout()
    return fig


def plot_traits_radar(scores: Dict[str, float]) -> plt.Figure:
    """
    Plot personality traits as radar chart.
    
    Args:
        scores: Dictionary of trait names to scores (0-100)
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Prepare data
    labels = list(scores.keys())
    values = list(scores.values())
    num_vars = len(labels)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#4A90E2', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='#4A90E2')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8, color='gray')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add title
    ax.set_title('Personality Traits Radar', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for angle, value, label in zip(angles[:-1], values[:-1], labels):
        x = angle
        y = value + 5  # Offset slightly
        ax.text(x, y, f'{value:.0f}', ha='center', va='center', size=9, color='#333')
    
    plt.tight_layout()
    return fig


def plot_habit_curve(
    days: np.ndarray,
    median: np.ndarray,
    p10: Optional[np.ndarray] = None,
    p90: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Plot habit formation curve with uncertainty bands.
    
    Args:
        days: Array of day values
        median: Median habit consistency values
        p10: 10th percentile values (optional)
        p90: 90th percentile values (optional)
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot uncertainty band if provided
    if p10 is not None and p90 is not None:
        ax.fill_between(days, p10 * 100, p90 * 100, alpha=0.2, color='#4A90E2', 
                        label='80% Confidence')
    
    # Plot median line
    ax.plot(days, median * 100, linewidth=2.5, color='#4A90E2', label='Expected Path')
    
    # Add milestone lines
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='Strong Habit')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.3, label='Moderate')
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.3, label='Building')
    
    # Customize
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Habit Consistency (%)', fontsize=12)
    ax.set_title('Habit Formation Projection', fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(days) - 1)
    ax.set_ylim(0, 100)
    
    # Add legend
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_persona_card(analysis: Dict) -> bytes:
    """
    Create a single-image persona card with all key information.
    
    Args:
        analysis: Dictionary containing analysis results
        
    Returns:
        PNG image as bytes
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('AI Persona Card', fontsize=16, fontweight='bold', y=0.98)
    
    # Add metadata
    fig.text(0.5, 0.94, f"Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}", 
             ha='center', fontsize=10, color='gray')
    fig.text(0.5, 0.91, f"ID: {analysis['id']}", ha='center', fontsize=9, color='gray')
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, 
                          left=0.08, right=0.95, top=0.85, bottom=0.05)
    
    # 1. Mini traits radar (top left)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    _draw_mini_radar(ax1, analysis['traits'])
    ax1.set_title('Personality', fontsize=11, pad=10)
    
    # 2. Top themes bar chart (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    top_themes = analysis['themes'][:5]
    if len(top_themes) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_themes)))
        bars = ax2.barh(range(len(top_themes)), top_themes.values, color=colors)
        ax2.set_yticks(range(len(top_themes)))
        ax2.set_yticklabels(top_themes.index, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlim(0, max(top_themes.values) * 1.1 if len(top_themes) > 0 else 1)
        ax2.set_title('Top Themes', fontsize=11)
        ax2.tick_params(axis='x', labelsize=8)
    
    # 3. Key stats (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    stats_text = f"""Key Statistics:
    
Vocabulary: {analysis['lexical_stats']['vocab_size']} words
Complexity: {analysis['lexical_stats']['type_token_ratio']:.2f}
Questions: {analysis['lexical_stats']['question_count']}
Exclamations: {analysis['lexical_stats']['exclamation_count']}
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    ax3.set_title('Metrics', fontsize=11, pad=10)
    
    # 4. Habit sparkline (bottom)
    ax4 = fig.add_subplot(gs[1, :])
    if 'habit_data' in analysis:
        days = analysis['habit_data']['days'][:30]  # First 30 days
        median = analysis['habit_data']['median'][:30]
        ax4.plot(days, median * 100, linewidth=2, color='#4A90E2')
        ax4.fill_between(days, 0, median * 100, alpha=0.3, color='#4A90E2')
        ax4.set_xlabel('Days', fontsize=10)
        ax4.set_ylabel('Consistency %', fontsize=10)
        ax4.set_title(f"Habit Projection: {analysis.get('habit_keyword', 'general').title()}", 
                     fontsize=11)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.2)
        ax4.tick_params(labelsize=9)
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    plt.close(fig)
    
    return png_bytes


def _draw_mini_radar(ax, traits: Dict[str, float]):
    """Helper function to draw mini radar chart."""
    labels = list(traits.keys())
    values = list(traits.values())
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=1.5, color='#4A90E2', markersize=4)
    ax.fill(angles, values, alpha=0.25, color='#4A90E2')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l[:3] for l in labels], size=8)  # Abbreviated labels
    ax.set_ylim(0, 100)
    ax.set_yticks([50, 100])
    ax.set_yticklabels(['50', '100'], size=7, color='gray')
    ax.grid(True, alpha=0.3)


def create_comparison_plots(analysis1: Dict, analysis2: Dict) -> plt.Figure:
    """
    Create comparison plots for two analyses.
    
    Args:
        analysis1: First analysis dictionary
        analysis2: Second analysis dictionary
        
    Returns:
        Matplotlib figure with comparison plots
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Radar comparison (left)
    ax1 = fig.add_subplot(121, projection='polar')
    
    # Plot both radars
    for analysis, color, label in [(analysis1, '#4A90E2', 'Analysis 1'), 
                                   (analysis2, '#E94B3C', 'Analysis 2')]:
        traits = analysis['traits']
        labels = list(traits.keys())
        values = list(traits.values())
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, color=color, 
                label=label, markersize=6, alpha=0.7)
        ax1.fill(angles, values, alpha=0.1, color=color)
    
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.set_title('Personality Comparison', fontsize=12, fontweight='bold', pad=20)
    
    # Habit comparison (right)
    ax2 = fig.add_subplot(122)
    
    if 'habit_data' in analysis1 and 'habit_data' in analysis2:
        days1 = analysis1['habit_data']['days']
        median1 = analysis1['habit_data']['median']
        days2 = analysis2['habit_data']['days']
        median2 = analysis2['habit_data']['median']
        
        ax2.plot(days1, median1 * 100, linewidth=2, color='#4A90E2', 
                label='Analysis 1', alpha=0.7)
        ax2.plot(days2, median2 * 100, linewidth=2, color='#E94B3C', 
                label='Analysis 2', alpha=0.7)
        
        ax2.set_xlabel('Days', fontsize=11)
        ax2.set_ylabel('Habit Consistency (%)', fontsize=11)
        ax2.set_title('Habit Projection Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    return fig