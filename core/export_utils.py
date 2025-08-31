# ai_persona_mapper/core/export_utils.py
"""
Export utilities for AI Persona Mapper.
Handles data export, file saving, and report generation.
"""

import io
import base64
import json
import csv
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, name: str, dpi: int = 200) -> bytes:
    """
    Save matplotlib figure to bytes.
    
    Args:
        fig: Matplotlib figure
        name: Figure name (not used in bytes output, but kept for consistency)
        dpi: Resolution for saving
        
    Returns:
        PNG image as bytes
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.read()
    buf.close()
    plt.close(fig)
    return image_bytes


def export_summary_csv(analysis: Dict) -> str:
    """
    Export analysis summary to CSV format.
    
    Args:
        analysis: Analysis dictionary containing all results
        
    Returns:
        CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['AI Persona Mapper - Analysis Summary'])
    writer.writerow(['Generated', analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow(['Analysis ID', analysis['id']])
    writer.writerow([])
    
    # Personality Traits
    writer.writerow(['PERSONALITY TRAITS'])
    writer.writerow(['Trait', 'Score (0-100)', 'Level'])
    for trait, score in analysis['traits'].items():
        level = 'High' if score > 70 else 'Medium' if score > 40 else 'Low'
        writer.writerow([trait, f'{score:.1f}', level])
    writer.writerow([])
    
    # Top Themes
    writer.writerow(['TOP THEMES'])
    writer.writerow(['Theme', 'Presence (%)'])
    themes = analysis['themes']
    for theme, score in themes[:5].items():
        writer.writerow([theme.title(), f'{score*100:.1f}%'])
    writer.writerow([])
    
    # Lexical Statistics
    writer.writerow(['LEXICAL STATISTICS'])
    stats = analysis['lexical_stats']
    writer.writerow(['Vocabulary Size', stats['vocab_size']])
    writer.writerow(['Total Tokens', stats['total_tokens']])
    writer.writerow(['Type-Token Ratio', f"{stats['type_token_ratio']:.3f}"])
    writer.writerow(['Average Word Length', f"{stats['avg_word_length']:.1f}"])
    writer.writerow(['Question Count', stats['question_count']])
    writer.writerow(['Exclamation Count', stats['exclamation_count']])
    writer.writerow(['Sentence Count', stats['sentence_count']])
    writer.writerow(['Avg Sentence Length', f"{stats['avg_sentence_length']:.1f}"])
    writer.writerow([])
    
    # Key Insights
    writer.writerow(['KEY INSIGHTS'])
    if 'insights' in analysis and not analysis['insights'].empty:
        for _, row in analysis['insights'].iterrows():
            writer.writerow([row['Insight'], row['Value/Note']])
    writer.writerow([])
    
    # Habit Information
    writer.writerow(['HABIT PROJECTION'])
    writer.writerow(['Tracked Habit', analysis['habit_keyword'].title()])
    if 'habit_data' in analysis:
        habit_data = analysis['habit_data']
        median_final = habit_data['median'][-1] if len(habit_data['median']) > 0 else 0
        writer.writerow(['Final Consistency', f'{median_final*100:.1f}%'])
        writer.writerow(['Days Projected', len(habit_data['days'])])
    
    return output.getvalue()


def export_traits_json(analysis: Dict) -> str:
    """
    Export personality traits as JSON.
    
    Args:
        analysis: Analysis dictionary
        
    Returns:
        JSON string
    """
    export_data = {
        'id': analysis['id'],
        'timestamp': analysis['timestamp'].isoformat(),
        'traits': {k: round(v, 2) for k, v in analysis['traits'].items()},
        'dominant_theme': analysis['themes'].index[0] if len(analysis['themes']) > 0 else 'general',
        'habit_keyword': analysis['habit_keyword'],
        'lexical_summary': {
            'vocabulary_size': analysis['lexical_stats']['vocab_size'],
            'type_token_ratio': round(analysis['lexical_stats']['type_token_ratio'], 3),
            'questions': analysis['lexical_stats']['question_count'],
            'exclamations': analysis['lexical_stats']['exclamation_count']
        }
    }
    
    return json.dumps(export_data, indent=2)


def prepare_export_data(analysis: Dict) -> Dict[str, Any]:
    """
    Prepare analysis data for export in various formats.
    
    Args:
        analysis: Complete analysis dictionary
        
    Returns:
        Dictionary with prepared export data
    """
    export_ready = {
        'metadata': {
            'id': analysis['id'],
            'timestamp': analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'seed': analysis.get('seed', 42)
        },
        'personality': {
            'traits': analysis['traits'],
            'dominant_traits': sorted(
                analysis['traits'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        },
        'themes': {
            'distribution': analysis['themes'].to_dict(),
            'dominant': analysis['themes'].index[0] if len(analysis['themes']) > 0 else None,
            'diversity': len(analysis['themes'][analysis['themes'] > 0.05])
        },
        'lexical': analysis['lexical_stats'],
        'habit': {
            'keyword': analysis['habit_keyword'],
            'projection_days': len(analysis['habit_data']['days']) if 'habit_data' in analysis else 0,
            'final_consistency': None
        }
    }
    
    # Add final consistency if available
    if 'habit_data' in analysis and 'median' in analysis['habit_data']:
        median = analysis['habit_data']['median']
        if len(median) > 0:
            export_ready['habit']['final_consistency'] = float(median[-1])
    
    return export_ready


def create_markdown_report(analysis: Dict) -> str:
    """
    Create a markdown-formatted report of the analysis.
    
    Args:
        analysis: Analysis dictionary
        
    Returns:
        Markdown string
    """
    report = []
    
    # Header
    report.append("# AI Persona Mapper - Analysis Report")
    report.append(f"\n**Generated:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Analysis ID:** {analysis['id']}\n")
    
    # Executive Summary
    report.append("## Executive Summary")
    
    # Find dominant trait
    traits = analysis['traits']
    dominant_trait = max(traits.items(), key=lambda x: x[1])
    report.append(f"- **Dominant Personality Trait:** {dominant_trait[0]} ({dominant_trait[1]:.1f}/100)")
    
    # Find dominant theme
    if len(analysis['themes']) > 0 and analysis['themes'].iloc[0] > 0:
        report.append(f"- **Primary Theme:** {analysis['themes'].index[0].title()} ({analysis['themes'].iloc[0]*100:.1f}%)")
    
    # Vocabulary insight
    ttr = analysis['lexical_stats']['type_token_ratio']
    vocab_desc = "Rich and diverse" if ttr > 0.7 else "Moderate" if ttr > 0.5 else "Focused and repetitive"
    report.append(f"- **Language Style:** {vocab_desc} vocabulary (TTR: {ttr:.3f})")
    
    # Habit projection
    if 'habit_data' in analysis:
        final_consistency = analysis['habit_data']['median'][-1] * 100
        report.append(f"- **{analysis['habit_keyword'].title()} Habit Projection:** {final_consistency:.1f}% consistency after {len(analysis['habit_data']['days'])} days")
    
    # Personality Profile
    report.append("\n## Personality Profile")
    report.append("\n| Trait | Score | Interpretation |")
    report.append("|-------|-------|----------------|")
    
    for trait, score in sorted(traits.items(), key=lambda x: x[1], reverse=True):
        level = "🟢 High" if score > 70 else "🟡 Medium" if score > 40 else "🔴 Low"
        report.append(f"| {trait} | {score:.1f} | {level} |")
    
    # Thematic Analysis
    report.append("\n## Thematic Analysis")
    report.append("\nTop themes detected in your text:\n")
    
    for i, (theme, score) in enumerate(analysis['themes'][:5].items(), 1):
        if score > 0:
            bar = "█" * int(score * 20)
            report.append(f"{i}. **{theme.title()}**: {bar} {score*100:.1f}%")
    
    # Key Insights
    if 'insights' in analysis and not analysis['insights'].empty:
        report.append("\n## Key Insights")
        for _, row in analysis['insights'].iterrows():
            report.append(f"- **{row['Insight']}:** {row['Value/Note']}")
    
    # Recommendations
    report.append("\n## Personalized Recommendations")
    
    # Based on traits, provide recommendations
    if traits['Focus'] < 40:
        report.append("- 🎯 **Focus Enhancement:** Consider using time-blocking techniques and minimizing distractions")
    
    if traits['Chaos'] > 70:
        report.append("- 📋 **Structure Building:** Implement daily routines to balance your spontaneous nature")
    
    if traits['Energy'] > 70:
        report.append("- ⚡ **Energy Management:** Channel your high energy into multiple productive activities")
    
    if traits['Discipline'] < 40:
        report.append("- 📱 **Accountability Tools:** Use habit tracking apps or find an accountability partner")
    
    if traits['NightOwlness'] > 60:
        report.append("- 🌙 **Schedule Optimization:** Align important tasks with your evening energy peaks")
    
    # Footer
    report.append("\n---")
    report.append("*Report generated by AI Persona Mapper*")
    
    return "\n".join(report)


def batch_export(analyses: List[Dict], format: str = 'csv') -> bytes:
    """
    Export multiple analyses in batch.
    
    Args:
        analyses: List of analysis dictionaries
        format: Export format ('csv' or 'json')
        
    Returns:
        Exported data as bytes
    """
    if format == 'json':
        # Export as JSON array
        export_data = []
        for analysis in analyses:
            export_data.append({
                'id': analysis['id'],
                'timestamp': analysis['timestamp'].isoformat(),
                'traits': analysis['traits'],
                'dominant_theme': analysis['themes'].index[0] if len(analysis['themes']) > 0 else 'general',
                'habit_keyword': analysis['habit_keyword']
            })
        return json.dumps(export_data, indent=2).encode('utf-8')
    
    else:  # CSV format
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['ID', 'Timestamp', 'Focus', 'Chaos', 'Energy', 
                        'Discipline', 'Openness', 'Socialness', 'NightOwlness',
                        'Dominant Theme', 'Habit'])
        
        # Write data
        for analysis in analyses:
            traits = analysis['traits']
            dominant_theme = analysis['themes'].index[0] if len(analysis['themes']) > 0 else 'general'
            
            writer.writerow([
                analysis['id'],
                analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                f"{traits['Focus']:.1f}",
                f"{traits['Chaos']:.1f}",
                f"{traits['Energy']:.1f}",
                f"{traits['Discipline']:.1f}",
                f"{traits['Openness']:.1f}",
                f"{traits['Socialness']:.1f}",
                f"{traits['NightOwlness']:.1f}",
                dominant_theme,
                analysis['habit_keyword']
            ])
        
        return output.getvalue().encode('utf-8')


# Utility function for base64 encoding (useful for web export)
def encode_image_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')