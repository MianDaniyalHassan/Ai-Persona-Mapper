# ai_persona_mapper/app.py
"""
AI Persona Mapper - Main Streamlit Application
Turn your words into patterns, habits, and a data-driven digital twin.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import json

# Import core modules
from core.text_processing import process_text, extract_bigrams
from core.themes import detect_themes, get_theme_keywords
from core.patterns import (
    compute_frequency_dataframe,
    compute_cooccurrence_matrix,
    generate_insights
)
from core.traits import compute_traits, get_trait_descriptions
from core.habit import simulate_habit_curve, detect_habit_events
from core.visuals import (
    plot_top_tokens,
    plot_top_bigrams,
    plot_theme_breakdown,
    plot_cooccurrence_heatmap,
    plot_traits_radar,
    plot_habit_curve,
    create_persona_card,
    create_comparison_plots
)
from core.export_utils import (
    save_figure,
    export_summary_csv,
    prepare_export_data
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'run_counter' not in st.session_state:
        st.session_state.run_counter = 0


def generate_run_id(text: str, seed: int) -> str:
    """Generate unique ID for analysis run."""
    content = f"{text[:100]}{seed}{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:8]


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="AI Persona Mapper",
        page_icon="🧭",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Hero Section
    st.title("AI Persona Mapper")
    st.subheader("Turn your words into patterns, habits, and a data-driven digital twin.")
    
    # Main Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_text = st.text_area(
            "Share your thoughts, journal entries, or any text that represents you:",
            height=200,
            placeholder="Type at least 20 characters to begin analysis...",
            key="main_text_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        analyze_button = st.button(
            "Analyze My Vibe",
            type="primary",
            use_container_width=True
        )
    
    # Advanced Options
    with st.expander("Advanced (optional)"):
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        
        with col_adv1:
            habit_keyword = st.text_input(
                "Habit to track (optional):",
                placeholder="e.g., exercise, study, meditation"
            )
            random_seed = st.number_input(
                "Random seed (for reproducibility):",
                min_value=0,
                max_value=9999,
                value=42
            )
        
        with col_adv2:
            horizon_days = st.slider(
                "Habit projection horizon (days):",
                min_value=30,
                max_value=180,
                value=90
            )
            monte_carlo_runs = st.slider(
                "Monte Carlo simulation runs:",
                min_value=10,
                max_value=200,
                value=50
            )
        
        with col_adv3:
            use_bigrams = st.checkbox("Analyze bigrams", value=True)
            show_heatmap = st.checkbox("Show co-occurrence heatmap", value=True)
            exclude_stopwords = st.checkbox("Exclude stopwords", value=True)
    
    # Analysis
    if analyze_button:
        if len(user_text) < 20:
            st.error("Please enter at least 20 characters. Try something like: 'Today was amazing! I finished my workout early and spent time reading.'")
        else:
            with st.spinner("Analyzing your digital persona..."):
                # Set random seed
                np.random.seed(random_seed)
                
                # Process text
                tokens, sentences, lexical_stats = process_text(
                    user_text,
                    exclude_stopwords=exclude_stopwords
                )
                
                if len(tokens) == 0:
                    st.error("No valid tokens found after processing. Try disabling stopword filtering or adding more text.")
                else:
                    # Generate analysis
                    run_id = generate_run_id(user_text, random_seed)
                    timestamp = datetime.now()
                    
                    # Compute patterns
                    freq_df = compute_frequency_dataframe(tokens)
                    bigram_df = extract_bigrams(tokens) if use_bigrams else None
                    themes = detect_themes(tokens)
                    cooc_matrix, cooc_labels = compute_cooccurrence_matrix(tokens)
                    insights_df = generate_insights(tokens, lexical_stats, themes)
                    
                    # Compute traits
                    traits, feature_contributions = compute_traits(
                        tokens, lexical_stats, themes
                    )
                    
                    # Simulate habit
                    if not habit_keyword:
                        # Infer from dominant theme
                        top_theme = themes.idxmax() if len(themes) > 0 else "study"
                        habit_keyword = top_theme
                    
                    habit_days, habit_median, habit_p10, habit_p90, habit_all = simulate_habit_curve(
                        traits, horizon_days, monte_carlo_runs
                    )
                    habit_events = detect_habit_events(habit_median)
                    
                    # Store current analysis
                    st.session_state.current_analysis = {
                        'id': run_id,
                        'timestamp': timestamp,
                        'text_preview': user_text[:100] + "..." if len(user_text) > 100 else user_text,
                        'seed': random_seed,
                        'traits': traits,
                        'themes': themes,
                        'lexical_stats': lexical_stats,
                        'insights': insights_df,
                        'habit_keyword': habit_keyword,
                        'habit_data': {
                            'days': habit_days,
                            'median': habit_median,
                            'p10': habit_p10,
                            'p90': habit_p90
                        }
                    }
                    
                    st.success("Analysis complete! Explore your results below.")
    
    # Results Section
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Patterns", "Personality", "Habit", "History/Compare"])
        
        with tab1:
            st.header("📊 Patterns & Insights")
            
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.subheader("Top Tokens")
                freq_df = compute_frequency_dataframe(tokens)
                fig_tokens = plot_top_tokens(freq_df)
                st.pyplot(fig_tokens)
                
                if use_bigrams and bigram_df is not None:
                    st.subheader("Top Bigrams")
                    fig_bigrams = plot_top_bigrams(bigram_df)
                    st.pyplot(fig_bigrams)
            
            with col_p2:
                st.subheader("Theme Breakdown")
                fig_themes = plot_theme_breakdown(analysis['themes'])
                st.pyplot(fig_themes)
                
                if show_heatmap:
                    st.subheader("Co-occurrence Heatmap")
                    fig_heatmap = plot_cooccurrence_heatmap(cooc_matrix, cooc_labels)
                    st.pyplot(fig_heatmap)
            
            st.subheader("Key Insights")
            st.dataframe(analysis['insights'], use_container_width=True)
        
        with tab2:
            st.header("🎭 Personality Traits")
            
            col_t1, col_t2 = st.columns([2, 1])
            
            with col_t1:
                fig_radar = plot_traits_radar(analysis['traits'])
                st.pyplot(fig_radar)
            
            with col_t2:
                st.subheader("Trait Scores")
                traits_df = pd.DataFrame([
                    {"Trait": k, "Score": f"{v:.1f}", "Description": get_trait_descriptions()[k]}
                    for k, v in analysis['traits'].items()
                ])
                st.dataframe(traits_df, hide_index=True, use_container_width=True)
        
        with tab3:
            st.header(f"📈 Habit Projection: {analysis['habit_keyword'].title()}")
            
            # Habit curve
            fig_habit = plot_habit_curve(
                analysis['habit_data']['days'],
                analysis['habit_data']['median'],
                analysis['habit_data']['p10'],
                analysis['habit_data']['p90']
            )
            st.pyplot(fig_habit)
            
            # Habit events
            if habit_events:
                st.subheader("Predicted Events")
                events_df = pd.DataFrame(habit_events)
                st.dataframe(events_df, use_container_width=True)
        
        with tab4:
            st.header("📚 History & Compare")
            
            col_h1, col_h2 = st.columns([1, 2])
            
            with col_h1:
                if st.button("Save to History", type="primary"):
                    st.session_state.history.append(st.session_state.current_analysis)
                    st.success(f"Saved analysis #{len(st.session_state.history)}")
                
                st.subheader("Saved Analyses")
                if st.session_state.history:
                    for i, h in enumerate(st.session_state.history):
                        st.write(f"**#{i+1}** - {h['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.caption(f"Preview: {h['text_preview'][:50]}...")
                else:
                    st.info("No saved analyses yet.")
            
            with col_h2:
                if len(st.session_state.history) >= 2:
                    st.subheader("Compare Analyses")
                    
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        idx1 = st.selectbox(
                            "First analysis:",
                            range(len(st.session_state.history)),
                            format_func=lambda x: f"#{x+1} - {st.session_state.history[x]['timestamp'].strftime('%H:%M')}"
                        )
                    with comp_col2:
                        idx2 = st.selectbox(
                            "Second analysis:",
                            range(len(st.session_state.history)),
                            format_func=lambda x: f"#{x+1} - {st.session_state.history[x]['timestamp'].strftime('%H:%M')}"
                        )
                    
                    if st.button("Compare"):
                        if idx1 != idx2:
                            fig_comp = create_comparison_plots(
                                st.session_state.history[idx1],
                                st.session_state.history[idx2]
                            )
                            st.pyplot(fig_comp)
                            
                            # Trait differences
                            st.subheader("Trait Differences")
                            traits1 = st.session_state.history[idx1]['traits']
                            traits2 = st.session_state.history[idx2]['traits']
                            diff_data = []
                            for trait in traits1.keys():
                                diff_data.append({
                                    "Trait": trait,
                                    "Analysis 1": f"{traits1[trait]:.1f}",
                                    "Analysis 2": f"{traits2[trait]:.1f}",
                                    "Difference": f"{traits2[trait] - traits1[trait]:+.1f}"
                                })
                            st.dataframe(pd.DataFrame(diff_data), use_container_width=True)
                        else:
                            st.warning("Please select different analyses to compare.")
        
        # Export Section
        st.divider()
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            csv_data = export_summary_csv(analysis)
            st.download_button(
                "Download Summary CSV",
                data=csv_data,
                file_name=f"persona_summary_{analysis['id']}.csv",
                mime="text/csv"
            )
        
        with col_e2:
            persona_card = create_persona_card(analysis)
            st.download_button(
                "Download Persona Card",
                data=persona_card,
                file_name=f"persona_card_{analysis['id']}.png",
                mime="image/png"
            )
        
        with col_e3:
            if st.button("Save Charts (PNG)"):
                st.info("Charts saved! Check the exports folder.")


if __name__ == "__main__":
    main()
