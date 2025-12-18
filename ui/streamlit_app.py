"""
SHL Assessment Recommendation Engine - Streamlit UI
A modern, user-friendly interface for getting assessment recommendations.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation.recommender import Recommender

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CATALOG_PATH = str(PROJECT_ROOT / "data" / "shl_catalog.json")
RULES_PATH = str(PROJECT_ROOT / "config" / "rules.yaml")
MODEL_PATH = str(PROJECT_ROOT / "models" / "ranking_model.pkl")


@st.cache_data
def load_catalog():
    """Load and preprocess the SHL assessment catalog."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "shl_product_catalog" in raw:
        data = raw["shl_product_catalog"]["products"]
    else:
        data = raw
    df = pd.DataFrame(data)
    if "skills" in df.columns:
        df["skills"] = df["skills"].apply(
            lambda x: [s.strip() for s in x.split(",")] if isinstance(x, str) else x
        )
    return df


@st.cache_resource
def get_recommender():
    """Initialize the recommender (cached for performance)."""
    return Recommender(
        catalog_path=CATALOG_PATH,
        rules_path=RULES_PATH,
        model_path=MODEL_PATH,
        top_n=10,
    )


def extract_all_skills(catalog_df):
    """Extract unique skills from catalog."""
    all_skills_set = set()
    if not catalog_df.empty and "skills" in catalog_df.columns:
        for skills_val in catalog_df["skills"]:
            if isinstance(skills_val, list):
                all_skills_set.update(skills_val)
            elif isinstance(skills_val, str):
                all_skills_set.update([s.strip() for s in skills_val.split(",")])
    return sorted(all_skills_set)


def main():
    # Page config
    st.set_page_config(
        page_title="SHL Assessment Recommender",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #64748B;
            margin-bottom: 2rem;
        }
        .score-badge {
            background-color: #10B981;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .metric-container {
            background-color: #F8FAFC;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3B82F6;
        }
        .stButton>button {
            background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header"> SHL Assessment Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"> by ML Recommendation engine </p>', unsafe_allow_html=True)

    # Load data
    catalog_df = load_catalog()
    all_skills = extract_all_skills(catalog_df)
    recommender = get_recommender()

    # Sidebar for inputs
    with st.sidebar:
        st.header(" Job Requirements")
        st.markdown("---")
        
        job_role = st.text_input(
            " Job Role",
            value="Account Manager",
            help="Enter the job title you're hiring for"
        )
        
        job_level = st.selectbox(
            " Job Level",
            options=["Entry-Level", "Entry to Mid", "Mid-Professional", "All levels"],
            index=2,
            help="Select the seniority level for the position"
        )
        
        hiring_stage = st.selectbox(
            " Hiring Stage",
            options=["Screening", "Interview"],
            index=0,
            help="Screening: Early stage (prefer shorter tests)\nInterview: Later stage (detailed assessments)"
        )
        
        st.markdown("---")
        
        skills = st.multiselect(
            " Required Skills",
            options=all_skills,
            default=[s for s in ["client communication", "project coordination"] if s in all_skills],
            help="Select skills required for the job"
        )
        
        st.markdown("---")
        
        top_n = st.slider(
            " Number of Recommendations",
            min_value=3,
            max_value=10,
            value=5,
            help="How many assessments to recommend"
        )
        
        st.markdown("---")
        
        recommend_btn = st.button(" Get Recommendations", use_container_width=True)

    # Main content area
    col_main, col_info = st.columns([3, 1])

    with col_info:
        st.markdown("###  Model Info")
        st.markdown("""
        <div class="metric-container">
            <strong>Algorithm:</strong> Gradient Boosted Trees<br>
            <strong>Features:</strong> 7 engineered features<br>
            <strong>Catalog:</strong> 24 SHL assessments<br>
            <strong>Scoring:</strong> ML (60%) + Similarity (40%)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### How It Works")
        st.markdown("""
        1. **Rules Filter**: Business logic (job level, duration)
        2. **Similarity**: TF-IDF text matching
        3. **ML Ranking**: Gradient Boosted prediction
        4. **Final Score**: Weighted combination
        """)

    with col_main:
        if recommend_btn:
            if not skills:
                st.warning("⚠️ Please select at least one skill.")
            else:
                job_req = {
                    "job_role": job_role,
                    "job_level": job_level,
                    "skills": skills,
                    "hiring_stage": hiring_stage,
                }

                with st.spinner(" Analyzing assessments..."):
                    recommender.top_n = top_n
                    results = recommender.recommend(job_req)

                if not results:
                    st.error(" No recommendations found. Try adjusting your criteria.")
                else:
                    st.success(f"✅ Found {len(results)} recommended assessments!")
                    st.markdown("---")

                    for i, rec in enumerate(results, 1):
                        with st.container():
                            col_rank, col_name, col_score = st.columns([0.5, 4, 1.5])
                            
                            with col_rank:
                                if i == 1:
                                    st.markdown(f"### 🥇")
                                elif i == 2:
                                    st.markdown(f"### 🥈")
                                elif i == 3:
                                    st.markdown(f"### 🥉")
                                else:
                                    st.markdown(f"### #{i}")
                            
                            with col_name:
                                st.markdown(f"### {rec['assessment_name']}")
                            
                            with col_score:
                                score_pct = rec['final_score'] * 100
                                if score_pct >= 70:
                                    st.markdown(f"<span class='score-badge' style='background-color: #10B981;'>{score_pct:.1f}%</span>", unsafe_allow_html=True)
                                elif score_pct >= 50:
                                    st.markdown(f"<span class='score-badge' style='background-color: #F59E0B;'>{score_pct:.1f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span class='score-badge' style='background-color: #EF4444;'>{score_pct:.1f}%</span>", unsafe_allow_html=True)

                            with st.expander("View Score Breakdown"):
                                exp = rec["explanation"]
                                
                                col_a, col_b, col_c = st.columns(3)
                                
                                with col_a:
                                    st.metric("ML Score", f"{exp['ml_score']:.3f}", help="Machine learning prediction (0-1)")
                                
                                with col_b:
                                    st.metric("Similarity", f"{exp['similarity_score']:.3f}", help="TF-IDF text similarity (0-1)")
                                
                                with col_c:
                                    st.metric("Rule Score", f"{exp['rule_score']:.2f}", help="Business rules score")

                                st.markdown("**Rule Details:**")
                                rule_cols = st.columns(4)
                                with rule_cols[0]:
                                    status = "✅" if exp.get('job_level_pass', False) else "❌"
                                    st.write(f"Level Match: {status}")
                                with rule_cols[1]:
                                    st.write(f"Level Score: {exp.get('job_level_score', 0):.2f}")
                                with rule_cols[2]:
                                    st.write(f"Duration Penalty: {exp.get('duration_penalty', 0):.2f}")
                                with rule_cols[3]:
                                    st.write(f"Domain Boost: {exp.get('domain_boost', 0):.2f}")

                            st.markdown("---")

        else:
            st.markdown("""
            ###  Welcome!
            
            Use the sidebar to enter job requirements and click **Get Recommendations** to find the best SHL assessments.
            
            **Quick Start:**
            1. Enter a job role (e.g., "Account Manager", "Software Developer")
            2. Select the job level and hiring stage
            3. Choose required skills from the dropdown
            4. Click the button to get personalized recommendations
            
            ---
            
            ####  About This System
            
            This recommendation engine uses a **hybrid approach**:
            - **Rule-based filtering**: Applies business logic (job level matching, duration constraints)
            - **Text similarity**: Matches job requirements to assessment descriptions using TF-IDF
            - **Machine learning**: Gradient Boosted Trees model trained on engineered features
            
            The final score combines ML prediction (60%) and text similarity (40%) for optimal ranking.
            """)

            with st.expander(" View Assessment Catalog"):
                display_df = catalog_df[["assessment_name", "job_level", "category", "duration_minutes"]].copy()
                display_df.columns = ["Assessment", "Job Level", "Category", "Duration (min)"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
