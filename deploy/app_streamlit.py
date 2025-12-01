import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from tensorflow import keras
import json
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Placeholder class
# ------------------------------------------------------------------------------
class FieldRecommendationSystem:
    def __init__(self):
        self.model_type = "neural_network"
        self.model = None

    def predict(self, board, combination, marks):
        raise NotImplementedError()


# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(page_title="RP Field Recommendation System", page_icon="🎓", layout="wide")


# ------------------------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------------------------
MODEL_DIR = "../results"
PREDICTIONS_LOG = "predictions_log.xlsx"
DATASET_FILE = os.path.join(MODEL_DIR, "../../dataset/rp_merged_dataset_cleaned_marks_to_80_where_was_1.json")


# ------------------------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models_and_encoders():
    field_encoder = joblib.load(os.path.join(MODEL_DIR, "field_encoder.pkl"))
    board_encoder = joblib.load(os.path.join(MODEL_DIR, "board_encoder.pkl"))
    combination_encoder = joblib.load(os.path.join(MODEL_DIR, "combination_encoder.pkl"))
    subject_scaler = joblib.load(os.path.join(MODEL_DIR, "subject_scaler.pkl"))
    board_ohe = joblib.load(os.path.join(MODEL_DIR, "board_ohe.pkl"))
    combination_ohe = joblib.load(os.path.join(MODEL_DIR, "combination_ohe.pkl"))
    subject_columns = joblib.load(os.path.join(MODEL_DIR, "subject_columns.pkl"))

    model_path_h5 = os.path.join(MODEL_DIR, "field_recommendation_model.h5")
    model_path_pkl = os.path.join(MODEL_DIR, "field_recommendation_model.pkl")

    if os.path.exists(model_path_h5):
        model = keras.models.load_model(model_path_h5)
        model_type = "neural_network"
    else:
        model = joblib.load(model_path_pkl)
        model_type = "tree_based"

    return {
        'model': model,
        'model_type': model_type,
        'field_encoder': field_encoder,
        'board_encoder': board_encoder,
        'combination_encoder': combination_encoder,
        'subject_scaler': subject_scaler,
        'board_ohe': board_ohe,
        'combination_ohe': combination_ohe,
        'subject_columns': subject_columns
    }


# ------------------------------------------------------------------------------
# LOAD SUBJECT MAPPING
# ------------------------------------------------------------------------------
@st.cache_resource
def load_board_combination_subjects():
    mapping = {}

    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = df[df["marks"].apply(lambda x: isinstance(x, dict))]

        for _, row in df.iterrows():
            board = row["examinationBoard"]
            combination = row["combination"]
            subjects = list(row["marks"].keys())

            if board not in mapping:
                mapping[board] = {}
            mapping[board][combination] = subjects
    else:
        mapping = {
            "RTB": {"BDC": [
                "Applied Mathematics B", "Construction Technical Drawing",
                "Reinforced Concrete Design", "Building elevation and Roof construction",
                "Applied Physics B", "Building materials and their applications",
                "Finishing works in building Construction", "Ikinyarwanda", "English",
                "Applied Chemistry A", "Entrepreneurship", "Practical BDC"
            ]}
        }

    return mapping


# ------------------------------------------------------------------------------
# ACADEMIC YEAR
# ------------------------------------------------------------------------------
def get_academic_year(key):
    now = datetime.now()
    y, m = now.year, now.month
    base_year = y + 1 if m >= 12 else y

    options = [
        f"{base_year}-{base_year+1} (Available)",
        f"{base_year+1}-{base_year+2} (Available)",
        f"{base_year-1}-{base_year} (Disabled)",
        f"{base_year+2}-{base_year+3} (Disabled)",
    ]

    selected = st.selectbox("Select Academic Year:", options, key=key)

    if "(Available)" not in selected:
        st.error("This academic year is disabled.")
        st.stop()

    return selected.split(" ")[0]


# ------------------------------------------------------------------------------
# SAVE LOG
# ------------------------------------------------------------------------------
def save_record_to_log(record_dict):
    df_new = pd.DataFrame([record_dict])

    if os.path.exists(PREDICTIONS_LOG):
        df_old = pd.read_excel(PREDICTIONS_LOG)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(PREDICTIONS_LOG, index=False)


# ------------------------------------------------------------------------------
# STATISTICS FUNCTIONS
# ------------------------------------------------------------------------------
def create_field_distribution_pie(df):
    """Create pie chart for field distribution"""
    field_counts = df['Predicted_Field'].value_counts()
    fig = px.pie(
        values=field_counts.values,
        names=field_counts.index,
        title='Distribution of Students Across Fields',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_board_distribution_bar(df):
    """Create bar chart for examination board distribution"""
    board_counts = df['Examination_Board'].value_counts()
    fig = px.bar(
        x=board_counts.index,
        y=board_counts.values,
        title='Students by Examination Board',
        labels={'x': 'Examination Board', 'y': 'Number of Students'},
        color=board_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False)
    return fig


def create_combination_distribution_bar(df):
    """Create bar chart for combination distribution"""
    comb_counts = df['Combination'].value_counts().head(15)
    fig = px.bar(
        x=comb_counts.values,
        y=comb_counts.index,
        orientation='h',
        title='Top 15 Combinations',
        labels={'x': 'Number of Students', 'y': 'Combination'},
        color=comb_counts.values,
        color_continuous_scale='Viridis'
    )
    return fig


def create_average_score_by_field_bar(df):
    """Create bar chart for average scores by field"""
    avg_scores = df.groupby('Predicted_Field')['Average_Score'].mean().sort_values(ascending=False)
    fig = px.bar(
        x=avg_scores.index,
        y=avg_scores.values,
        title='Average Score by Field',
        labels={'x': 'Field', 'y': 'Average Score'},
        color=avg_scores.values,
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def create_confidence_distribution_histogram(df):
    """Create histogram for confidence distribution"""
    fig = px.histogram(
        df,
        x='Confidence',
        nbins=20,
        title='Distribution of Prediction Confidence',
        labels={'Confidence': 'Confidence Score (%)', 'count': 'Number of Students'},
        color_discrete_sequence=['#636EFA']
    )
    fig.add_vline(x=df['Confidence'].mean(), line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {df['Confidence'].mean():.2f}%")
    return fig


def create_admission_year_trend(df):
    """Create line chart for admission year trends"""
    year_counts = df['Admission_Year'].value_counts().sort_index()
    fig = px.line(
        x=year_counts.index,
        y=year_counts.values,
        title='Student Admissions Over Time',
        labels={'x': 'Admission Year', 'y': 'Number of Students'},
        markers=True
    )
    fig.update_traces(line_color='#FF6692', line_width=3)
    return fig


def create_field_by_board_stacked_bar(df):
    """Create stacked bar chart for fields by board"""
    cross_tab = pd.crosstab(df['Examination_Board'], df['Predicted_Field'])
    fig = px.bar(
        cross_tab,
        title='Field Distribution by Examination Board',
        labels={'value': 'Number of Students', 'Examination_Board': 'Board'},
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    return fig


def create_score_range_distribution(df):
    """Create pie chart for score range distribution"""
    bins = [0, 50, 60, 70, 80, 90, 100]
    labels = ['0-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    df['Score_Range'] = pd.cut(df['Average_Score'], bins=bins, labels=labels, include_lowest=True)
    range_counts = df['Score_Range'].value_counts().sort_index()
    
    fig = px.pie(
        values=range_counts.values,
        names=range_counts.index,
        title='Students by Score Range',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    return fig


def create_confidence_by_field_box(df):
    """Create box plot for confidence by field"""
    fig = px.box(
        df,
        x='Predicted_Field',
        y='Confidence',
        title='Prediction Confidence Distribution by Field',
        labels={'Predicted_Field': 'Field', 'Confidence': 'Confidence (%)'},
        color='Predicted_Field',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    return fig


def create_top_fields_timeline(df):
    """Create area chart for top fields over time"""
    df_copy = df.copy()
    top_fields = df_copy['Predicted_Field'].value_counts().head(5).index
    df_filtered = df_copy[df_copy['Predicted_Field'].isin(top_fields)]
    
    timeline = df_filtered.groupby(['Admission_Year', 'Predicted_Field']).size().reset_index(name='Count')
    
    fig = px.area(
        timeline,
        x='Admission_Year',
        y='Count',
        color='Predicted_Field',
        title='Top 5 Fields Trend Over Time',
        labels={'Count': 'Number of Students', 'Admission_Year': 'Year'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    return fig


def create_combination_field_heatmap(df):
    """Create heatmap for combination vs field"""
    # Get top combinations and fields
    top_combinations = df['Combination'].value_counts().head(10).index
    top_fields = df['Predicted_Field'].value_counts().head(10).index
    
    df_filtered = df[df['Combination'].isin(top_combinations) & df['Predicted_Field'].isin(top_fields)]
    
    heatmap_data = pd.crosstab(df_filtered['Combination'], df_filtered['Predicted_Field'])
    
    fig = px.imshow(
        heatmap_data,
        title='Heatmap: Top 10 Combinations vs Top 10 Fields',
        labels=dict(x="Field", y="Combination", color="Count"),
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig.update_xaxes(tickangle=-45)
    return fig


# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------
def main():

    models = load_models_and_encoders()
    mapping = load_board_combination_subjects()

    st.title("🎓 Rwanda Polytechnic Field Recommendation System")

    tab_single, tab_batch, tab_statistics, tab_history, tab_about = st.tabs(
        ["Single Prediction", "Batch Prediction", "Statistics", "History", "About"]
    )

    # --------------------------------------------------------------------------
    # SINGLE PREDICTION
    # --------------------------------------------------------------------------
    with tab_single:

        col1, col2 = st.columns(2)
        with col1:
            board = st.selectbox("Examination Board:", [""] + list(mapping.keys()))

        with col2:
            combinations = mapping[board].keys() if board else []
            combination = st.selectbox("Combination:", [""] + list(combinations))

        if board and combination:

            subjects = mapping[board][combination]
            st.subheader("Enter Subject Marks (0–100)")
            marks = {}

            cols = st.columns(3)
            for i, s in enumerate(subjects):
                with cols[i % 3]:
                    marks[s] = st.number_input(s, 0, 100, 0)

            academic_year = get_academic_year("single_year")

            if st.button("🔍 Predict Field"):

                try:
                    # Encode board/combination
                    board_enc = models["board_encoder"].transform([board])[0]
                    comb_enc = models["combination_encoder"].transform([combination])[0]

                    board_ohe = models["board_ohe"].transform([[board_enc]])
                    comb_ohe = models["combination_ohe"].transform([[comb_enc]])

                    subj_vec = np.array([marks.get(s, 0) for s in models["subject_columns"]])
                    subj_scaled = models["subject_scaler"].transform(subj_vec.reshape(1, -1))

                    X = np.concatenate([board_ohe, comb_ohe, subj_scaled], axis=1)

                    # Prediction
                    model = models["model"]
                    if models["model_type"] == "neural_network":
                        proba = model.predict(X, verbose=0)[0]
                    else:
                        proba = model.predict_proba(X)[0]

                    pred_idx = np.argmax(proba)
                    predicted_field = models["field_encoder"].inverse_transform([pred_idx])[0]

                    confidence = round(float(proba[pred_idx]) * 100, 2)
                    avg_score = round(float(np.mean(list(marks.values()))), 2)

                    top3_idx = np.argsort(proba)[-3:][::-1]
                    top3 = [(models["field_encoder"].inverse_transform([i])[0],
                             round(float(proba[i]) * 100, 2)) for i in top3_idx]

                    st.success(f"🎯 Recommended Field: **{predicted_field}**")
                    st.metric("Confidence (out of 100)", confidence)
                    st.metric("Average Score (out of 100)", avg_score)

                    st.subheader("Top 3 Predictions")
                    for i, (f, c) in enumerate(top3, 1):
                        st.write(f"{i}. **{f}** — {c}/100")

                    record = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Admission_Year": academic_year,
                        "Examination_Board": board,
                        "Combination": combination,
                        "Predicted_Field": predicted_field,
                        "Confidence": confidence,
                        "Average_Score": avg_score,
                        "Top_3_Predictions": ", ".join([f"{f} ({c}/100)" for f, c in top3]),
                    }

                    for s in models["subject_columns"]:
                        record[s] = marks.get(s, 0)

                    save_record_to_log(record)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    # --------------------------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------------------------
    with tab_batch:

        st.subheader("📁 Batch Prediction")

        sample_template = pd.DataFrame([
            {       "examinationBoard": "RTB",
                    "combination": "BDC",
                    "Applied Mathematics B": 96,
                    "Construction Technical Drawing": 79,
                    "Reinforced Concrete Design": 99,
                    "Building elevation and Roof construction": 97,
                    "Applied Physics B": 95,
                    "Building materials and their applications": 90,
                    "Finishing works in building Construction": 100,
                    "Ikinyarwanda": 99,
                    "English": 98,
                    "Applied Chemistry A": 93,
                    "Entrepreneurship": 100,
                    "Practical BDC": 100}
        ])

        st.download_button(
            "⬇ Download Batch CSV Template",
            sample_template.to_csv(index=False),
            "batch_template.csv",
            mime="text/csv"
        )

        uploaded = st.file_uploader("Upload Batch CSV", type=["csv"])

        if uploaded:
            df_batch = pd.read_csv(uploaded)
            st.dataframe(df_batch)

            batch_year = get_academic_year("batch_year")

            if st.button("⚡ Run Batch Prediction"):
                results = []

                for _, row in df_batch.iterrows():
                    try:
                        board = row["examinationBoard"]
                        combination = row["combination"]

                        marks = {s: row[s] if s in row else 0 for s in models["subject_columns"]}

                        # Average on non-zero marks only
                        non_zero = [v for v in marks.values() if v > 0]
                        avg_score = round(float(np.mean(non_zero)), 2) if non_zero else 0

                        board_enc = models["board_encoder"].transform([board])[0]
                        comb_enc = models["combination_encoder"].transform([combination])[0]

                        board_ohe = models["board_ohe"].transform([[board_enc]])
                        comb_ohe = models["combination_ohe"].transform([[comb_enc]])

                        subj_vec = np.array([marks[s] for s in models["subject_columns"]])
                        subj_scaled = models["subject_scaler"].transform(subj_vec.reshape(1, -1))

                        X = np.concatenate([board_ohe, comb_ohe, subj_scaled], axis=1)

                        model = models["model"]
                        proba = model.predict(X, verbose=0)[0] if models["model_type"] == "neural_network" else model.predict_proba(X)[0]

                        pred_idx = np.argmax(proba)
                        predicted_field = models["field_encoder"].inverse_transform([pred_idx])[0]
                        confidence = round(float(proba[pred_idx]) * 100, 2)

                        top3_idx = np.argsort(proba)[-3:][::-1]
                        top3 = [(models["field_encoder"].inverse_transform([i])[0],
                                 round(float(proba[i]) * 100, 2)) for i in top3_idx]

                        record = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Admission_Year": batch_year,
                            "Examination_Board": board,
                            "Combination": combination,
                            "Predicted_Field": predicted_field,
                            "Confidence": confidence,
                            "Average_Score": avg_score,
                            "Top_3_Predictions": ", ".join([f"{f} ({c}/100)" for f, c in top3]),
                        }

                        for s in models["subject_columns"]:
                            record[s] = marks.get(s, 0)

                        results.append(record)
                        save_record_to_log(record)

                    except Exception as e:
                        st.error(f"Row error: {e}")

                results_df = pd.DataFrame(results)
                st.subheader("Batch Results")
                st.dataframe(results_df)

                st.download_button(
                    "⬇ Download Batch Results",
                    results_df.to_csv(index=False),
                    "batch_predictions.csv",
                    mime="text/csv"
                )

    # --------------------------------------------------------------------------
    # STATISTICS TAB
    # --------------------------------------------------------------------------
    with tab_statistics:
        st.header("📊 Student Admission Statistics")
        
        if not os.path.exists(PREDICTIONS_LOG):
            st.warning("No prediction data available yet. Make some predictions first!")
        else:
            df_stats = pd.read_excel(PREDICTIONS_LOG)
            
            if df_stats.empty:
                st.warning("No data to display. The predictions log is empty.")
            else:
                # Key Metrics
                st.subheader("Key Metrics Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Students", len(df_stats))
                with col2:
                    st.metric("Average Score", f"{df_stats['Average_Score'].mean():.2f}")
                with col3:
                    st.metric("Average Confidence", f"{df_stats['Confidence'].mean():.2f}%")
                with col4:
                    st.metric("Unique Fields", df_stats['Predicted_Field'].nunique())
                
                st.divider()
                
                # Filter Section
                st.subheader("🔍 Filter Data")
                col_f1, col_f2, col_f3 = st.columns(3)
                
                with col_f1:
                    selected_years = st.multiselect(
                        "Admission Year",
                        options=df_stats['Admission_Year'].unique(),
                        default=df_stats['Admission_Year'].unique()
                    )
                
                with col_f2:
                    selected_boards = st.multiselect(
                        "Examination Board",
                        options=df_stats['Examination_Board'].unique(),
                        default=df_stats['Examination_Board'].unique()
                    )
                
                with col_f3:
                    selected_fields = st.multiselect(
                        "Predicted Field",
                        options=df_stats['Predicted_Field'].unique(),
                        default=df_stats['Predicted_Field'].unique()
                    )
                
                # Apply filters
                df_filtered = df_stats[
                    (df_stats['Admission_Year'].isin(selected_years)) &
                    (df_stats['Examination_Board'].isin(selected_boards)) &
                    (df_stats['Predicted_Field'].isin(selected_fields))
                ]
                
                if df_filtered.empty:
                    st.warning("No data matches the selected filters.")
                else:
                    st.divider()
                    
                    # Charts Section
                    st.subheader("📈 Visual Analytics")
                    
                    # Row 1: Field Distribution and Board Distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_field_distribution_pie(df_filtered), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_board_distribution_bar(df_filtered), use_container_width=True)
                    
                    # Row 2: Combination Distribution and Score by Field
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_combination_distribution_bar(df_filtered), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_average_score_by_field_bar(df_filtered), use_container_width=True)
                    
                    # Row 3: Confidence Distribution and Score Range
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_confidence_distribution_histogram(df_filtered), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_score_range_distribution(df_filtered), use_container_width=True)
                    
                    # Row 4: Field by Board and Confidence by Field
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_field_by_board_stacked_bar(df_filtered), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_confidence_by_field_box(df_filtered), use_container_width=True)
                    
                    # Row 5: Timeline charts
                    if len(df_filtered['Admission_Year'].unique()) > 1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(create_admission_year_trend(df_filtered), use_container_width=True)
                        with col2:
                            st.plotly_chart(create_top_fields_timeline(df_filtered), use_container_width=True)
                    
                    # Row 6: Heatmap (full width)
                    st.plotly_chart(create_combination_field_heatmap(df_filtered), use_container_width=True)
                    
                    st.divider()
                    
                    # Detailed Statistics Table
                    st.subheader("📋 Detailed Statistics by Field")
                    field_stats = df_filtered.groupby('Predicted_Field').agg({
                        'Predicted_Field': 'count',
                        'Average_Score': 'mean',
                        'Confidence': 'mean'
                    }).rename(columns={
                        'Predicted_Field': 'Student Count',
                        'Average_Score': 'Avg Score',
                        'Confidence': 'Avg Confidence'
                    }).round(2).sort_values('Student Count', ascending=False)
                    
                    st.dataframe(field_stats, use_container_width=True)
                    
                    # Download Reports
                    st.divider()
                    st.subheader("📥 Download Reports")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "⬇ Download Filtered Data (CSV)",
                            df_filtered.to_csv(index=False),
                            "filtered_statistics.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            "⬇ Download Summary Statistics (CSV)",
                            field_stats.to_csv(),
                            "summary_statistics.csv",
                            mime="text/csv"
                        )

    # --------------------------------------------------------------------------
    # HISTORY TAB
    # --------------------------------------------------------------------------
    with tab_history:
        st.subheader("📜 Prediction History")
        if os.path.exists(PREDICTIONS_LOG):
            df_history = pd.read_excel(PREDICTIONS_LOG)
            
            # Search and filter options
            search_term = st.text_input("🔍 Search (Field, Board, Combination)", "")
            
            if search_term:
                mask = (
                    df_history['Predicted_Field'].str.contains(search_term, case=False, na=False) |
                    df_history['Examination_Board'].str.contains(search_term, case=False, na=False) |
                    df_history['Combination'].str.contains(search_term, case=False, na=False)
                )
                df_history = df_history[mask]
            
            st.dataframe(df_history, use_container_width=True)
            
            st.download_button(
                "⬇ Download Full History",
                df_history.to_csv(index=False),
                "prediction_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions saved yet.")

    # --------------------------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------------------------
    with tab_about:
        st.markdown("""
        ## 🎓 Rwanda Polytechnic Field Recommendation System
        
        ### Overview
        This AI-driven system helps students and administrators make informed decisions about field 
        selection based on academic performance, examination board, and subject combinations.
        
        ### Features
        - **Single Prediction**: Get instant field recommendations for individual students
        - **Batch Prediction**: Process multiple students simultaneously via CSV upload
        - **Statistics Dashboard**: Comprehensive analytics with interactive charts and reports
        - **History Tracking**: View and search all past predictions
        
        ### Technology Stack
        - Machine Learning: TensorFlow/Keras Neural Networks
        - Data Processing: Pandas, NumPy
        - Visualization: Plotly
        - Interface: Streamlit
        
        ### How It Works
        1. Students input their examination board, combination, and subject marks
        2. The AI model analyzes the data using trained neural networks
        3. System provides field recommendations with confidence scores
        4. All predictions are logged for statistical analysis
        
        ### Statistics Features
        The Statistics tab provides:
        - Field distribution analysis
        - Performance metrics by field
        - Examination board comparisons
        - Trend analysis over admission years
        - Interactive filtering and downloadable reports
        
        ---
        **Version**: 2.0  
        **Last Updated**: December 2024  
        **Developed for**: Rwanda Polytechnic
        """)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()