import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from datetime import datetime
import warnings
import os

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration: File paths ---
FINAL_MODEL_FILE = 'final_tuned_model.joblib'
VECTORIZER_FILE = 'final_tfidf_vectorizer.joblib'
OHE_FILE = 'final_ohe_cats.joblib'

# --- Configuration: Business Parameter ---
CPM = 80  # Average revenue per 1000 views in INR (can be changed as needed)

# ---------------------------------------------------------------------
# 1. Check if all required files are available
# ---------------------------------------------------------------------
def check_for_files():
    print("[STATUS] Checking for model files...")
    files_needed = [FINAL_MODEL_FILE, VECTORIZER_FILE, OHE_FILE]
    missing_files = [f for f in files_needed if not os.path.exists(f)]
    
    if missing_files:
        print("=" * 70)
        print("ERROR: Required files not found.")
        print("=" * 70)
        print("The following model files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run '3_tune_and_save_model.py' first to create these files.")
        print("=" * 70)
        return False
    print("[OK] All model files found. Ready to predict.")
    return True

# ---------------------------------------------------------------------
# 2. Predict Views and Estimated Revenue
# ---------------------------------------------------------------------
def predict_views_and_revenue(video_data):
    try:
        model = joblib.load(FINAL_MODEL_FILE)
        tfidf_vectorizer = joblib.load(VECTORIZER_FILE)
        ohe = joblib.load(OHE_FILE)
    except Exception as e:
        print(f"Error loading model files: {e}")
        return None, None

    df = pd.DataFrame([video_data])

    print(f"\n[INFO] Processing features for video:")
    print(f"Title: {df['title'].iloc[0]}")

    # Fill missing text values
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')

    # Time-based features
    df['upload_datetime'] = datetime.now()
    df['Days_Since_Published'] = 0
    df['upload_hour'] = df['upload_datetime'].dt.hour
    df['upload_dow'] = df['upload_datetime'].dt.dayofweek

    # Length features
    df['Title_Length'] = df['title'].str.len()
    df['Description_Length'] = df['description'].str.len()

    # Transform text data
    print(" - Creating TF-IDF features from title text...")
    X_tfidf = tfidf_vectorizer.transform(df['title'].astype(str))

    # Transform categorical data
    print(" - Encoding categorical features...")
    cat_cols = ['guest_type', 'topic_category', 'format_type']
    df[cat_cols] = df[cat_cols].fillna('Other').astype(str)
    X_ohe = ohe.transform(df[cat_cols])

    # Transform numerical data
    print(" - Combining numerical features...")
    num_cols = ['duration', 'Title_Length', 'Description_Length', 
                'upload_hour', 'upload_dow', 'Days_Since_Published']
    X_numeric = df[num_cols].fillna(0).values

    # Combine all features
    X_final = sparse.hstack([
        X_tfidf,
        sparse.csr_matrix(X_numeric),
        sparse.csr_matrix(X_ohe)
    ]).tocsr()

    print(" - All features successfully combined.")

    # Make prediction
    print("\n[STATUS] Generating prediction...")
    predicted_log_views = model.predict(X_final)
    predicted_views = np.expm1(predicted_log_views)
    predicted_views = int(predicted_views[0])

    # Estimate revenue
    estimated_revenue = (predicted_views / 1000) * CPM

    return predicted_views, round(estimated_revenue, 2)

# ---------------------------------------------------------------------
# 3. Main Program Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("        ACTIVITY 3 (Part B): VIEW & REVENUE PREDICTOR")
    print("=" * 70)

    if not check_for_files():
        exit()

    print("\nThis program estimates YouTube video performance before uploading.")
    print(f"Revenue is calculated at an average CPM of ₹{CPM} per 1000 views.")
    print("-" * 70)

    # Example 1
    new_video_1 = {
        'title': "China's new move in South China Sea | Maj Gen GD Bakshi | India's National Security",
        'description': "Major General G. D. Bakshi discusses the latest developments in the South China Sea and what it means for India's national security.",
        'duration': 1800,
        'guest_type': 'Military_Veteran',
        'topic_category': 'Geopolitics',
        'format_type': 'Interview'
    }

    print("\n------------------- PREDICTION 1 -------------------")
    predicted_views_1, estimated_income_1 = predict_views_and_revenue(new_video_1)
    if predicted_views_1 is not None:
        print("\nPrediction Result (Video 1)")
        print("-" * 70)
        print(f"Title: {new_video_1['title']}")
        print("Details: 30-min Interview | Guest: Military Veteran | Topic: Geopolitics")
        print(f"Estimated Views: {predicted_views_1:,}")
        print(f"Estimated Revenue: ₹{estimated_income_1:,}")
        print("-" * 70)

    # Example 2
    new_video_2 = {
        'title': "India's Economic Future | Can We Become a $5 Trillion Economy? | Expert Analysis",
        'description': "A deep dive into India's economic policies, challenges, and future opportunities. Renowned economist explains the path to becoming a $5 trillion economy.",
        'duration': 1500,
        'guest_type': 'Academic',
        'topic_category': 'Internal_Affairs',
        'format_type': 'Interview'
    }

    print("\n------------------- PREDICTION 2 -------------------")
    predicted_views_2, estimated_income_2 = predict_views_and_revenue(new_video_2)
    if predicted_views_2 is not None:
        print("\nPrediction Result (Video 2)")
        print("-" * 70)
        print(f"Title: {new_video_2['title']}")
        print("Details: 25-min Interview | Guest: Academic | Topic: Internal Affairs")
        print(f"Estimated Views: {predicted_views_2:,}")
        print(f"Estimated Revenue: ₹{estimated_income_2:,}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("All predictions completed successfully.")
    print("=" * 70)
    print(f"Note: Revenue is calculated at ₹{CPM} CPM (per 1000 views).")
    print("You can change the CPM value in the code to test different ad rates.")
    print("=" * 70)
