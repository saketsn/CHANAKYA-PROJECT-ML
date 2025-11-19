import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from datetime import datetime
import warnings
import os

# --- Model & Transformer Imports ---
# Based on Activity 2, Part 1, we selected Random Forest.
from sklearn.ensemble import RandomForestRegressor
# If LightGBM had won, you would change the import above
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration: File paths ---
# Input file (from Activity 1)
RAW_FILTERED_CSV = 'raw_chanakya_filtered_by_age.csv' 

# Output files (to be created by this script)
FINAL_MODEL_FILE = 'final_video_predictor.joblib'
VECTORIZER_FILE = 'title_tfidf_vectorizer.joblib'
OHE_FILE = 'ohe_cats.joblib'

def build_final_model():
    """
    Trains the single best model (Random Forest) on the ENTIRE dataset
    and saves the model and all transformers to disk.
    """
    
    print("="*70)
    print("      ACTIVITY 2 (Part B): FINAL MODEL BUILDER")
    print("="*70)
    print("This script will train the champion model (Random Forest) on 100% \n"
          "of the data and save it for future use in Activity 3.")
    
    # --- 1. Load ALL Data ---
    print(f"\n[STATUS] Loading full dataset from {RAW_FILTERED_CSV}...")
    print("         (Using 100% of data ensures the final model is as accurate as possible)")
    
    if not os.path.exists(RAW_FILTERED_CSV):
        print(f"\n--- 🔴 ERROR: FILE NOT FOUND 🔴 ---")
        print(f"         File not found: {RAW_FILTERED_CSV}")
        print("         Please ensure you have run 'scrape_chanakya.py' from Activity 1 first.")
        print("---------------------------------------\n")
        return

    df = pd.read_csv(RAW_FILTERED_CSV)
    print(f"[OK]     Loaded all {len(df)} videos for final training.")

    # --- 2. Preprocessing and Feature Engineering ---
    print("\n[STATUS] Applying preprocessing and feature engineering to all data...")
    
    # Clean text data
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['upload_datetime'] = pd.to_datetime(df['upload_datetime'], errors='coerce')

    # Create time-based features
    now = datetime.now()
    df['Days_Since_Published'] = df['upload_datetime'].apply(lambda x: (now - x).days if pd.notnull(x) else np.nan)
    df['upload_hour'] = df['upload_datetime'].dt.hour.fillna(0).astype(int)
    df['upload_dow'] = df['upload_datetime'].dt.dayofweek.fillna(0).astype(int)

    # Create length features
    df['Title_Length'] = df['title'].str.len()
    df['Description_Length'] = df['description'].str.len()

    # Create Target variable
    TARGET = 'View_Count_Log'
    df[TARGET] = np.log1p(df['view_count'].astype(float))
    y_full = df[TARGET]
    print("  > [OK] All features engineered.")

    # --- 3. Create, Fit, and Save Feature Transformers ---
    print("\n[STATUS] Creating and saving 'Transformers'...")
    print("         (These files 'remember' the vocabulary and categories from the training data")
    print("          and are required to make predictions on new data in Activity 3.)")
    
    # A) Text Features (TF-IDF)
    print("  > Building TF-IDF vectorizer for titles...")
    tfidf = TfidfVectorizer(max_features=2000, min_df=3, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df['title'].astype(str))
    joblib.dump(tfidf, VECTORIZER_FILE)
    print(f"    - [SAVED] {VECTORIZER_FILE}")

    # B) Categorical Features (One-Hot Encoder)
    print("  > Building One-Hot Encoder for categories...")
    cat_cols = ['guest_type', 'topic_category', 'format_type']
    df[cat_cols] = df[cat_cols].fillna('Other').astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_ohe = ohe.fit_transform(df[cat_cols])
    joblib.dump(ohe, OHE_FILE)
    print(f"    - [SAVED] {OHE_FILE}")

    # C) Numerical Features
    print("  > Preparing numerical features...")
    num_cols = ['duration', 'Title_Length', 'Description_Length', 'upload_hour', 'upload_dow', 'Days_Since_Published']
    X_numeric = df[num_cols].fillna(0).values

    # --- 4. Assemble Final Training Data ---
    print("\n[STATUS] Assembling final data for training...")
    X_numeric_sparse = sparse.csr_matrix(X_numeric)
    X_ohe_sparse = sparse.csr_matrix(X_ohe)
    
    X_full = sparse.hstack([X_tfidf, X_numeric_sparse, X_ohe_sparse]).tocsr()
    print(f"[OK]     Final training data is ready (Shape: {X_full.shape})")

    # --- 5. Train and Save the FINAL Model ---
    print("\n[STATUS] Training the final Random Forest model on ALL data...")
    print("         (This may take a moment...)")
    
    # We use the parameters from our winning model
    final_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        min_samples_leaf=5
    )
    # We train it on ALL the data (X_full, y_full)
    final_model.fit(X_full, y_full)

    # Save the trained model to a file
    joblib.dump(final_model, FINAL_MODEL_FILE)
    print(f"  > [SAVED] Final model saved to: {FINAL_MODEL_FILE}")


    print("\n" + "="*70)
    print(" ✅  SUCCESS! ACTIVITY 2 IS COMPLETE  ✅")
    print("="*70)
    print("The system is now ready for Activity 3 (Prediction).")
    print("The following files have been created:")
    print(f"  1. {FINAL_MODEL_FILE} (The trained model)")
    print(f"  2. {VECTORIZER_FILE} (The title text processor)")
    print(f"  3. {OHE_FILE} (The category processor)")
    print("="*70)

if __name__ == '__main__':
    build_final_model()

