import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from time import time
import warnings
from datetime import datetime 
import os

# --- Model & Transformer Imports ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
# Using RandomizedSearchCV for efficient tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- Configuration: File paths ---
# Input file (from Activity 1)
RAW_FILTERED_CSV = 'raw_chanakya_filtered_by_age.csv' 

# Output files this script WILL CREATE (with new names for the tuned versions)
FINAL_MODEL_FILE = 'final_tuned_model.joblib'
FINAL_VECTORIZER_FILE = 'final_tfidf_vectorizer.joblib'
FINAL_OHE_FILE = 'final_ohe_cats.joblib'

def tune_and_save_model():
    """
    Loads all data, runs hyperparameter tuning, re-trains the best model
    on the entire dataset, and saves the final model and transformers.
    """
    
    print("="*70)
    print("      ACTIVITY 3 (Part A): MODEL HYPERPARAMETER TUNING")
    print("="*70)
    print("This script will find the best 'settings' for our Random Forest")
    print("model to make it as accurate as possible. This is the 'expert tuning' step.")
    
    # --- 1. Load ALL Data ---
    print(f"\n[STATUS] Loading full dataset from {RAW_FILTERED_CSV}...")
    print("         (We will re-fit transformers on 100% of data for final use)")
    
    if not os.path.exists(RAW_FILTERED_CSV):
        print(f"\n--- 🔴 ERROR: FILE NOT FOUND 🔴 ---")
        print(f"         File not found: {RAW_FILTERED_CSV}")
        print("         Please ensure 'scrape_chanakya.py' ran successfully.")
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

    # --- 3. Create, Fit, and Save FINAL Feature Transformers ---
    print("\n[STATUS] Creating and saving 'Transformers'...")
    print("         (These files 'remember' the vocabulary and categories from the training data")
    print("          and are required to make predictions on new data.)")
    
    # A) Text Features (TF-IDF)
    print("  > Building TF-IDF vectorizer for titles...")
    tfidf = TfidfVectorizer(max_features=2000, min_df=3, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df['title'].astype(str))
    joblib.dump(tfidf, FINAL_VECTORIZER_FILE)
    print(f"    - [SAVED] {FINAL_VECTORIZER_FILE}")

    # B) Categorical Features (One-Hot Encoder)
    print("  > Building One-Hot Encoder for categories...")
    cat_cols = ['guest_type', 'topic_category', 'format_type']
    df[cat_cols] = df[cat_cols].fillna('Other').astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_ohe = ohe.fit_transform(df[cat_cols])
    joblib.dump(ohe, FINAL_OHE_FILE)
    print(f"    - [SAVED] {FINAL_OHE_FILE}")

    # C) Numerical Features
    num_cols = ['duration', 'Title_Length', 'Description_Length', 'upload_hour', 'upload_dow', 'Days_Since_Published']
    X_numeric = df[num_cols].fillna(0).values

    # --- 4. Assemble Final Training Data ---
    print("\n[STATUS] Assembling final data for tuning...")
    X_numeric_sparse = sparse.csr_matrix(X_numeric)
    X_ohe_sparse = sparse.csr_matrix(X_ohe)
    
    X_full = sparse.hstack([X_tfidf, X_numeric_sparse, X_ohe_sparse]).tocsr()
    print(f"[OK]     Final training data is ready (Shape: {X_full.shape})")

    # --- 5. Hyperparameter Tuning ---
    print("\n[STATUS] Starting Hyperparameter Tuning (RandomizedSearchCV)...")
    print("         This will intelligently test 50 different model configurations")
    print("         to find the optimal settings. This is the longest step and")
    print("         may take several minutes. Please be patient.")

    # Define the "knobs" (parameters) to tune for Random Forest
    # We are providing a range of values for the model to try
    param_grid = {
        'n_estimators': [100, 200, 300, 500, 700], # Number of trees
        'max_depth': [10, 20, 30, None],          # How deep trees can grow
        'min_samples_leaf': [2, 4, 6],            # Min samples in a leaf node
        'min_samples_split': [2, 5, 10],          # Min samples to split a node
        'max_features': ['sqrt', 'log2', 1.0]     # Features to consider at each split
    }

    # Set up the base model
    rf = RandomForestRegressor(random_state=42)

    # Set up the randomized search. 
    # n_iter: How many combinations to try
    # cv: Cross-validation (splits data 3 times to test robustness)
    # n_jobs=-1: Use all available CPU cores
    rf_random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,  # Try 50 different combinations
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        verbose=1,  # Show progress (set to 0 for less output)
        random_state=42,
        n_jobs=-1,
        scoring='r2' # Our goal is to maximize R-squared
    )

    # Run the search!
    print("\n         Fitting 3 folds for each of 50 candidates, totalling 150 fits...")
    start_time = time()
    rf_random_search.fit(X_full, y_full)
    end_time = time()
    
    print(f"\n[OK]     Tuning complete in {end_time - start_time:.2f} seconds.")
    print("\n--- 🏆 Best Model Settings Found 🏆 ---")
    print(f"  > Best R-squared score found during tuning: {rf_random_search.best_score_:.4f}")
    print("  > With these settings:")
    # Print the best parameters found
    print(f"  {rf_random_search.best_params_}")
    
    # --- 6. Save the FINAL Tuned Model ---
    print("\n[STATUS] Saving the best performing model to disk...")

    # The 'best_estimator_' is the final model, already trained on
    # all data using the best settings found during the search.
    best_model = rf_random_search.best_estimator_

    # Save the trained model to a file
    joblib.dump(best_model, FINAL_MODEL_FILE)
    print(f"  > [SAVED] Final tuned model saved to: {FINAL_MODEL_FILE}")

    print("\n" + "="*70)
    print(" ✅  SUCCESS! ACTIVITY 3 (Part A) IS COMPLETE  ✅")
    print("="*70)
    print("The system is now fully trained, tuned, and ready for predictions.")
    print("The following files have been created:")
    print(f"  1. {FINAL_MODEL_FILE} (The new, tuned model)")
    print(f"  2. {FINAL_VECTORIZER_FILE} (The title text processor)")
    print(f"  3. {FINAL_OHE_FILE} (The category processor)")
    print("\nRun \033[1m'4_predict_views.py'\033[0m to make predictions.")
    print("="*70)

if __name__ == '__main__':
    tune_and_save_model()