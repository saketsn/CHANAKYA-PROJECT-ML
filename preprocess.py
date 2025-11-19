"""
preprocess.py
Takes raw CSV produced by scrape_chanakya.py and creates cleaned_chanakya.csv
This version now INCLUDES an estimated income feature.
Also saves TF-IDF vectorizer and the transformed TF-IDF arrays if desired.
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib
from scipy import sparse

# --- Configuration ---
RAW_FILTERED = 'raw_chanakya_filtered_by_age.csv'
OUTPUT_CLEAN = 'cleaned_chanakya.csv'
TFIDF_VECT = 'title_tfidf_vectorizer.joblib'
TFIDF_MATRIX = 'title_tfidf.npz'
OHE_VECT = 'ohe_cats.joblib'
MIN_TFIDF_DF = 3
MAX_FEATURES = 2000
ASSUMED_RPM_INR = 120 # Assumed Revenue Per 1,000 views in INR

# --- Functions ---

def load_raw():
    assert os.path.exists(RAW_FILTERED), f"{RAW_FILTERED} not found. Run scraping first."
    df = pd.read_csv(RAW_FILTERED)
    return df

def basic_cleaning(df):
    # Normalize empty text
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    # convert types
    df['view_count'] = df['view_count'].fillna(0).astype(int)
    df['like_count'] = df['like_count'].fillna(0).astype(int)
    df['duration'] = df['duration'].fillna(0).astype(int)
    # parse upload_datetime back to datetime if present
    if 'upload_datetime' in df.columns:
        df['upload_datetime'] = pd.to_datetime(df['upload_datetime'], errors='coerce')
    return df

def feature_engineering(df):
    now = datetime.now()
    # Days since published
    df['Days_Since_Published'] = df['upload_datetime'].apply(lambda x: (now - x).days if pd.notnull(x) else np.nan)

    # Title and description lengths
    df['Title_Length'] = df['title'].str.len()
    df['Description_Length'] = df['description'].str.len()

    # Cyclic features for upload hour/day
    df['upload_hour'] = df['upload_datetime'].dt.hour.fillna(0).astype(int)
    df['upload_dow'] = df['upload_datetime'].dt.dayofweek.fillna(0).astype(int)

    # Target (log) for views
    df['View_Count_Log'] = np.log1p(df['view_count'].astype(float))

    # --- NEW: ESTIMATED INCOME CALCULATION ---
    # Calculate estimated income based on the assumed RPM
    df['Estimated_Income_INR'] = (df['view_count'].fillna(0) / 1000) * ASSUMED_RPM_INR
    
    return df

def text_features_tfidf(df, save_vectorizer=True):
    titles = df['title'].astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_TFIDF_DF, ngram_range=(1,2))
    X_title = tfidf.fit_transform(titles)
    if save_vectorizer:
        joblib.dump(tfidf, TFIDF_VECT)
        sparse.save_npz(TFIDF_MATRIX, X_title)
        print(f"[preprocess] Saved TF-IDF vectorizer to {TFIDF_VECT} and matrix to {TFIDF_MATRIX}")
    return X_title, tfidf

def categorical_encode(df, save_ohe=True):
    cat_cols = ['guest_type', 'topic_category', 'format_type']
    df[cat_cols] = df[cat_cols].fillna('Other').astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_data = ohe.fit_transform(df[cat_cols])
    if save_ohe:
        joblib.dump(ohe, OHE_VECT)
        print(f"[preprocess] Saved one-hot encoder to {OHE_VECT}")
    cat_df = pd.DataFrame(cat_data, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    return cat_df, ohe

def assemble_and_save(df, tfidf_matrix):
    # Numerical columns safe for a pre-publish model
    num_cols = ['duration', 'Title_Length', 'Description_Length', 'upload_hour', 'upload_dow', 'Days_Since_Published']
    df_num = df[num_cols].fillna(0).copy()
    
    # Categorical one-hot encoded columns
    cat_df, _ = categorical_encode(df, save_ohe=True)

    # --- UPDATED: ADDED 'Estimated_Income_INR' TO THE LIST OF COLUMNS TO SAVE ---
    # We keep the original and engineered features for analysis, plus our two targets.
    cols_to_keep = [
        'video_id', 'video_url', 'title', 'upload_datetime', 
        'view_count', 'like_count', 'comment_count', 'tags', 
        'guest_type', 'topic_category', 'format_type', 
        'View_Count_Log', 'Estimated_Income_INR' # <-- ADDED HERE
    ]
    
    df_final = pd.concat([
        df[cols_to_keep].reset_index(drop=True),
        df_num.reset_index(drop=True),
        cat_df.reset_index(drop=True)
    ], axis=1)
    
    df_final.to_csv(OUTPUT_CLEAN, index=False)
    print(f"[preprocess] Saved cleaned dataset with estimated income to {OUTPUT_CLEAN}")
    return df_final

def main():
    df = load_raw()
    df = basic_cleaning(df)
    df = feature_engineering(df)
    X_title, tfidf = text_features_tfidf(df, save_vectorizer=True)
    df_final = assemble_and_save(df, X_title)
    print("\n[preprocess] Done. Your cleaned dataset is now ready for modeling both views and income.")

if __name__ == '__main__':
    main()