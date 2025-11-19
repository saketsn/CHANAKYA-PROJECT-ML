import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import json
import os


def load_fold_data(csv_path):
    """
    Load a fold dataset from CSV.
    """
    return pd.read_csv(csv_path)


def save_json(data, path):
    """
    Save a Python dictionary to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def build_feature_matrices(df, tfidf=None, ohe=None, fit=False):
    """
    Create the full feature matrix for model training or prediction.

    Parameters:
    - df: DataFrame containing the fold's data.
    - tfidf: TF-IDF vectorizer (None if training).
    - ohe: One-Hot Encoder (None if training).
    - fit: If True, fit transformers; if False, use transform only.

    Returns:
    - final_matrix: Sparse matrix containing all features.
    - tfidf: Fitted TF-IDF vectorizer.
    - ohe: Fitted One-Hot encoder.
    """

    df = df.copy()

    # ----------------------------------------------------------
    # Text Fields
    # ----------------------------------------------------------

    # Title always exists
    df["title"] = df["title"].fillna("")

    # Your cleaned dataset does NOT contain raw 'description'
    # So we safely create a blank column to maintain pipeline shape.
    if "description" not in df.columns:
        df["description"] = ""

    df["description"] = df["description"].fillna("")

    # ----------------------------------------------------------
    # Categorical Fields
    # ----------------------------------------------------------
    cat_cols = ["guest_type", "topic_category", "format_type"]

    # Ensure these exist (in case of fold slicing)
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "Other"

    df[cat_cols] = df[cat_cols].fillna("Other").astype(str)

    # ----------------------------------------------------------
    # Numerical Fields
    # ----------------------------------------------------------
    num_cols = [
        "duration", 
        "Title_Length", 
        "Description_Length",
        "upload_hour", 
        "upload_dow", 
        "Days_Since_Published"
    ]

    # Make sure all numeric columns exist
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0

    numeric_matrix = sparse.csr_matrix(df[num_cols].fillna(0).values)

    # ----------------------------------------------------------
    # TF-IDF Vectorization (Title)
    # ----------------------------------------------------------
    if fit:
        tfidf = TfidfVectorizer(
            max_features=2000,
            min_df=3,
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf.fit_transform(df["title"])
    else:
        tfidf_matrix = tfidf.transform(df["title"])

    # ----------------------------------------------------------
    # One-Hot Encoding (Categories)
    # ----------------------------------------------------------
    if fit:
        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )
        ohe_matrix = ohe.fit_transform(df[cat_cols])
    else:
        ohe_matrix = ohe.transform(df[cat_cols])

    ohe_matrix = sparse.csr_matrix(ohe_matrix)

    # ----------------------------------------------------------
    # Final Combined Feature Matrix
    # ----------------------------------------------------------
    final_matrix = sparse.hstack([
        tfidf_matrix,
        numeric_matrix,
        ohe_matrix
    ]).tocsr()

    return final_matrix, tfidf, ohe
