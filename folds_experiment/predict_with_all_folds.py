import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Business Parameter (consistent with Activity 1)
CPM = 120   # Revenue per 1000 views in INR


# -------------------------------------------------------------
# Helper: Check that fold models exist
# -------------------------------------------------------------
def check_fold_files():
    missing = []

    for fold in range(1, 6):
        path = f"folds/fold_{fold}"
        required = [
            f"{path}/fold_{fold}_model.joblib",
            f"{path}/fold_{fold}_tfidf.joblib",
            f"{path}/fold_{fold}_ohe.joblib"
        ]

        for file in required:
            if not os.path.exists(file):
                missing.append(file)

    if missing:
        print("The following required files are missing:\n")
        for f in missing:
            print(" -", f)
        print("\nPlease run fold_runner.py first.")
        return False

    return True


# -------------------------------------------------------------
# Helper: Build features for a new video (same as training)
# -------------------------------------------------------------
def build_features(video_data, tfidf, ohe):
    df = pd.DataFrame([video_data])

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    # Time-related engineered features
    df["upload_datetime"] = datetime.now()
    df["Days_Since_Published"] = 0
    df["upload_hour"] = df["upload_datetime"].dt.hour
    df["upload_dow"] = df["upload_datetime"].dt.dayofweek

    # Length-based engineered features
    df["Title_Length"] = df["title"].str.len()
    df["Description_Length"] = df["description"].str.len()

    # Categorical
    cat_cols = ["guest_type", "topic_category", "format_type"]
    df[cat_cols] = df[cat_cols].fillna("Other").astype(str)

    # Numerical
    num_cols = [
        "duration",
        "Title_Length",
        "Description_Length",
        "upload_hour",
        "upload_dow",
        "Days_Since_Published"
    ]

    # Transformations
    X_tfidf = tfidf.transform(df["title"])
    X_ohe = ohe.transform(df[cat_cols])
    X_num = sparse.csr_matrix(df[num_cols].values)

    # Combine
    X_final = sparse.hstack([X_tfidf, X_num, sparse.csr_matrix(X_ohe)]).tocsr()

    return X_final


# -------------------------------------------------------------
# Main function: Predict using all 5 fold models
# -------------------------------------------------------------
def predict_with_all_folds(video_data):
    predictions = []

    print("------------------------------------------------------------")
    print("Running prediction using all 5 fold models")
    print("------------------------------------------------------------")

    for fold in range(1, 6):

        fold_path = f"folds/fold_{fold}"

        model = joblib.load(f"{fold_path}/fold_{fold}_model.joblib")
        tfidf = joblib.load(f"{fold_path}/fold_{fold}_tfidf.joblib")
        ohe = joblib.load(f"{fold_path}/fold_{fold}_ohe.joblib")

        X_new = build_features(video_data, tfidf, ohe)
        log_pred = model.predict(X_new)
        views_pred = int(np.expm1(log_pred)[0])

        predictions.append(views_pred)

        print(f"Fold {fold} predicted: {views_pred:,} views")

    print("------------------------------------------------------------")

    # Final averaged prediction
    avg_views = int(np.mean(predictions))
    estimated_revenue = round((avg_views / 1000) * CPM, 2)

    return avg_views, estimated_revenue, predictions


# -------------------------------------------------------------
# Program Entry
# -------------------------------------------------------------
if __name__ == "__main__":

    print("============================================================")
    print(" FINAL PREDICTION USING 5-FOLD ENSEMBLE MODEL")
    print("============================================================")

    if not check_fold_files():
        exit()

    # Example Video (you can replace this later)
    video_data = {
        "title": "Will India Become a Tech Superpower? | Future Growth Analysis",
        "description": "A complete analysis of India's technological rise and global competitiveness.",
        "duration": 1600,
        "guest_type": "Academic",
        "topic_category": "Internal_Affairs",
        "format_type": "Interview"
    }

    print("\nInput Video Details:")
    print(f"Title: {video_data['title']}")
    print(f"Duration: {video_data['duration']} seconds")
    print(f"Guest: {video_data['guest_type']}")
    print(f"Topic: {video_data['topic_category']}")
    print(f"Format: {video_data['format_type']}")
    print("------------------------------------------------------------")

    avg_views, revenue, fold_outputs = predict_with_all_folds(video_data)

    print("\nFinal Aggregated Prediction:")
    print("------------------------------------------------------------")
    print(f"Predictions from 5 folds: {fold_outputs}")
    print(f"Average Estimated Views: {avg_views:,}")
    print(f"Estimated Revenue (@ CPM ₹{CPM}): ₹{revenue:,}")
    print("------------------------------------------------------------")

    print("\nPrediction process completed successfully.")
    print("============================================================")
