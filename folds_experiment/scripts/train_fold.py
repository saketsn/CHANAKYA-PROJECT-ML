import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scripts.utils import load_fold_data, build_feature_matrices

def train_fold(fold_number):
    print("------------------------------------------------------------")
    print(f"Training model for Fold {fold_number}...")
    print("------------------------------------------------------------")

    # Correct path
    fold_path = f"folds/fold_{fold_number}"

    df = load_fold_data(f"{fold_path}/fold_{fold_number}_data.csv")

    # Train-test split inside the fold
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

    train_df.to_csv(f"{fold_path}/fold_{fold_number}_train.csv", index=False)
    test_df.to_csv(f"{fold_path}/fold_{fold_number}_test.csv", index=False)

    # Build feature matrices
    X_train, tfidf, ohe = build_feature_matrices(train_df, fit=True)
    y_train = train_df["View_Count_Log"]

    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=2,
        max_depth=30,
        max_features=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model + transformers
    joblib.dump(model, f"{fold_path}/fold_{fold_number}_model.joblib")
    joblib.dump(tfidf, f"{fold_path}/fold_{fold_number}_tfidf.joblib")
    joblib.dump(ohe, f"{fold_path}/fold_{fold_number}_ohe.joblib")

    print(f"Fold {fold_number} model training completed successfully.")
    print(f"Model and transformers saved in: {fold_path}")
    print("------------------------------------------------------------")

    return True
