import pandas as pd
import os

FOLDS = 5
FOLD_SIZE = 100

def create_folds():
    print("------------------------------------------------------------")
    print("Creating dataset folds (5 folds × 100 videos each)...")
    print("------------------------------------------------------------")

    # Load the cleaned dataset from the parent folder
    df = pd.read_csv("../cleaned_chanakya.csv")

    # Use only the first 500 rows (as required: 100 × 5)
    df = df.head(FOLDS * FOLD_SIZE)

    # Folds should be saved INSIDE folds_experiment/folds/
    base_path = "folds"

    os.makedirs(base_path, exist_ok=True)  # Ensure parent folder exists

    for i in range(FOLDS):
        start = i * FOLD_SIZE
        end = start + FOLD_SIZE

        fold_df = df.iloc[start:end]

        # Example path: folds/fold_1
        fold_path = f"{base_path}/fold_{i+1}"

        os.makedirs(fold_path, exist_ok=True)

        # Save fold data
        fold_df.to_csv(f"{fold_path}/fold_{i+1}_data.csv", index=False)

        print(f"Fold {i+1} created with rows {start} to {end - 1}")

    print("All folds created successfully.")
    print("------------------------------------------------------------")
