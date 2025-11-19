import os
import pandas as pd
from scripts.create_folds import create_folds
from scripts.train_fold import train_fold
from scripts.evaluate_fold import evaluate_fold

def main():
    print("============================================================")
    print("5-FOLD MODEL TRAINING PIPELINE")
    print("============================================================")

    # Step 1: Create folds
    create_folds()

    all_metrics = []

    # Step 2: Train + evaluate each fold
    for fold in range(1, 6):
        train_fold(fold)
        metrics = evaluate_fold(fold)
        all_metrics.append(metrics)

    # Step 3: Save summary
    df = pd.DataFrame(all_metrics)
    df.to_csv("fold_summary.csv", index=False)

    print("------------------------------------------------------------")
    print("All folds completed.")
    print("Summary saved to fold_summary.csv")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
