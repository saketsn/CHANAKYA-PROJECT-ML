import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scripts.utils import load_fold_data, build_feature_matrices, save_json

def evaluate_fold(fold_number):
    print("------------------------------------------------------------")
    print(f"Evaluating model performance for Fold {fold_number}...")
    print("------------------------------------------------------------")

    # Correct path
    fold_path = f"folds/fold_{fold_number}"

    test_df = load_fold_data(f"{fold_path}/fold_{fold_number}_test.csv")

    # Load model + transformers
    model = joblib.load(f"{fold_path}/fold_{fold_number}_model.joblib")
    tfidf = joblib.load(f"{fold_path}/fold_{fold_number}_tfidf.joblib")
    ohe = joblib.load(f"{fold_path}/fold_{fold_number}_ohe.joblib")

    # Build test features
    X_test, _, _ = build_feature_matrices(test_df, tfidf=tfidf, ohe=ohe, fit=False)
    y_test = test_df["View_Count_Log"]

    y_pred = model.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5   # fixed for your sklearn version
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "fold": fold_number,
        "r2_score": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4)
    }

    save_json(metrics, f"{fold_path}/fold_{fold_number}_metrics.json")

    print(f"Fold {fold_number} Evaluation Results:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print("Metrics saved successfully.")
    print("------------------------------------------------------------")

    return metrics
