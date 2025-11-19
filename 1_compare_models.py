import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from time import time
import warnings
from tabulate import tabulate  # Used to create clean tables in the terminal

# --- Model Imports ---
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# --- Evaluation Imports ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
CLEANED_DATA_CSV = 'cleaned_chanakya.csv'
TFIDF_MATRIX_FILE = 'title_tfidf.npz'
OHE_FILE = 'ohe_cats.joblib'

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def run_model_comparison():
    """
    Loads preprocessed data from Activity 1, trains 3 models (Ridge,
    Random Forest, LightGBM), and prints a detailed comparison table
    and justification for selecting the best model.
    """
    
    print("="*70)
    print("      ACTIVITY 2: YOUTUBE VIEW PREDICTOR - MODEL SHOWDOWN         ")
    print("="*70)
    
    # --- 1. Load Preprocessed Data ---
    print("\n[Step 1: Load Data]")
    print(f"Loading data from '{CLEANED_DATA_CSV}' and feature files...")
    
    try:
        # We load the files created in Activity 1
        # 'cleaned_chanakya.csv' contains our numerical and categorical data
        # 'title_tfidf.npz' contains the text features for our titles
        # 'ohe_cats.joblib' contains the 'encoder' for our categories
        df = pd.read_csv(CLEANED_DATA_CSV)
        X_tfidf = sparse.load_npz(TFIDF_MATRIX_FILE)
        ohe = joblib.load(OHE_FILE)
    except FileNotFoundError as e:
        print(f"\n--- 🔴 ERROR: FILE NOT FOUND 🔴 ---")
        print(f"         Fatal Error: Required file not found: {e.filename}")
        print("         Please ensure 'preprocess.py' from Activity 1 ran successfully.")
        print("         Exiting process.")
        print("---------------------------------------\n")
        return

    print(f"\n[OK] Successfully loaded {len(df)} videos.")

    # --- 2. Assemble Features (X) and Target (y) ---
    print("\n[Step 2: Assembling Features and Target]")
    print("Combining all our data into a final dataset for the model.")

    # The target variable (y) is what we want to predict:
    TARGET = 'View_Count_Log'
    y = df[TARGET]
    print(f"  > Target (y) set to: '{TARGET}'")
    print("     (This is the log-transformed view count. We predict this to make it easier for")
    print("      the model to handle the wide range of view counts, e.g., 10k vs 10M views).")


    print("\n  > Assembling Features (X): These are the 'clues' the model will use.")
    
    # Feature Set 1: Numerical Features
    NUMERICAL_FEATURES = [
        'duration', 'Title_Length', 'Description_Length', 
        'upload_hour', 'upload_dow', 'Days_Since_Published'
    ]
    X_numeric = df[NUMERICAL_FEATURES].fillna(0).values
    print(f"    - Using {len(NUMERICAL_FEATURES)} numerical features (duration, title length, time of day, etc.)")

    # Feature Set 2: Categorical Features
    OHE_FEATURES = ohe.get_feature_names_out()
    X_ohe = df[OHE_FEATURES].values
    print(f"    - Using {len(OHE_FEATURES)} categorical features (e.g., 'guest_type_Diplomat', 'topic_category_Geopolitics')")

    # Feature Set 3: Text Features
    print(f"    - Using {X_tfidf.shape[1]} text features (words from the video titles)")

    # Combine all features into one big matrix
    X_all = sparse.hstack([
        X_tfidf, 
        sparse.csr_matrix(X_numeric), 
        sparse.csr_matrix(X_ohe)
    ]).tocsr() # .tocsr() optimizes for matrix operations

    print(f"\n  [OK] Total features combined: {X_all.shape[1]}")

    # --- 3. Split Data for Training and Testing ---
    print("\n[Step 3: Splitting Data for a Fair Test]")
    print("  > Splitting data into a Training Set (80%) and a Testing Set (20%).")
    print("  > Why? The model *learns* patterns from the Training set.")
    print("  > It is then *graded* on the Testing set, which it has never seen before.")
    print("  > This proves the model actually learned and isn't just 'memorizing' the answers.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42 # random_state=42 ensures we get the same split every time
    )

    print(f"  > Training examples: {X_train.shape[0]}")
    print(f"  > Testing examples:  {X_test.shape[0]}")

    # --- 4. Train and Evaluate Models ---
    print("\n[Step 4: Running the Model 'Bake-Off']")
    print("Training and evaluating three different models to see which is best.")

    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5),
        'LightGBM': LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, learning_rate=0.1, verbose=-1)
    }

    results = []
    
    for name, model in models.items():
        print(f"\n  --- Training {name} ---")
        start_time = time()
        model.fit(X_train, y_train)
        train_time = time() - start_time
        
        # Make predictions on the hidden test set
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) # Corrected: compares test values to predicted values

        results.append({
            'Model': name,
            'R-squared': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Train Time (sec)': train_time
        })
        print(f"  ✅ Finished {name} in {train_time:.2f} seconds.")

    # --- 5. Compare Models (The "Showdown") ---
    print("\n" + "="*70)
    print("           MODEL COMPARISON RESULTS (Test Set)")
    print("="*70)
    
    # Convert results to a DataFrame for sorting and display
    results_df = pd.DataFrame(results).set_index('Model')
    
    # Sort the table by R-squared (highest is best)
    results_df = results_df.sort_values(by='R-squared', ascending=False)
    
    # Print the formatted table using tabulate
    headers = ["Model", "R-squared (Accuracy)", "RMSE (Error)", "MAE (Error)", "Train Time (sec)"]
    table_data = []
    for index, row in results_df.iterrows():
        table_data.append([
            index,
            row['R-squared'],
            row['RMSE'],
            row['MAE'],
            row['Train Time (sec)']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid', floatfmt=".4f"))
    
    print("\n--- How to Read This Table (Explanation of Metrics) ---")
    print("\n  \033[1mR-squared (Accuracy):\033[0m")
    print("  - This is the most important score. It's a percentage (0 to 1.0).")
    print("  - It measures how much of the video's success the model can successfully explain.")
    print("  - \033[1mHIGHER IS BETTER.\033[0m A score of 0.729 means the model explains 72.9% of the results.")

    print("\n  \033[1mRMSE & MAE (Error):\033[0m")
    print("  - These measure the average prediction error. They tell you 'how wrong' the model is.")
    print("  - \033[1mLOWER IS BETTER.\033[0m")

    # --- 6. Final Verdict and Justification ---
    best_model_name = results_df.index[0]
    best_model_stats = results_df.iloc[0]

    print("\n" + "="*70)
    print(" 🏆 FINAL VERDICT & MODEL SELECTION 🏆")
    print("="*70)
    print(f"\nThe best performing model is: \033[1m{best_model_name}\033[0m")
    
    print("\n--- Detailed Justification ---")
    print(f"1. \033[1mHighest Accuracy:\033[0m The {best_model_name} model has the highest R-squared score")
    print(f"   ({best_model_stats['R-squared']:.4f}). This means it successfully explained \033[1m{best_model_stats['R-squared']*100:.1f}% of the variance\033[0m")
    print("   in video views. This is the strongest indicator of a good model.")
    
    print(f"\n2. \033[1mLowest Prediction Error:\033[0m It also has the lowest error scores (RMSE: {best_model_stats['RMSE']:.4f}, MAE: {best_model_stats['MAE']:.4f}).")
    print("   This confirms its predictions are, on average, the most precise.")

    print(f"\n3. \033[1mWhy Not the Other Models?\033[0m")
    
    # Dynamic comparison against the other models
    for i in range(1, len(results_df)):
        other_model_name = results_df.index[i]
        other_model_stats = results_df.iloc[i]
        
        print(f"\n   - \033[1mCompared to {other_model_name}:\033[0m")
        if other_model_name == 'Ridge Regression':
            print(f"     The {other_model_name} was very fast, but its accuracy (R-squared: {other_model_stats['R-squared']:.4f})")
            print("     was the lowest. This proves that the factors for video success are complex")
            print("     and require a more advanced model than a simple linear one.")
        else:
            print(f"     The {other_model_name} was a strong competitor, but its R-squared ({other_model_stats['R-squared']:.4f}) and")
            print(f"     RMSE ({other_model_stats['RMSE']:.4f}) were slightly worse than the {best_model_name}.")
            print(f"     Therefore, {best_model_name} is the superior choice for this project.")

    
    print("\n--- END OF ACTIVITY 2 (PART A) ---")
    print(f"We have selected '{best_model_name}' as our champion model.")
    print("\nNow, run \033[1m'2_build_final_model.py'\033[0m to train this model on 100% of the data")
    print("and save it. This will prepare us for Activity 3 (Prediction).")
    print("="*70)

if __name__ == '__main__':
    run_model_comparison()

