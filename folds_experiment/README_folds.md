# YouTube View & Revenue Prediction –  Project

A complete machine learning pipeline built to predict the expected view count and estimated revenue of future (unreleased) YouTube videos. The project uses metadata gathered from the channel “The Chanakya Dialogues Hindi” and applies feature engineering, model training, tuning, and a 5-fold evaluation framework to produce a reliable prediction system.

---

## 1. Project Overview

This project answers a practical question:

**“Can we estimate how well a video will perform before uploading it?”**

To do this, we predict:

- Long-term view count  
- Estimated ad revenue  
- Topic/guest/content performance trends  

Using machine learning, the system converts intuition-based content planning into a measurable, data-driven decision-making process.

---

## 2. Goals of the Project

### Machine Learning Goals
- Build a regression model that predicts  
  **View_Count_Log = log(1 + views)**
- Engineer features from:
  - Text (title)
  - Categories (topic, guest type, format)
  - Numerical attributes (duration, upload time)
- Evaluate using R², RMSE, MAE
- Build a 5-fold block experiment to test stability

### Business Goals
- Estimate future performance of video ideas
- Predict ad revenue before publishing
- Help creators choose topics, guests, and formats strategically

---

## 3. Dataset Construction

### 3.1 Scraping
Video metadata was collected using `yt-dlp` and saved as:


#### Key features created:

**Target variable**

**Text features**
- TF-IDF on title  
- max_features = 2000  
- ngram_range = (1, 2)  
- min_df = 3  

**Categorical One-Hot Encoding**
- guest_type  
- topic_category  
- format_type  

**Numeric features**
- duration  
- Title_Length  
- Description_Length  
- upload_hour  
- upload_dow  
- Days_Since_Published  

Output stored as:


### How to run 
 source venv/Scripts/activate

"""
scrape_chanakya.py
    - Runs the scraper and creates: raw_chanakya_filtered_by_age.csv

preprocess.py
    - Takes the raw CSV and creates:
        • cleaned_chanakya.csv
        • title_tfidf.npz
        • ohe_cats.joblib

1_compare_models.py
    - Uses the cleaned data and transformers
    - Compares multiple ML models and prints the comparison report

2_build_final_model.py

3_tune_and_save_model.py
    - Uses raw_chanakya_filtered_by_age.csv
    - Tunes the best model and saves the final production files:
        • final_tuned_model.joblib
        • final_tfidf_vectorizer.joblib
        • final_ohe_cats.joblib

4_predict_views.py
    - Loads the final production model + transformers
    - Generates and prints final prediction results
"""

"""
5 fold run 
1-> python fold_runner.py
2-> python predict_with_all_folds.py

"""


