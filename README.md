ML Project Phase 4 - Food Price Crisis Prediction

Name: Jawad Ali
Roll Number: BCSF23M541
Dataset: Global Food Price Inflation
Kaggle: https://www.kaggle.com/code/bcsf23m541jawadali/csf23m-ml-project-phase4-food-price-crisis-predict
GitHub: https://github.com/jawadali321-oss/CSF23M_ML-Project-Phase4-Food-Price-Crisis-Prediction

Overview

Phase 4 is about methodology implementation. The goal was to train the models planned in Phase 1, handle class imbalance, evaluate results properly, and validate the model in ways that reflect real deployment.

Dataset

Rows: 4434
Target column: crisis_next_3m (binary - will there be a food crisis in the next 3 months)
Input: Engineered CSV from Phase 3 with 29 features
Region coverage: 100% using official region labels from Phase 2

Model Results

Model                  F1-Score   Precision   Recall   ROC-AUC
Logistic Regression    0.8211     0.7091      0.9750   0.9944
Random Forest          0.8606     0.8353      0.8875   0.9947
XGBoost                0.8780     0.8571      0.9000   0.9954

Best Model: XGBoost with F1 0.8780 and AUC 0.9954
Primary metric used throughout: F1-Score

Time-Based Validation

Trained on data up to 2020, tested on 2021 onwards. Simulates real deployment conditions.

Train period: up to 2020 - 3584 rows
Test period: after 2020 - 850 rows
Time-based F1: 0.8981
Time-based AUC: 0.9911

Leave-One-Country-Out Validation

Trained on all countries except one, tested on that country alone. Proves model works in conflict zones it has never seen.

Yemen              N=166   Crisis Rate=0.036   F1=0.6667   Recall=1.0000
Syrian Arab Rep    N=142   Crisis Rate=0.331   F1=0.9565   Recall=0.9362
Somalia            N=190   Crisis Rate=0.058   F1=1.0000   Recall=1.0000
Afghanistan        N=190   Crisis Rate=0.053   F1=0.9524   Recall=1.0000
Nigeria            N=190   Crisis Rate=0.142   F1=0.8136   Recall=0.8889

Average F1: 0.8778
Average Recall: 0.9650

Region-wise SHAP Analysis

100% data coverage using official region labels. inflation_volatility dominated all regions with 10 out of 10 features shared across all region pairs. A single global model can serve WFP across all conflict zones without region-specific retraining.

Minimal Viable Feature Set

15 features out of 29 maintain 98% of full model F1. Directly reduces data collection burden for WFP field workers in active conflict zones.

FCAI Validation

Scenario A - FCAI only no raw components: F1 = 0.8554
Scenario B - Raw components only no FCAI: F1 = 0.8795
Scenario C - Full model FCAI and everything: F1 = 0.8727

FCAI efficiency: 97.3% of raw-feature performance replacing 7 features with one composite score.

Novelty Contributions

1. First crisis classifier on this dataset - only EDA existed on Kaggle before this work
2. Region-wise SHAP with 100% data coverage - universal predictors confirmed across all conflict zones
3. Minimal viable feature set - 15 features identified for WFP field deployment

Files

Phase4_Jawad_Ali_BCSF23M541.py       Main Phase 4 script
Phase4_Jawad_Ali_BCSF23M541.ipynb    Kaggle notebook with full outputs
