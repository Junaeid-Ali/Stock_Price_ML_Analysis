# Stock_Price_ML_Analysis
# Stock Price ML Analysis

A machine learning internship project exploring a stock price dataset through three progressively advanced levels — from data preprocessing to classification and clustering. Each level builds on engineered features (date decomposition, label encoding, standard scaling) to predict and analyze stock price behavior.

## Project Structure

**Level 1 — Data Preprocessing & Regression**
- Missing value imputation, categorical encoding, and date feature extraction (year/month/day)
- Train/test split followed by StandardScaler (fit on train only, to avoid data leakage)
- Linear Regression to predict `close` price, evaluated with R², MSE, and RMSE

**Level 2 — Classification & Clustering**
- Binary `price_class` target (High/Low) derived from the median close price
- Decision Tree Classifier (full and pruned versions), visualized with `plot_tree`
- K-Means Clustering with the Elbow Method to determine optimal cluster count, plus cluster-level summary statistics

**Level 3 — Ensemble Models & SVM**
- Random Forest Classifier with cross-validation and feature importance ranking
- Support Vector Machine (Linear and RBF kernels) trained on a stratified sample for tractable runtime, evaluated on Accuracy, Precision, Recall, F1, and AUC
- 2D decision boundary visualization comparing predicted regions against actual test points

## Key Techniques
Data leakage prevention, stratified sampling, label encoding, feature scaling, classification metrics, ensemble methods, kernel-based classification, and model interpretability via feature importance and decision boundaries.

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Dataset
Stock price data (open, high, low, close, volume) sourced from Kaggle.
