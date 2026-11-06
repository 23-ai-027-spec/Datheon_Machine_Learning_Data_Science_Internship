# Datheon Machine Learning & Data Science Internship

This repository contains the projects I completed during the Datheon Machine Learning & Data Science internship. Each task is a Jupyter notebook with code, plots and a written conclusion.

## Repository Structure

Datheon_Machine_Learning_Data_Science_Internship/
├── README.md
├── requirements.txt
├── Task1_Iris_Classification/
│   └── iris_classification_pipeline.ipynb
├── Task2_California_Housing/
│   └── california_housing_regression.ipynb
└── Task4_NLP_Sentiment/
    └── nlp_sentiment_classification.ipynb

## Task 1: Multi-Class Classification Pipeline (Iris)

- **Dataset:** Iris (built into scikit-learn)
- **Steps:** exploratory pairplots, StandardScaler, 80/20 stratified split, Logistic Regression and Decision Tree
- **Metrics:** accuracy, precision, recall, F1-score
- **Visuals:** confusion matrix heatmaps and decision boundary plots
- **Results (test set):**

| Model | Accuracy | F1-score |
|---|---|---|
| Logistic Regression | 0.9333 | 0.9333 |
| Decision Tree | 0.9667 | 0.9666 |

- **Notebook:** `Task1_Iris_Classification/iris_classification_pipeline.ipynb`

## Task 2: California Housing Price Prediction (Ridge & Lasso)

- **Dataset:** California Housing (scikit-learn)
- **Steps:** missing value imputation, correlation heatmap, feature engineering (interaction and ratio features, one-hot encoding), multicollinearity check with VIF, standardization
- **Models:** OLS, Ridge (L2) and Lasso (L1), with alpha tuned using GridSearchCV
- **Metrics:** RMSE, MAE, R2
- **Visuals:** predicted vs actual plots and residual distribution plots
- **Results (test set):**

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| OLS | (fill in) | (fill in) | (fill in) |
| Ridge | (fill in) | (fill in) | (fill in) |
| Lasso | (fill in) | (fill in) | (fill in) |

- **Notebook:** `Task2_California_Housing/california_housing_regression.ipynb`

## Task 4: NLP Sentiment Classification on Movie Reviews (TF-IDF)

- **Dataset:** NLTK movie_reviews (2,000 labeled reviews)
- **Preprocessing:** HTML and punctuation removal, lowercasing, stopword removal (negations kept), lemmatization
- **Features:** TfidfVectorizer with max_features=5000
- **Models:** Multinomial Naive Bayes and Logistic Regression
- **Extra:** `predict_sentiment()` function to test custom review text
- **Results (test set):**

| Model | Accuracy | F1-score |
|---|---|---|
| Multinomial Naive Bayes | 0.8075 | 0.7913 |
| Logistic Regression | 0.8550 | 0.8543 |

- **Notebook:** `Task4_NLP_Sentiment/nlp_sentiment_classification.ipynb`

## How to Run

Option 1: Google Colab
1. Open colab.research.google.com
2. File > Upload notebook, and choose a notebook from this repository
3. Runtime > Run all

Option 2: Locally
pip install -r requirements.txt
jupyter notebook

## Tools Used

Python, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, NLTK