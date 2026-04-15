# Wine Quality Regression Pipeline

This project builds a full machine learning pipeline on the UCI Wine Quality white-wine dataset. The goal is to predict the numeric `quality` score from physicochemical wine measurements, segment wines with clustering, and test whether adding cluster information helps ensemble models.

## Dataset
- **Name:** Wine Quality (white wine subset)
- **Source URL:** https://archive.ics.uci.edu/dataset/186/wine+quality
- **Raw file used:** `data/raw/winequality-white.csv`
- **DOI:** 10.24432/C56S3T
- **License:** CC BY 4.0

## Install and run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Run notebooks in order: T1 -> T2 -> T3 -> T4.

## Final model results
| Model_Type      | Model             | CV_RMSE   | CV_MAE   | CV_R2   |   Test_RMSE |   Test_MAE |   Test_R2 |
|:----------------|:------------------|:----------|:---------|:--------|------------:|-----------:|----------:|
| Task 2 Baseline | KNN Regressor     |           |          |         |      0.7604 |     0.5793 |    0.295  |
| Ensemble        | Random Forest     | 0.7041    | 0.5462   | 0.3662  |      0.7414 |     0.5676 |    0.3298 |
| Ensemble        | Gradient Boosting | 0.7082    | 0.5516   | 0.3590  |      0.7381 |     0.5771 |    0.3357 |

## Example figure
![Correlation heatmap](reports/eda_correlation_heatmap.png)

## Report
This project is a complete machine learning pipeline for predicting white wine quality. In Task 1, I explored the data and found that alcohol and density are the most important features. In Task 2, I tested basic models and found that KNN is better than Linear Regression because the data is not strictly linear. In Task 3, I used K-Means to group the wines into clusters based on their properties (like sweet vs. dry). Finally, in Task 4, I used these cluster labels as a new feature for Random Forest and Gradient Boosting. Gradient Boosting showed the best result (lowest Test RMSE of ~0.738). This proves that combining unsupervised clustering with ensemble models is a very effective way to predict wine quality.
