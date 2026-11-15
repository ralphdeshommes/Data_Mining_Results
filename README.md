# MAGIC Gamma Telescope Classification

A scikit-learn pipeline that classifies telescope events as **gamma rays** (signal) or **cosmic-ray particles** (hadron background) from 10 measured shape features, and compares five classifiers.

## What it does
1. Loads `telescope_data.csv` and explores it (class balance, correlation heatmap, per-feature box plots).
2. Splits the data 70/30 with a stratified split (`random_state=42`) and scales features with `StandardScaler`.
3. Trains five models: KNN, Decision Tree, Random Forest, Gaussian Naive Bayes, and an RBF SVM.
4. Reports accuracy, precision, recall, F1, and a confusion matrix for each, plus Random Forest feature importances and a comparison chart.

## Results (61 held-out test events)
| Model | Accuracy |
|---|---|
| Random Forest | 75.4% |
| Decision Tree | 73.8% |
| SVM (RBF) | 72.1% |
| KNN | 65.6% |
| Naive Bayes | 60.7% |

The dataset is balanced (101 gamma, 100 hadron), so chance is about 50%.

## Limitations
- `telescope_data.csv` is a **201-event balanced subsample** of the 19,020-event public dataset. With only 61 test events, each event is about 1.6 points of accuracy, so the ranking between close models is not reliable.
- The data is Monte Carlo simulated, not real observations.
- No hyperparameter tuning or cross-validation.

## Run it
```
pip install pandas numpy matplotlib seaborn scikit-learn
python magic_classifier.py
```
Plots are written to the current folder.

## Data
MAGIC Gamma Telescope dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
