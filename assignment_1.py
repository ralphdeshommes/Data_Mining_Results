import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the telescope data
df = pd.read_csv('telescope_data.csv')

# 1. DATA EXPLORATION
print("=" * 60)
print("DATA EXPLORATION")
print("=" * 60)
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nClass Distribution:")
print(df['class'].value_counts())
print("\nMissing Values:")
print(df.isnull().sum())

# 2. DATA VISUALIZATION
print("\n" + "=" * 60)
print("DATA VISUALIZATION")
print("=" * 60)

# Class distribution plot
plt.figure(figsize=(8, 6))
df['class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
print("Saved: class_distribution.png")

# Correlation heatmap
plt.figure(figsize=(12, 10))
# Convert class to numeric for correlation
df_numeric = df.copy()
df_numeric['class'] = df_numeric['class'].map({'g': 0, 'h': 1})
correlation = df_numeric.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved: correlation_heatmap.png")

# Feature distributions
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.ravel()
features = df.columns[:-1]
for idx, feature in enumerate(features):
    if idx < len(axes):
        df.boxplot(column=feature, by='class', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Class')
        axes[idx].set_xlabel('Class')
plt.tight_layout()
plt.savefig('feature_boxplots.png')
print("Saved: feature_boxplots.png")

# 3. DATA PREPROCESSING
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training class distribution: {np.unique(y_train, return_counts=True)}")
print(f"Testing class distribution: {np.unique(y_test, return_counts=True)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed")

# 4. MODEL TRAINING AND EVALUATION
print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)

# Dictionary to store results
results = {}

# K-Nearest Neighbors
print("\n--- K-Nearest Neighbors ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
results['KNN'] = acc_knn
print(f"Accuracy: {acc_knn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# Decision Tree
print("\n--- Decision Tree ---")
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
results['Decision Tree'] = acc_dt
print(f"Accuracy: {acc_dt:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# Random Forest
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
results['Random Forest'] = acc_rf
print(f"Accuracy: {acc_rf:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': df.columns[:-1],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Naive Bayes
print("\n--- Naive Bayes ---")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
acc_nb = accuracy_score(y_test, y_pred_nb)
results['Naive Bayes'] = acc_nb
print(f"Accuracy: {acc_nb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# Support Vector Machine
print("\n--- Support Vector Machine ---")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
results['SVM'] = acc_svm
print(f"Accuracy: {acc_svm:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# 5. MODEL COMPARISON
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False)
print("\nModel Accuracies:")
print(results_df.to_string(index=False))

# Plot model comparison
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Accuracy'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45, ha='right')
plt.ylim([0.7, 1.0])
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nSaved: model_comparison.png")

# 6. SUMMARY
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
best_model = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']
print(f"\nBest Model: {best_model}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("\nAll visualizations have been saved to the current directory.")
print("Analysis complete!")
