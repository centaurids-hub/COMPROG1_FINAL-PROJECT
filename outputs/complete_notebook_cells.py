# CELL 1: Imports and Configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import time
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "filtered_dataset.csv"
OUTPUT_DIR = "outputs"
ANOXIC_THRESH = 5.8
FEATURES = ["TEMP", "PH", "AMMONIA(mg/l)", "NITRATE(PPM)", "TURBIDITY"]
TARGET = "anoxic"
RANDOM_STATE = 42
TEST_SIZE = 0.2

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CELL 2: Load Data
df = pd.read_csv(DATA_PATH)
df["anoxic"] = (df["DO"] < ANOXIC_THRESH).astype(int)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
y = df[TARGET].values
X_orig = df[FEATURES].values
print(f"Dataset: {len(df)} samples | Classes: Normal={sum(y==0)}, Anoxic={sum(y==1)}")

# CELL 3: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_orig)
poly_names = poly.get_feature_names_out(FEATURES)
print(f"Original: {X_orig.shape[1]} features | Polynomial: {X_poly.shape[1]} features")

# CELL 4: Scale and Split (Original)
scaler_orig = StandardScaler()
X_orig_scaled = scaler_orig.fit_transform(X_orig)
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_orig_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# CELL 5: Scale and Split (Polynomial)
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly)
X_train_poly, X_test_poly, _, _ = train_test_split(
    X_poly_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# CELL 6: MODEL 1 — Logistic Regression
print("="*60)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*60)
lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
lr_cv = cross_val_score(lr, X_orig_scaled, y, cv=cv, scoring='accuracy')
lr.fit(X_train_orig, y_train)
lr_pred = lr.predict(X_test_orig)
lr_prob = lr.predict_proba(X_test_orig)[:, 1]
print(f"CV: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"Test: {accuracy_score(y_test, lr_pred):.4f} | AUC: {roc_auc_score(y_test, lr_prob):.4f}")

# CELL 7: MODEL 2 — Random Forest
print("="*60)
print("MODEL 2: RANDOM FOREST")
print("="*60)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE)
rf_cv = cross_val_score(rf, X_orig_scaled, y, cv=cv, scoring='accuracy')
rf.fit(X_train_orig, y_train)
rf_pred = rf.predict(X_test_orig)
rf_prob = rf.predict_proba(X_test_orig)[:, 1]
print(f"CV: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"Test: {accuracy_score(y_test, rf_pred):.4f} | AUC: {roc_auc_score(y_test, rf_prob):.4f}")

# CELL 8: MODEL 3 — SVM
print("="*60)
print("MODEL 3: SVM (RBF)")
print("="*60)
svm = SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
svm_cv = cross_val_score(svm, X_orig_scaled, y, cv=cv, scoring='accuracy')
svm.fit(X_train_orig, y_train)
svm_pred = svm.predict(X_test_orig)
svm_prob = svm.predict_proba(X_test_orig)[:, 1]
print(f"CV: {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")
print(f"Test: {accuracy_score(y_test, svm_pred):.4f} | AUC: {roc_auc_score(y_test, svm_prob):.4f}")

# CELL 9: MODEL 4 — KNN
print("="*60)
print("MODEL 4: KNN (k=5)")
print("="*60)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_cv = cross_val_score(knn, X_orig_scaled, y, cv=cv, scoring='accuracy')
knn.fit(X_train_orig, y_train)
knn_pred = knn.predict(X_test_orig)
knn_prob = knn.predict_proba(X_test_orig)[:, 1]
print(f"CV: {knn_cv.mean():.4f} ± {knn_cv.std():.4f}")
print(f"Test: {accuracy_score(y_test, knn_pred):.4f} | AUC: {roc_auc_score(y_test, knn_prob):.4f}")

# CELL 10: Polynomial Features Comparison
print("="*60)
print("POLYNOMIAL FEATURES COMPARISON")
print("="*60)
models_poly = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE),
    'SVM (RBF)': SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance')
}
for name, model in models_poly.items():
    cv_scores = cross_val_score(model, X_poly_scaled, y, cv=cv, scoring='accuracy')
    model.fit(X_train_poly, y_train)
    pred = model.predict(X_test_poly)
    prob = model.predict_proba(X_test_poly)[:, 1] if hasattr(model, 'predict_proba') else None
    test_acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob) if prob is not None else None
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"{name:20s} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f} | Test: {test_acc:.4f} | AUC: {auc_str}")

# CELL 11: Hyperparameter Tuning — KNN
print("="*60)
print("HYPERPARAMETER TUNING: KNN")
print("="*60)
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=cv, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train_orig, y_train)
print(f"Best: {grid_knn.best_params_} | CV: {grid_knn.best_score_:.4f}")
print(f"Test: {accuracy_score(y_test, grid_knn.best_estimator_.predict(X_test_orig)):.4f}")

# CELL 12: Hyperparameter Tuning — Random Forest
print("="*60)
print("HYPERPARAMETER TUNING: RANDOM FOREST")
print("="*60)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE), 
                       param_grid_rf, cv=cv, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train_orig, y_train)
print(f"Best: {grid_rf.best_params_} | CV: {grid_rf.best_score_:.4f}")
print(f"Test: {accuracy_score(y_test, grid_rf.best_estimator_.predict(X_test_orig)):.4f}")

# CELL 13: GRAPH 1 — Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models_list = ['Logistic\nRegression', 'Random\nForest', 'SVM\n(RBF)', 'KNN\n(k=5)']
orig_cv = [lr_cv.mean(), rf_cv.mean(), svm_cv.mean(), knn_cv.mean()]
orig_test = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, rf_pred), 
             accuracy_score(y_test, svm_pred), accuracy_score(y_test, knn_pred)]
poly_cv = [0.4256, 0.4282, 0.3923, 0.4731]
poly_test = [0.2308, 0.2308, 0.3846, 0.4615]

x = np.arange(len(models_list))
width = 0.35

ax1 = axes[0]
ax1.bar(x - width/2, orig_cv, width, label='Original (5 feat)', color='#2980b9', alpha=0.85, edgecolor='white')
ax1.bar(x + width/2, poly_cv, width, label='Polynomial (20 feat)', color='#e74c3c', alpha=0.85, edgecolor='white')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('5-Fold CV Accuracy', fontsize=11)
ax1.set_title('Cross-Validation Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models_list, fontsize=10)
ax1.legend()
ax1.set_ylim(0, 0.7)

ax2 = axes[1]
ax2.bar(x - width/2, orig_test, width, label='Original (5 feat)', color='#2980b9', alpha=0.85, edgecolor='white')
ax2.bar(x + width/2, poly_test, width, label='Polynomial (20 feat)', color='#e74c3c', alpha=0.85, edgecolor='white')
ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('Test Set Accuracy', fontsize=11)
ax2.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models_list, fontsize=10)
ax2.legend()
ax2.set_ylim(0, 0.7)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_model_comparison.png', dpi=180, bbox_inches='tight')
plt.show()

# CELL 14: GRAPH 2 — Feature Importance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
lr.fit(X_train_orig, y_train)
rf.fit(X_train_orig, y_train)
svm_lin = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=RANDOM_STATE)
svm_lin.fit(X_train_orig, y_train)

# Logistic Regression
ax1 = axes[0, 0]
coefs = np.abs(lr.coef_[0])
sorted_idx = np.argsort(coefs)[::-1]
ax1.barh(range(len(FEATURES)), coefs[sorted_idx], color='#e74c3c', alpha=0.85)
ax1.set_yticks(range(len(FEATURES)))
ax1.set_yticklabels([FEATURES[i] for i in sorted_idx])
ax1.set_title('Logistic Regression Feature Importance', fontweight='bold')
ax1.invert_yaxis()

# Random Forest
ax2 = axes[0, 1]
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
ax2.barh(range(len(FEATURES)), importances[sorted_idx], color='#27ae60', alpha=0.85)
ax2.set_yticks(range(len(FEATURES)))
ax2.set_yticklabels([FEATURES[i] for i in sorted_idx])
ax2.set_title('Random Forest Feature Importance', fontweight='bold')
ax2.invert_yaxis()

# SVM Linear
ax3 = axes[1, 0]
svm_coefs = np.abs(svm_lin.coef_[0])
sorted_idx = np.argsort(svm_coefs)[::-1]
ax3.barh(range(len(FEATURES)), svm_coefs[sorted_idx], color='#9b59b6', alpha=0.85)
ax3.set_yticks(range(len(FEATURES)))
ax3.set_yticklabels([FEATURES[i] for i in sorted_idx])
ax3.set_title('SVM (Linear) Feature Importance', fontweight='bold')
ax3.invert_yaxis()

# KNN Proxy
ax4 = axes[1, 1]
correlations = [np.corrcoef(X_orig[:, i], y)[0, 1] for i in range(len(FEATURES))]
abs_corr = np.abs(correlations)
sorted_idx = np.argsort(abs_corr)[::-1]
ax4.barh(range(len(FEATURES)), abs_corr[sorted_idx], color='#f39c12', alpha=0.85)
ax4.set_yticks(range(len(FEATURES)))
ax4.set_yticklabels([FEATURES[i] for i in sorted_idx])
ax4.set_title('KNN Proxy (Correlation)', fontweight='bold')
ax4.invert_yaxis()

fig.suptitle('Feature Importance Across Four ML Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_feature_importance_all_models.png', dpi=180, bbox_inches='tight')
plt.show()

# CELL 15: GRAPH 3 — Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
models_cm = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'SVM (RBF)': svm,
    'KNN (k=5)': knn
}
positions = [(0,0), (0,1), (1,0), (1,1)]
for (name, model), (row, col) in zip(models_cm.items(), positions):
    ax = axes[row, col]
    model.fit(X_train_orig, y_train)
    y_pred = model.predict(X_test_orig)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    ax.imshow(cm, cmap='Blues', alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anoxic'])
    ax.set_yticklabels(['Normal', 'Anoxic'])
    ax.set_title(f'{name}\nAccuracy: {acc:.3f}', fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight='bold')

fig.suptitle('Confusion Matrices — Four ML Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_confusion_matrices_all_models.png', dpi=180, bbox_inches='tight')
plt.show()

# CELL 16: GRAPH 4 — CV Stability Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = [lr_cv, rf_cv, svm_cv, knn_cv]
labels = ['Logistic\nRegression', 'Random\nForest', 'SVM\n(RBF)', 'KNN\n(k=5)']
bp = ax.boxplot(cv_data, labels=labels, patch_artist=True,
                medianprops=dict(color='white', linewidth=2), widths=0.6)
colors = ['#2980b9', '#27ae60', '#e74c3c', '#9b59b6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guess (50%)')
ax.set_ylabel('5-Fold CV Accuracy', fontsize=12)
ax.set_title('Cross-Validation Stability Across Four ML Models', fontsize=13, fontweight='bold')
ax.set_ylim(0.2, 0.8)
ax.legend()
for i, scores in enumerate(cv_data):
    ax.scatter(i+1, scores.mean(), color='white', s=100, zorder=5, edgecolor='black', linewidth=1.5, marker='D')
    ax.text(i+1, scores.mean()+0.03, f'μ={scores.mean():.3f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_cv_stability_boxplot.png', dpi=180, bbox_inches='tight')
plt.show()

print("\n✅ ALL CELLS EXECUTED SUCCESSFULLY")
