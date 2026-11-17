# Breast Cancer Classification (WDBC)
# Benign vs malignant using PCA + Logistic Regression / SVM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score, precision_score, accuracy_score)
# 1. Load the data
# wdbc.data has no header, so we build the column names from wdbc.names
base_names = ["radius", "texture", "perimeter", "area", "smoothness",
              "compactness", "concavity", "concave_points", "symmetry",
              "fractal_dimension"]

columns = ["id", "diagnosis"]
for suffix in ["mean", "se", "worst"]:
    columns += [f"{name}_{suffix}" for name in base_names]

df = pd.read_csv("wdbc.data", header=None, names=columns)

# 2.  exploration

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("\nClass counts:")
print(df["diagnosis"].value_counts())

print("\nFeature summary (first few mean features):")
print(df[["radius_mean", "texture_mean", "area_mean", "smoothness_mean"]].describe().round(2))

print("\nAverage values by diagnosis:")
print(df.groupby("diagnosis")[["radius_mean", "area_mean", "concavity_mean"]].mean().round(3))

# a lot of these features are basically the same thing (radius, perimeter, area)
corr = df.drop(columns=["id", "diagnosis"]).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
print("\nHighly correlated pairs (>0.9):", (upper.stack() > 0.9).sum())

# 3. X and y

X = df.drop(columns=["id", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)   # 1 = malignant, 0 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. How many principal components do we need?

scaler = StandardScaler()
pca_full = PCA().fit(scaler.fit_transform(X_train))
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = int(np.argmax(cum_var >= 0.95)) + 1
print(f"\nComponents needed for 95% variance: {n_95}")

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=3)
plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Build models (scaler -> PCA -> classifier)

# class_weight="balanced" gives malignant cases more weight,
# which helps push recall up
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
    ]),
}

# 6. Cross-validation on the training set (scored on recall)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n5-fold CV recall on training data:")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
    print(f"  {name}: {scores.mean():.3f} (+/- {scores.std():.3f})")


# 7. Evaluate on the held-out test set

print("\nTest set results")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    print(f"\n{name}")
    print(f"  Accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"  Recall   : {recall_score(y_test, pred):.3f}")
    print(f"  Precision: {precision_score(y_test, pred):.3f}")
    print(f"  Confusion matrix -> TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))