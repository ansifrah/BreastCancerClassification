# Breast Cancer Classification

This project builds a machine learning pipeline to classify breast tumors as benign or malignant using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The script uses a combination of data preprocessing, feature standardization, principal component analysis (PCA), and two classifiers:

- Logistic Regression
- Support Vector Machine (SVM) with an RBF kernel

It evaluates each model using 5-fold cross-validation and reports classification metrics on a held-out test set.

## Dataset

The project uses the WDBC dataset from the UCI Machine Learning Repository.

- File: `wdbc.data`
- Metadata: `wdbc.names`
- Total instances: 569
- Features: 30 real-valued measurements describing cell nucleus attributes
- Target: diagnosis (`B` = benign, `M` = malignant)

## Project structure

- `breast_cancer_classification.py` — main training and evaluation script
- `wdbc.data` — dataset values
- `wdbc.names` — dataset documentation and attribute descriptions

## Workflow

The script performs the following steps:

1. Loads the dataset and names the columns based on the WDBC feature schema.
2. Explores the dataset, including shape, missing values, class distribution, and summary statistics.
3. Builds a binary target (`1` = malignant, `0` = benign).
4. Splits the data into training and testing sets using stratification.
5. Fits a `StandardScaler` and computes PCA components required to retain 95% of the variance.
6. Trains and compares:
   - Logistic Regression with class balancing
   - SVM (RBF) with class balancing
7. Evaluates models with cross-validation on recall and then on the test set using:
   - accuracy
   - recall
   - precision
   - confusion matrix
   - classification report

## Requirements

Install the required Python dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

Run the project from the repository root:

```bash
python breast_cancer_classification.py
```

This will print dataset diagnostics and model evaluation results, and it may display a PCA explained-variance plot.

## Notes

- The code uses PCA to reduce dimensionality while keeping enough information to explain 95% of the variance.
- The class-weight balancing helps improve recall for the minority malignant class.
- The dataset is a well-known benchmark for binary medical classification and is commonly used to evaluate diagnostic models.