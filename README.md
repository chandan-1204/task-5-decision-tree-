# Decision Trees & Random Forests – Task 5

This project implements Decision Tree and Random Forest classifiers using Scikit-learn.  
It includes preprocessing, model training, visualization, feature importance analysis, and cross-validation.

------------------------------------------------------------

## Project Structure

decision_tree_random_forest.py     # Main script
tree_rf_outputs/                    # Auto-generated output folder
│
├── decision_tree.png
├── confusion_matrix_tree.png
├── confusion_matrix_rf.png
├── feature_importance_tree.csv
├── feature_importance_rf.csv
├── cross_validation_scores.csv
├── model_comparison_summary.csv
├── decision_tree_pipeline.joblib
└── rf_pipeline.joblib

------------------------------------------------------------

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- graphviz (optional for tree visualization)

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn joblib graphviz

Graphviz download (optional):  
https://graphviz.org/download/

------------------------------------------------------------

## Features Implemented

1. Dataset Loading & Preprocessing
- Automatically detects dataset (CSV/XLSX)
- Detects binary target column
- Handles numeric and categorical columns
- Imputes missing values
- One-hot encodes categorical variables
- Standardizes numeric features
- Automatically drops ID-like columns

2. Decision Tree Classifier
- Full decision tree training
- Evaluation: accuracy, precision, recall, F1-score
- Confusion matrix saved as PNG
- Exports feature importances (CSV)
- Saves a visualized tree (decision_tree.png)

3. Random Forest Classifier
- Trains a Random Forest ensemble model
- Confusion matrix saved as PNG
- Feature importances saved as CSV
- Saves trained model pipeline (.joblib)

4. Cross-Validation
- Performs 5-fold cross-validation
- Saves fold-by-fold accuracy scores (CSV)

5. Output Folder
All files are saved inside:

tree_rf_outputs/

------------------------------------------------------------

## How to Run

1. Place your dataset (e.g., heart.csv) in the project directory.  
2. Run the script:

python decision_tree_random_forest.py

3. All outputs will appear in:

tree_rf_outputs/

------------------------------------------------------------

## Output Files Explained

- decision_tree.png  
  Visualized decision tree structure.

- confusion_matrix_tree.png  
  Confusion matrix for Decision Tree.

- confusion_matrix_rf.png  
  Confusion matrix for Random Forest.

- feature_importance_tree.csv  
  Feature importance values from Decision Tree.

- feature_importance_rf.csv  
  Feature importance values from Random Forest.

- cross_validation_scores.csv  
  Accuracy scores from 5-fold CV.

- model_comparison_summary.csv  
  Comparison table of DT vs RF performance metrics.

- decision_tree_pipeline.joblib  
  Saved Decision Tree model.

- rf_pipeline.joblib  
  Saved Random Forest model.

------------------------------------------------------------
