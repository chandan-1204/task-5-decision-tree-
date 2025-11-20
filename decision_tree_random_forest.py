
import os
import fnmatch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ----------------- CONFIG -----------------
RAW_PATH = "C:\\Users\\chand\\Desktop\\House_prices\\heart.csv"
SEARCH_DIRS = ['.', './data', '/mnt/data', os.path.expanduser('~'), os.path.join(os.path.expanduser('~'),'OneDrive','Documents')]
OUT_DIR = 'tree_rf_outputs'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- helper: find dataset -----------------
def find_dataset(path_hint=RAW_PATH):
    if path_hint and os.path.exists(path_hint):
        return path_hint
    patterns = ['*heart*.csv','*heart*.xlsx','*data*.csv','*data*.xlsx','*.csv','*.xlsx']
    candidates = []
    for d in SEARCH_DIRS:
        try:
            for root, _, files in os.walk(d):
                for pat in patterns:
                    for name in fnmatch.filter(files, pat):
                        candidates.append(os.path.join(root, name))
        except Exception:
            pass
    return candidates[0] if candidates else None

dataset_path = find_dataset()
if dataset_path is None:
    raise FileNotFoundError("No dataset found. Put CSV/XLSX in project or update RAW_PATH.")
print("Loading dataset:", dataset_path)

# ----------------- load -----------------
if dataset_path.lower().endswith(('.xls','.xlsx')):
    df = pd.read_excel(dataset_path)
else:
    df = pd.read_csv(dataset_path)

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# ----------------- detect target -----------------
# common binary target names
TARGET_COL = None
common_targets = ['target','label','y','diagnosis','heartdisease','heart_disease','disease','condition','outcome','chd']
for name in common_targets:
    for c in df.columns:
        if c.lower() == name:
            TARGET_COL = c
            break
    if TARGET_COL:
        break

# If not auto-found, find any column with 2 unique values (excluding id-like)
if TARGET_COL is None:
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            TARGET_COL = c
            break

# fallback: last column
if TARGET_COL is None:
    TARGET_COL = df.columns[-1]

print("Detected target column:", TARGET_COL)
print(df[TARGET_COL].value_counts(dropna=False).head())

# ----------------- prepare target y (map to 0/1) -----------------
def to_binary(series):
    if series.dtype.kind in 'biufc':
        uniq = sorted(series.dropna().unique().tolist())
        if set(uniq).issubset({0,1}):
            return series.astype(int)
        if len(uniq) == 2:
            return series.map({uniq[0]:0, uniq[1]:1}).astype(int)
    # try mapping common strings
    s = series.astype(str).str.lower()
    pos = s.isin(['1','yes','y','true','t','positive','malignant','disease'])
    neg = s.isin(['0','no','n','false','f','negative','benign','healthy'])
    if pos.sum() + neg.sum() >= len(series) * 0.5:  # majority mapped
        out = pd.Series(np.nan, index=series.index)
        out[pos] = 1
        out[neg] = 0
        # fill remaining by factorize if still binary
        if out.isna().any():
            fac = pd.factorize(series)[0]
            uniq = np.unique(fac)
            if len(uniq) == 2:
                mapping = {uniq[0]:0, uniq[1]:1}
                out = pd.Series(fac).map(mapping).astype(int)
        return out.astype(int)
    # final fallback: factorize (map to 0/1 if two categories)
    fac = pd.factorize(series)[0]
    if len(np.unique(fac)) == 2:
        return pd.Series(fac).astype(int)
    # if not binary, raise
    raise ValueError("Target column could not be converted to binary (0/1). Column: " + str(series.name))

y = to_binary(df[TARGET_COL])
# drop rows where target is NaN (shouldn't happen)
mask = y.notna()
df = df.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

# ----------------- prepare X -----------------
X = df.drop(columns=[TARGET_COL]).copy()

# drop columns that are obviously IDs (all unique)
id_like = [c for c in X.columns if X[c].nunique() == X.shape[0]]
if id_like:
    print("Dropping ID-like columns:", id_like)
    X = X.drop(columns=id_like)

# detect numeric and categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)

# ----------------- preprocessing pipelines -----------------
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# handle OneHotEncoder sparse param depending on sklearn version
ohe_kwargs = {'handle_unknown':'ignore'}
try:
    # sklearn >=1.2
    ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
except TypeError:
    ohe = OneHotEncoder(sparse=False, **ohe_kwargs)

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', ohe)
])

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, cat_cols)
    ], remainder='drop'
)

# ----------------- Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train/test shapes:", X_train.shape, X_test.shape)

# ----------------- Decision Tree -----------------
dt_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
])
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

# metrics for decision tree
def metrics_report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}

dt_metrics = metrics_report(y_test, y_pred_dt)
print("\nDecision Tree metrics:", dt_metrics)
print("\nDecision Tree classification report:\n", classification_report(y_test, y_pred_dt, digits=4))

# confusion matrix save
cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
cm_dt_path = os.path.join(OUT_DIR, 'confusion_matrix_tree.png')
plt.savefig(cm_dt_path); plt.close()
print("Saved", cm_dt_path)

# ----------------- visualize tree (use plot_tree) -----------------
# get feature names after preprocessing (column names from transformer)
def get_feature_names(preprocessor, numeric_cols, cat_cols):
    out_names = []
    if numeric_cols:
        out_names.extend(numeric_cols)
    if cat_cols:
        # get categories from fitted onehot encoder
        ohe_step = None
        try:
            ohe_step = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cats = ohe_step.categories_
            cat_names = []
            for col, vals in zip(cat_cols, cats):
                cat_names.extend([f"{col}__{v}" for v in vals])
            out_names.extend(cat_names)
        except Exception:
            # fallback: generic names
            out_names.extend([f"cat_{i}" for i in range(len(cat_cols))])
    return out_names

# extract fitted decision tree estimator and feature names
fitted_pre = dt_pipeline.named_steps['pre']
fitted_clf = dt_pipeline.named_steps['clf']
feat_names = get_feature_names(fitted_pre, numeric_cols, cat_cols)

plt.figure(figsize=(18,10))
plot_tree(fitted_clf,
          feature_names=feat_names,
          class_names=[str(c) for c in np.unique(y)],
          filled=True,
          proportion=True,
          rounded=True,
          fontsize=8)
plt.title("Decision Tree")
plt.tight_layout()
tree_path = os.path.join(OUT_DIR, 'decision_tree.png')
plt.savefig(tree_path, dpi=200)
plt.close()
print("Saved tree visualization to", tree_path)

# save feature importances for decision tree
try:
    dt_importances = pd.DataFrame({
        'feature': feat_names,
        'importance': fitted_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    dt_importances.to_csv(os.path.join(OUT_DIR, 'feature_importance_tree.csv'), index=False)
    print("Saved decision tree feature importances.")
except Exception as e:
    print("Could not save decision tree importances:", e)

# ----------------- Random Forest -----------------
rf_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

rf_metrics = metrics_report(y_test, y_pred_rf)
print("\nRandom Forest metrics:", rf_metrics)
print("\nRandom Forest classification report:\n", classification_report(y_test, y_pred_rf, digits=4))

# confusion matrix for RF
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
cm_rf_path = os.path.join(OUT_DIR, 'confusion_matrix_rf.png')
plt.savefig(cm_rf_path); plt.close()
print("Saved", cm_rf_path)

# feature importances from RF
try:
    rf_clf = rf_pipeline.named_steps['clf']
    rf_feat_imp = pd.DataFrame({
        'feature': feat_names,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    rf_feat_imp.to_csv(os.path.join(OUT_DIR, 'feature_importance_rf.csv'), index=False)
    print("Saved Random Forest feature importances.")
except Exception as e:
    print("Could not save RF importances:", e)

# ----------------- cross-validation -----------------
# perform CV on the Random Forest pipeline (uses raw X,y; pipeline will preprocess inside CV)
cv_scores = cross_val_score(rf_pipeline, X, y, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
cv_df = pd.DataFrame({'cv_fold': list(range(1, CV_FOLDS+1)), 'accuracy': cv_scores})
cv_df.to_csv(os.path.join(OUT_DIR, 'cross_validation_scores.csv'), index=False)
print("Saved cross-validation scores:", cv_scores, "mean:", cv_scores.mean())

# ----------------- save models -----------------
joblib.dump(dt_pipeline, os.path.join(OUT_DIR, 'decision_tree_pipeline.joblib'))
joblib.dump(rf_pipeline, os.path.join(OUT_DIR, 'rf_pipeline.joblib'))
print("Saved pipeline models to", OUT_DIR)

# ----------------- summary CSV -----------------
summary = pd.DataFrame([{
    'model': 'decision_tree',
    'accuracy': dt_metrics['accuracy'],
    'precision': dt_metrics['precision'],
    'recall': dt_metrics['recall'],
    'f1': dt_metrics['f1']
},{
    'model': 'random_forest',
    'accuracy': rf_metrics['accuracy'],
    'precision': rf_metrics['precision'],
    'recall': rf_metrics['recall'],
    'f1': rf_metrics['f1']
}])
summary.to_csv(os.path.join(OUT_DIR, 'model_comparison_summary.csv'), index=False)
print("Saved model comparison summary.")

print("\nAll outputs saved to:", OUT_DIR)
print("Done.")
