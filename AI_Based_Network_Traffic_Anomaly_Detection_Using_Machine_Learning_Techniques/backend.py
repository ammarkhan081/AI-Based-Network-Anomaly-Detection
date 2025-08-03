# backend.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# ===============================
# Load CSVs
# ===============================
def load_dataset(path):
    df = pd.read_csv(path)
    return df

# ===============================
# Preprocessing function (fit for training)
# ===============================
def preprocess_training_data(df):
    df = df.dropna()
    df = df.drop_duplicates()

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le

# ===============================
# Preprocessing function (transform for testing)
# ===============================
def preprocess_testing_data(df, scaler, le):
    df = df.dropna()
    df = df.drop_duplicates()

    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])

    X = df.drop('label', axis=1)
    y = df['label']
    X_scaled = scaler.transform(X)

    return X_scaled, y

# ===============================
# Train Model
# ===============================
def train_model(X_train, y_train, model_name='rf'):
    if model_name == 'rf':
        model = RandomForestClassifier()
    elif model_name == 'lr':
        model = LogisticRegression()
    elif model_name == 'svm':
        model = SVC(probability=True)
    elif model_name == 'dt':
        model = DecisionTreeClassifier()
    elif model_name == 'xgb':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'lgb':
        model = lgb.LGBMClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model

# ===============================
# Evaluate Model
# ===============================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    if y_prob is not None:
        print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# Predict New Sample
# ===============================
def predict_sample(model, X_new):
    return model.predict(X_new)

# ===============================
# Run All (Training + Evaluation)
# ===============================
if __name__ == "__main__":
    train_path = r'C:\Users\FCT\Desktop\data\UNSW_NB15_training-set.csv'
    test_path = r'C:\Users\FCT\Desktop\data\UNSW_NB15_testing-set.csv'

    # Load datasets
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    # Preprocess training and test set separately
    X_train, y_train, scaler, le = preprocess_training_data(train_df)
    X_test, y_test = preprocess_testing_data(test_df, scaler, le)

    # Train model
    model = train_model(X_train, y_train, model_name='rf')

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Predict single sample
    print("Prediction for one test sample:", predict_sample(model, X_test[0].reshape(1, -1)))
