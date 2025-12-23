import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

dagshub.init(repo_owner='sxmudera', repo_name='Heart-Failure-Project', mlflow=True)

# Load Data (Pastikan path file csv benar, copy folder hasil preprocessing ke sini jika perlu)
# Agar mudah, copy folder 'heart_failure_preprocessing' dari langkah 1 ke dalam folder 'Membangun_model'
train = pd.read_csv('../Eksperimen_SML_M._Ivan_Samudera/preprocessing/heart_failure_preprocessing/train_data.csv')
test = pd.read_csv('../Eksperimen_SML_M._Ivan_Samudera/preprocessing/heart_failure_preprocessing/test_data.csv')

X_train = train.drop('DEATH_EVENT', axis=1)
y_train = train['DEATH_EVENT']
X_test = test.drop('DEATH_EVENT', axis=1)
y_test = test['DEATH_EVENT']

mlflow.set_experiment("Heart Failure Tuning")

with mlflow.start_run():
    # 1. Tuning
    rf = RandomForestClassifier()
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    clf = GridSearchCV(rf, params, cv=3)
    clf.fit(X_train, y_train)
    
    # 2. Log Metrics Manual (Advance)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_params(clf.best_params_)
    
    # 3. Log Model
    mlflow.sklearn.log_model(clf.best_estimator_, "model")
    
    # 4. Log Artifact Tambahan (Advance minimal 2)
    # Artifact 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Artifact 2: Feature Importance
    plt.figure()
    plt.barh(X_train.columns, clf.best_estimator_.feature_importances_)
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    print("Selesai! Cek DagsHub.")