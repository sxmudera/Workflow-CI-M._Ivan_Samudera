import pandas as pd
import mlflow
import dagshub
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

token = os.getenv("MLFLOW_TRACKING_PASSWORD")
if token:
    os.environ['DAGSHUB_USER_TOKEN'] = token

dagshub.init(repo_owner='sxmudera', repo_name='Heart-Failure-Project', mlflow=True)

train = pd.read_csv('heart_failure_preprocessing/train_data.csv')
test = pd.read_csv('heart_failure_preprocessing/test_data.csv')

X_train = train.drop('DEATH_EVENT', axis=1)
y_train = train['DEATH_EVENT']
X_test = test.drop('DEATH_EVENT', axis=1)
y_test = test['DEATH_EVENT']

with mlflow.start_run():
    rf = RandomForestClassifier()
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    clf = GridSearchCV(rf, params, cv=3)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_params(clf.best_params_)
    
    mlflow.sklearn.log_model(
        sk_model=clf.best_estimator_,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")
    
    plt.figure()
    plt.barh(X_train.columns, clf.best_estimator_.feature_importances_)
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

joblib.dump(clf.best_estimator_, 'model.pkl')