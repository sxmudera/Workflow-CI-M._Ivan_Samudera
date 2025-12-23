import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

token = os.getenv("MLFLOW_TRACKING_PASSWORD")
if token:
    os.environ['DAGSHUB_USER_TOKEN'] = token

dagshub.init(repo_owner='sxmudera', repo_name='Heart-Failure-Project', mlflow=True)

with mlflow.start_run():
    train = pd.read_csv('train_data.csv')
    X_train = train.drop('DEATH_EVENT', axis=1)
    y_train = train['DEATH_EVENT']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, 'model.pkl')