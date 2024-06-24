import numpy as np
import pandas as pd
import joblib
import json

from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,precision_score

# load saved model
clf = joblib.load('sentiment_model.joblib')

# load test data
test_data = pd.read_csv('./data/features/test_bow.csv')

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

# predictions using model
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:,1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

with open('metrics.json','w') as f:
    json.dump(metrics_dict,f,indent=4)