import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier


# fetch training data from data/ features
train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Create model object and fit train data
clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train,y_train)

joblib.dump(clf,'sentiment_model.joblib')