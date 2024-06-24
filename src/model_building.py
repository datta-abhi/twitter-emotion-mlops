
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore") 

# fetching params from yaml file
params = yaml.safe_load(open('params.yaml','r'))['model-building']

# fetch training data from data/ features
train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Create model object and fit train data
clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
clf.fit(X_train,y_train)

joblib.dump(clf,'sentiment_model.joblib')