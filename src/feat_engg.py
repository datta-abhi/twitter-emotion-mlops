import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

# fetching data from data/ processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

# feature and target defined
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=1000)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

# adding the target back to train and test after vectorizing
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# saving the data after BOW vectorizing
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir/'data'/'features'

data_dir.mkdir(parents=True,exist_ok=True)

# Define the file paths
train_file_path = data_dir / 'train_bow.csv'
test_file_path = data_dir / 'test_bow.csv'

# Save the train and test data as CSV files
train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)