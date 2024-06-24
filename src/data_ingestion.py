import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# reading params yaml file for hyperparameter
test_size = yaml.safe_load(open('params.yaml','r'))['data-ingestion']['test_size']

# read data from url
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# drop tweet id
df.drop(columns=['tweet_id'],inplace=True)

# convert to binary classification
final_df = df[df['sentiment'].isin(['neutral','sadness'])]

# encode target feature
final_df['sentiment'] = final_df['sentiment'].map({'neutral': 1, 'sadness': 0})

# train test split
train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=28)

# define paths using pathlib
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir/'data'/'raw'

# create dir if they do not exist
data_dir.mkdir(parents=True,exist_ok=True)

# Define the file paths
train_file_path = data_dir / 'train.csv'
test_file_path = data_dir / 'test.csv'

# Save the train and test data as CSV files
train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

# print(df['sentiment'].value_counts())
# print('-'*25)
# print(final_df['sentiment'].value_counts())
# print(train_data.shape)
# print(test_data.shape)