import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# reading params yaml file for hyperparameter
def load_params(param_path: str,param: str) -> float:
    param_val = yaml.safe_load(open(param_path,'r'))['data-ingestion'][param]
    return param_val

# read data from url
def read_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def process_data(df: pd.DataFrame)-> pd.DataFrame:
    
    # drop tweet id
    df.drop(columns=['tweet_id'],inplace=True)

    # convert to binary classification
    final_df = df[df['sentiment'].isin(['neutral','sadness'])]

    # encode target feature
    final_df['sentiment'] = final_df['sentiment'].map({'neutral': 1, 'sadness': 0})
    
    return final_df

def save_data(data_dir:str,train_data: pd.DataFrame,test_data: pd.DataFrame)-> None:
    
    # create dir if they do not exist
    data_dir.mkdir(parents=True,exist_ok=True)

    # Define the file paths
    train_file_path = data_dir / 'train.csv'
    test_file_path = data_dir / 'test.csv'

    # Save the train and test data as CSV files
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
    
def data_ingestion_main():
    
    test_size = load_params('params.yaml','test_size')
    
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    
    final_df = process_data(df)    
    
    # train test split
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=28)
    
    # define paths using pathlib
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir/'data'/'raw'
    
    save_data(data_dir,train_data,test_data)
    
if __name__=="_main__":
    data_ingestion_main()