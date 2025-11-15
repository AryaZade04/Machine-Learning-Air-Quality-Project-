# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    """
    Loads the air quality dataset and performs preprocessing:
    - Missing value handling (mean imputation)
    - Label encoding for categorical columns
    """

    df = pd.read_csv(path)

    # Handle missing numerical values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Label encode categorical columns
    label_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


if __name__ == "__main__":
    # Example run
    df = load_and_preprocess("../data/Air_Quality_Cleaned_Data.csv")
    print(df.head())
