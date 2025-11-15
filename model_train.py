# model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from preprocessing import load_and_preprocess


def train_models(path):
    df = load_and_preprocess(path)

    # Select feature columns (remove target)
    X = df.drop(columns=["Data Value"])
    y = df["Data Value"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["Linear Regression"] = r2_score(y_test, lr.predict(X_test))

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    results["Decision Tree"] = r2_score(y_test, dt.predict(X_test))

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    results["Random Forest"] = r2_score(y_test, rf.predict(X_test))

    # XGBoost
    xgb = XGBRegressor(random_state=42, n_estimators=200)
    xgb.fit(X_train, y_train)
    results["XGBoost"] = r2_score(y_test, xgb.predict(X_test))

    return results


if __name__ == "__main__":
    path = "../data/Air_Quality_Cleaned_Data.csv"
    scores = train_models(path)

    print("\nModel Performance (R² Scores):")
    for model, score in scores.items():
        print(f"{model}: {score:.4f}")
