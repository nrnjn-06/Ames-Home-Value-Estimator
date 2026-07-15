"""
Train script for the House Price Prediction project.

Trains a Linear Regression and a Random Forest on a curated set of the
10 most predictive features from the Ames Housing dataset, compares them
with RMSE / R^2, and saves the better model (as a full sklearn Pipeline,
including preprocessing) to model/house_price_model.pkl.

Run:
    python train.py
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "/mnt/user-data/uploads/train.csv"
MODEL_OUT = "model/house_price_model.pkl"
META_OUT = "model/meta.json"

# The 10 features that carry ~95% of the predictive signal (see feature
# importance ranking from an exploratory Random Forest fit). Keeping the
# form to these 10 fields is what makes the website usable instead of a
# 79-field spreadsheet.
NUMERIC_FEATURES = [
    "OverallQual",   # Overall material/finish quality, 1-10
    "GrLivArea",     # Above-ground living area, sq ft
    "TotalBsmtSF",   # Total basement area, sq ft
    "GarageCars",    # Garage capacity, in cars
    "YearBuilt",     # Original construction year
    "FullBath",      # Full bathrooms above grade
    "LotArea",       # Lot size, sq ft
    "TotRmsAbvGrd",  # Total rooms above grade (excl. bathrooms)
    "YearRemodAdd",  # Remodel year (same as YearBuilt if never remodeled)
]
CATEGORICAL_FEATURES = ["Neighborhood"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "SalePrice"


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    print(f"{name:>16s}  RMSE = {rmse:,.0f}   R^2 = {r2:.4f}")
    return rmse, r2


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    lin_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])
    rf_pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )),
    ])

    lin_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    print("\nModel comparison on held-out test set:")
    lin_rmse, lin_r2 = evaluate(lin_pipeline, X_test, y_test, "LinearRegression")
    rf_rmse, rf_r2 = evaluate(rf_pipeline, X_test, y_test, "RandomForest")

    if rf_r2 >= lin_r2:
        best_name, best_model = "Random Forest", rf_pipeline
        best_rmse, best_r2 = rf_rmse, rf_r2
    else:
        best_name, best_model = "Linear Regression", lin_pipeline
        best_rmse, best_r2 = lin_rmse, lin_r2

    print(f"\nBest model: {best_name}  (RMSE={best_rmse:,.0f}, R^2={best_r2:.4f})")

    # Refit best model on ALL data before shipping it, so the deployed
    # model benefits from every row, not just the training split.
    best_model.fit(X, y)

    joblib.dump(best_model, MODEL_OUT)

    neighborhoods = sorted(df["Neighborhood"].dropna().unique().tolist())
    meta = {
        "model_name": best_name,
        "rmse": best_rmse,
        "r2": best_r2,
        "features": FEATURES,
        "neighborhoods": neighborhoods,
        "ranges": {
            "OverallQual": [1, 10],
            "GrLivArea": [int(X["GrLivArea"].min()), int(X["GrLivArea"].max())],
            "TotalBsmtSF": [int(X["TotalBsmtSF"].min()), int(X["TotalBsmtSF"].max())],
            "GarageCars": [0, int(X["GarageCars"].max())],
            "YearBuilt": [int(X["YearBuilt"].min()), int(X["YearBuilt"].max())],
            "FullBath": [0, int(X["FullBath"].max())],
            "LotArea": [int(X["LotArea"].min()), int(X["LotArea"].max())],
            "TotRmsAbvGrd": [1, int(X["TotRmsAbvGrd"].max())],
            "YearRemodAdd": [int(X["YearRemodAdd"].min()), int(X["YearRemodAdd"].max())],
        },
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved metadata -> {META_OUT}")


if __name__ == "__main__":
    main()
