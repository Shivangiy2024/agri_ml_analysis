"""
AGRICULTURAL ML - CROP YIELD PREDICTION
Week 1: Data cleaning, exploratory data analysis, and baseline ML model.

Input:
    agriculture.csv in the same folder as this file.

Required CSV columns:
    district, year, crop, area_ha, production_tonnes

Optional columns:
    state, yield_t_ha, rainfall_mm, temperature_c

Run:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python agricultural_ml.py

Note:
    Confirm the source dataset's units before interpreting yield values.
    This is a baseline project script, not a production-ready forecasting system.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "agriculture.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_data():
    """Load the input CSV and standardize column names."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.name}. Put it in: {BASE_DIR}"
        )

    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip().str.lower()

    required = [
        "district", "year", "crop", "area_ha", "production_tonnes"
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return data


def clean_data(data):
    """Remove duplicates/invalid records and calculate crop yield."""
    data = data.copy()
    data = data.drop_duplicates()

    numeric_columns = [
        "year", "area_ha", "production_tonnes"
    ]
    for column in ["yield_t_ha", "rainfall_mm", "temperature_c"]:
        if column in data.columns:
            numeric_columns.append(column)

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["district"] = data["district"].astype("string").str.strip()
    data["crop"] = data["crop"].astype("string").str.strip()

    data.loc[data["area_ha"] <= 0, "area_ha"] = np.nan
    data.loc[data["production_tonnes"] < 0, "production_tonnes"] = np.nan

    data = data.dropna(
        subset=[
            "district", "crop", "year",
            "area_ha", "production_tonnes"
        ]
    )

    data["calculated_yield"] = (
        data["production_tonnes"] / data["area_ha"]
    )

    if "yield_t_ha" in data.columns:
        data.loc[data["yield_t_ha"] <= 0, "yield_t_ha"] = np.nan
        data["yield_t_ha"] = data["yield_t_ha"].fillna(
            data["calculated_yield"]
        )
    else:
        data["yield_t_ha"] = data["calculated_yield"]

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["yield_t_ha"])
    data = data[data["yield_t_ha"] >= 0]

    return data


def run_eda(data):
    """Print summary statistics and save exploratory charts."""
    print("\n--- DATA SUMMARY ---")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print("\nMissing values:")
    print(data.isnull().sum())
    print("\nStatistical summary:")
    print(data.describe(include="all"))

    crop_yield = (
        data.groupby("crop")["yield_t_ha"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    crop_yield.plot(kind="bar")
    plt.title("Average Crop Yield")
    plt.xlabel("Crop")
    plt.ylabel("Yield (tonnes/hectare)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "crop_yield.png", dpi=150)
    plt.close()

    yearly_production = data.groupby("year")["production_tonnes"].sum()

    plt.figure(figsize=(10, 6))
    yearly_production.plot(kind="line", marker="o")
    plt.title("Year-wise Agricultural Production")
    plt.xlabel("Year")
    plt.ylabel("Production (tonnes)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "production_trend.png", dpi=150)
    plt.close()

    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.shape[1] >= 2:
        plt.figure(figsize=(9, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
        plt.close()

    crop_yield.to_csv(OUTPUT_DIR / "average_yield_by_crop.csv")
    print("\nAverage yield by crop:")
    print(crop_yield)


def prepare_features(data):
    """Encode crop/district and prepare numeric model features."""
    model_data = data.copy()

    # Use consistent category codes across the full dataset.
    model_data["crop_encoded"] = pd.Categorical(
        model_data["crop"].astype(str)
    ).codes
    model_data["district_encoded"] = pd.Categorical(
        model_data["district"].astype(str)
    ).codes

    feature_columns = [
        "year", "area_ha", "crop_encoded", "district_encoded"
    ]
    for column in ["rainfall_mm", "temperature_c"]:
        if column in model_data.columns:
            feature_columns.append(column)

    X = model_data[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    y = pd.to_numeric(model_data["yield_t_ha"], errors="coerce")

    valid = y.notna()
    return model_data.loc[valid].copy(), X.loc[valid], y.loc[valid], feature_columns


def chronological_split(data, X, y):
    """Split by year so later years are held out for testing."""
    years = sorted(data["year"].dropna().unique())
    if len(years) < 2:
        raise ValueError(
            "At least two different years are required for a chronological split."
        )

    split_position = max(1, int(len(years) * 0.8))
    split_position = min(split_position, len(years) - 1)
    cutoff_year = years[split_position]

    train_mask = data["year"] < cutoff_year
    test_mask = data["year"] >= cutoff_year

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Not enough records on both sides of the year split.")

    return (
        X.loc[train_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[test_mask],
        cutoff_year,
    )


def train_and_evaluate(data, X, y, feature_columns):
    """Train a baseline Random Forest and save evaluation outputs."""
    X_train, X_test, y_train, y_test, cutoff_year = chronological_split(
        data, X, y
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

    print("\n--- MODEL EVALUATION ---")
    print(f"Training records: {len(y_train)}")
    print(f"Testing records: {len(y_test)}")
    print(f"Test period starts at year: {cutoff_year}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    results = pd.DataFrame({
        "actual_yield_t_ha": y_test.to_numpy(),
        "predicted_yield_t_ha": predictions,
    })
    results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
        name="importance",
    ).sort_values(ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel("Actual yield (tonnes/hectare)")
    plt.ylabel("Predicted yield (tonnes/hectare)")
    plt.title("Actual vs Predicted Crop Yield")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading agricultural dataset...")
    raw_data = load_data()

    print("Cleaning data...")
    cleaned_data = clean_data(raw_data)
    if cleaned_data.empty:
        raise ValueError("No usable records remain after data cleaning.")

    cleaned_data.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)

    print("Running exploratory data analysis...")
    run_eda(cleaned_data)

    print("Preparing features and training model...")
    model_data, X, y, feature_columns = prepare_features(cleaned_data)
    train_and_evaluate(model_data, X, y, feature_columns)

    print("\nProject run completed.")
    print(f"Check output files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
