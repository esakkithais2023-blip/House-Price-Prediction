"""
House Price Prediction

Data Science Project: Predict house prices using features such as
location, number of bedrooms, area, and bathrooms.

Pipeline:
1. Data collection (synthetic dataset generator included — swap in a
   real CSV any time by editing load_data())
2. Preprocessing (missing values, encoding, scaling)
3. Exploratory Data Analysis (EDA)
4. Feature selection
5. Model training (Linear Regression + Random Forest comparison)
6. Evaluation (MAE, RMSE, R2)
7. Final demonstration (predict price for a new house)

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# 1. DATA COLLECTION
# ---------------------------------------------------------------------
def load_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Load the housing dataset.

    NOTE: No dataset was provided, so this generates a realistic
    synthetic dataset with the features called out in the brief:
    location, bedrooms, area, bathrooms.

    ---> To use a real dataset instead, replace this function body with:
         df = pd.read_csv("your_dataset.csv")
         return df
    """
    locations = ["Downtown", "Suburb", "Rural", "Uptown", "Waterfront"]
    location_price_factor = {
        "Downtown": 1.6, "Uptown": 1.4, "Waterfront": 1.8,
        "Suburb": 1.1, "Rural": 0.8,
    }

    location = np.random.choice(locations, n_samples, p=[0.25, 0.2, 0.15, 0.2, 0.2])
    area = np.random.normal(1500, 500, n_samples).clip(400, 5000)          # sq ft
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)                             # years

    # Base price formula + noise, to keep price plausibly related to features
    base_price = (
        area * 120
        + bedrooms * 15000
        + bathrooms * 10000
        - age * 800
    )
    loc_factor = np.array([location_price_factor[l] for l in location])
    noise = np.random.normal(0, 20000, n_samples)
    price = (base_price * loc_factor + noise).clip(30000, None)

    df = pd.DataFrame({
        "location": location,
        "area_sqft": area.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "price": price.round(0),
    })

    # Inject a few missing values, like a real-world dataset would have
    for col in ["area_sqft", "bathrooms"]:
        idx = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
        df.loc[idx, col] = np.nan

    return df


# ---------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()

    print("Missing values before cleaning:\n", df.isnull().sum(), sep="")

    # Fill numeric missing values with the median (robust to outliers)
    for col in ["area_sqft", "bathrooms"]:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical feature (location) with one-hot encoding
    df_encoded = pd.get_dummies(df, columns=["location"], drop_first=True)

    print("\nMissing values after cleaning:\n", df_encoded.isnull().sum().sum(),
          "total missing values remain")

    return df, df_encoded


# ---------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------
def run_eda(df: pd.DataFrame, outdir: str = "."):
    print("\n--- Dataset summary ---")
    print(df.describe(include="all"))

    # Price distribution
    plt.figure()
    sns.histplot(df["price"], kde=True, color="steelblue")
    plt.title("Distribution of House Prices")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(f"{outdir}/eda_price_distribution.png", dpi=150)
    plt.close()

    # Price by location
    plt.figure()
    sns.boxplot(data=df, x="location", y="price")
    plt.title("Price by Location")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{outdir}/eda_price_by_location.png", dpi=150)
    plt.close()

    # Area vs price
    plt.figure()
    sns.scatterplot(data=df, x="area_sqft", y="price", hue="bedrooms", palette="viridis")
    plt.title("Area vs Price (colored by bedrooms)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/eda_area_vs_price.png", dpi=150)
    plt.close()

    # Correlation heatmap (numeric features only)
    plt.figure()
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{outdir}/eda_correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nEDA plots saved to '{outdir}/' as PNG files.")


# ---------------------------------------------------------------------
# 4. FEATURE SELECTION
# ---------------------------------------------------------------------
def select_features(X: pd.DataFrame, y: pd.Series, k: int = 5):
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance scores (f_regression):")
    print(scores)

    selected_cols = X.columns[selector.get_support()]
    print(f"\nSelected top {len(selected_cols)} features: {list(selected_cols)}")
    return X[selected_cols], list(selected_cols)


# ---------------------------------------------------------------------
# 5 & 6. MODEL TRAINING + EVALUATION
# ---------------------------------------------------------------------
def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"\n{name}")
        print(f"  MAE : {mae:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R2  : {r2:.4f}")

    # Plot actual vs predicted for the best model (highest R2)
    best_name = max(results, key=lambda n: results[n]["R2"])
    best_model = results[best_name]["model"]
    best_preds = best_model.predict(X_test_scaled)

    plt.figure()
    plt.scatter(y_test, best_preds, alpha=0.5, color="teal")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Price ({best_name})")
    plt.tight_layout()
    plt.savefig("model_actual_vs_predicted.png", dpi=150)
    plt.close()

    print(f"\nBest model: {best_name} (R2 = {results[best_name]['R2']:.4f})")
    print("Plot saved as 'model_actual_vs_predicted.png'")

    return results, best_name, scaler


# ---------------------------------------------------------------------
# 7. FINAL DEMONSTRATION
# ---------------------------------------------------------------------
def demo_prediction(model, scaler, feature_cols, sample_input: dict):
    """Predict the price of a single new house given its features."""
    sample_df = pd.DataFrame([sample_input])

    # Ensure all expected columns exist (fill missing dummy columns with 0)
    for col in feature_cols:
        if col not in sample_df.columns:
            sample_df[col] = 0
    sample_df = sample_df[feature_cols]

    sample_scaled = scaler.transform(sample_df)
    predicted_price = model.predict(sample_scaled)[0]
    return predicted_price


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 60)
    print("HOUSE PRICE PREDICTION PROJECT")
    print("=" * 60)

    # 1. Collect data
    raw_df = load_data(n_samples=1000)

    # 2. Preprocess
    clean_df, encoded_df = preprocess(raw_df)

    # 3. EDA (on the cleaned but not-yet-encoded data, for readable plots)
    run_eda(clean_df)

    # 4. Feature selection
    X = encoded_df.drop(columns=["price"])
    y = encoded_df["price"]
    X_selected, selected_cols = select_features(X, y, k=5)

    # 5 & 6. Train + evaluate
    results, best_name, scaler = train_and_evaluate(X_selected, y)
    best_model = results[best_name]["model"]

    # 7. Final demonstration
    print("\n" + "=" * 60)
    print("FINAL DEMONSTRATION: Predicting price for a new house")
    print("=" * 60)

    sample_house = {
        "area_sqft": 1800,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 5,
        "location_Waterfront": 1,   # one-hot flag; will be 0 if not selected
    }
    predicted = demo_prediction(best_model, scaler, selected_cols, sample_house)
    print(f"Sample house: {sample_house}")
    print(f"Predicted price: {predicted:,.2f}")


if __name__ == "__main__":
    main()
