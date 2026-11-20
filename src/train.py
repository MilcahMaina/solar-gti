
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (BASE_FEATURES, DAYLIGHT_THRESHOLD, FULL_FEATURES, MODEL_DIR,
                    PROCESSED_PATH, RANDOM_STATE, TARGET, TEST_START)

def scores(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p)}
def evaluate(y, p):
    """Metrics on all test hours and on daylight hours only (night zeros flatter R2)."""
    p = np.clip(p, 0, None)
    day = y > DAYLIGHT_THRESHOLD
    out = scores(y, p)
    out.update({f"{k}_day": v for k, v in scores(y[day], p[day]).items()})
    return out

def sklearn_models():
    return {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                              n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": GridSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {"n_estimators": [200, 500], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            cv=TimeSeriesSplit(n_splits=5),          # folds respect time order
            scoring="neg_root_mean_squared_error"),
    }

def train_mlp(X, y):
    from tensorflow import keras
    keras.utils.set_random_seed(RANDOM_STATE)
    scaler = StandardScaler().fit(X)
    model = keras.Sequential([
        keras.Input(shape=(X.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(scaler.transform(X), y.values, epochs=300, batch_size=64, verbose=0,
              validation_split=0.2,                  # last 20% of training period
              callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    return model, scaler


def save_served_model(name, model, scaler=None):
    MODEL_DIR.mkdir(exist_ok=True)
    for f in ["model.joblib", "model.keras", "scaler.joblib"]:
        (MODEL_DIR / f).unlink(missing_ok=True)
    if name == "MLP":
        model.save(MODEL_DIR / "model.keras")
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    else:
        joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "model_info.json").write_text(
        json.dumps({"model": name, "features": BASE_FEATURES}, indent=2))


def main():
    df = pd.read_csv(PROCESSED_PATH, index_col="time", parse_dates=True)
    train, test = df[df.index < TEST_START], df[df.index >= TEST_START]
    y_tr, y_te = train[TARGET], test[TARGET]
    print(f"train {len(train):,} rows ({train.index.min():%Y-%m-%d} to {train.index.max():%Y-%m-%d}), "
          f"test {len(test):,} rows ({test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d})")

    # Reference point: GTI at the same hour yesterday
    rows = [{"features": "baseline", "model": "Persistence (t-24h)",
             **evaluate(y_te, test["gti_lag24"].values)}]
    served = {}

    for set_name, cols in [("weather", BASE_FEATURES), ("weather+lags", FULL_FEATURES)]:
        X_tr, X_te = train[cols], test[cols]
        for name, model in sklearn_models().items():
            model.fit(X_tr, y_tr)
            if name == "XGBoost":
                print(f"  XGBoost best params ({set_name}): {model.best_params_}")
                model = model.best_estimator_
            rows.append({"features": set_name, "model": name, **evaluate(y_te, model.predict(X_te))})
            if set_name == "weather":
                served[name] = (model, None)
            print(f"  done: {set_name:13s} {name}")
        mlp, scaler = train_mlp(X_tr, y_tr)
        p = mlp.predict(scaler.transform(X_te), verbose=0).ravel()
        rows.append({"features": set_name, "model": "MLP", **evaluate(y_te, p)})
        if set_name == "weather":
            served["MLP"] = (mlp, scaler)
        print(f"  done: {set_name:13s} MLP")

    results = pd.DataFrame(rows).round(3)
    MODEL_DIR.mkdir(exist_ok=True)
    results.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n" + results.to_string(index=False))

    # The API takes weather + time only (a user has no measured GTI history),
    # so serve the best weather-only model by test RMSE.
    weather = results[results["features"] == "weather"]
    best = weather.loc[weather["RMSE"].idxmin(), "model"]
    save_served_model(best, *served[best])
    print(f"\nServed model: {best} (weather features) -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
