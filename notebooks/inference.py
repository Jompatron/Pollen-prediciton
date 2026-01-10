# inference.py
from __future__ import annotations
from datetime import timezone

from pathlib import Path
from datetime import datetime, timedelta, date

import os
import pandas as pd
import matplotlib.pyplot as plt

import hopsworks
from dotenv import load_dotenv
from xgboost import XGBRegressor


# -----------------------
# Config (match your repo)
# -----------------------
HOPSWORKS_HOST = "c.app.hopsworks.ai"
HOPSWORKS_PROJECT = "AirQualityKTHLab1"

CITY = "stockholm"
POLLEN_TYPES = [
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
]

# Feature group / view versions (you used v5 in training + ingest)
FG_VERSION = 5
FV_VERSION = 5

# Forecast horizon (days)
HORIZON_DAYS = 7

# Monitoring FG where predictions go
PRED_FG_NAME = "pollen_predictions"
PRED_FG_VERSION = 1

ROOT_DIR = Path.cwd()


# -----------------------
# Helpers
# -----------------------
def _to_dt(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True)
    return ts.tz_convert(None)


def forecast_recursive(
    model: XGBRegressor,
    row_anchor: pd.Series,
    rows_to_predict: pd.DataFrame,
    feature_cols: list[str],
    max_horizon: int,
    pollen_type: str,
) -> list[float]:
    preds: list[float] = []

    # Start lags from anchor row (yesterday)
    lag_1 = float(row_anchor[pollen_type])
    lag_2 = float(row_anchor[f"{pollen_type}_lag1"])
    lag_3 = float(row_anchor[f"{pollen_type}_lag2"])

    for h in range(max_horizon):
        r = rows_to_predict.iloc[h].copy()

        # Overwrite lags (never use "future" values)
        r[f"{pollen_type}_lag1"] = lag_1
        r[f"{pollen_type}_lag2"] = lag_2
        r[f"{pollen_type}_lag3"] = lag_3

        X_row = r[feature_cols].to_frame().T  
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        #y_hat = float(model.predict(X_row)[0])
        
        y_hat = max(0.0, float(model.predict(X_row)[0])) #non negative predictions


        preds.append(y_hat)

        # Shift lags forward
        lag_3 = lag_2
        lag_2 = lag_1
        lag_1 = y_hat

    return preds


def plot_forecast(dates: list[pd.Timestamp], preds: list[float], pollen_type: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{HORIZON_DAYS}-day forecast for {pollen_type} ({CITY})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Pollen level")
    ax.plot(dates, preds, marker="o", linewidth=2, label="Forecast")
    ax.legend(loc="upper left", fontsize="small")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def get_latest_model_from_registry(mr, model_name: str, download_dir: Path) -> tuple[Path, dict]:
    """
    Downloads latest model version. Returns (local_model_dir, model_metadata).
    Uses a couple approaches to be robust to small SDK differences.
    """
    # Try "get_latest_model" style first; fall back to listing.
    model = None
    try:
        model = mr.get_model(model_name, version=None)  # some SDKs treat version=None as latest
    except Exception:
        pass

    if model is None:
        # Fall back: list and pick highest version
        models = mr.get_models(model_name)
        if not models:
            raise RuntimeError(f"No model found in registry with name='{model_name}'")
        # Models usually have .version
        model = sorted(models, key=lambda m: int(getattr(m, "version", 0)))[-1]

    local_path = model.download(str(download_dir))
    meta = {}
    try:
        meta = model.to_dict()
    except Exception:
        # Not critical
        meta = {"name": model_name, "version": getattr(model, "version", None)}
    return Path(local_path), meta


def ensure_prediction_feature_group(fs):
    # 1) Try to fetch existing FG
    try:
        fg = fs.get_feature_group(name=PRED_FG_NAME, version=PRED_FG_VERSION)
        if fg is not None:
            print(f"Using existing Feature Group: {PRED_FG_NAME} v{PRED_FG_VERSION}")
            return fg
    except Exception:
        pass

    # 2) Create FG (older HSFS API)
    try:
        fg = fs.create_feature_group(
            name=PRED_FG_NAME,
            version=PRED_FG_VERSION,
            primary_key=["run_date", "date"],
            description="Monitoring FG: daily pollen forecasts (wide format) + model versions used.",
            online_enabled=False,
            event_time="date",   # optional; remove if your API complains
        )
        print(f"Created Feature Group: {PRED_FG_NAME} v{PRED_FG_VERSION}")
        return fg
    except Exception as e:
        raise RuntimeError(f"Failed to create Feature Group {PRED_FG_NAME} v{PRED_FG_VERSION}: {e}") from e





# -----------------------
# Main inference pipeline
# -----------------------
def main():
    load_dotenv()

    today = date.today()
    yesterday = today - timedelta(days=1)

    print(f"today: {today}")
    print(f"yesterday: {yesterday}")
    print(f"cwd: {ROOT_DIR}")

    project = hopsworks.login(host=HOPSWORKS_HOST, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Load feature groups
    pollen_fg = fs.get_feature_group(name="pollen", version=FG_VERSION)
    weather_fg = fs.get_feature_group(name="weather", version=FG_VERSION)

    # Read data
    df_pollen_hist = pollen_fg.read()
    df_weather_hist = weather_fg.read()

    # Normalize timestamps (Hopsworks often returns tz-aware UTC)
    df_pollen_hist["date"] = pd.to_datetime(df_pollen_hist["date"], utc=True).dt.tz_convert(None).dt.normalize()
    df_weather_hist["date"] = pd.to_datetime(df_weather_hist["date"], utc=True).dt.tz_convert(None).dt.normalize()

    # Shared date bounds for forecast horizon
    start_dt = pd.Timestamp(today).normalize()
    end_dt = pd.Timestamp(today + timedelta(days=HORIZON_DAYS - 1)).normalize()

    # Future weather rows for forecast horizon
    future_weather = df_weather_hist.loc[
        (df_weather_hist["date"] >= start_dt) & (df_weather_hist["date"] <= end_dt)
    ].copy()
    future_weather = future_weather.sort_values("date").reset_index(drop=True)

    if len(future_weather) < HORIZON_DAYS:
        raise RuntimeError(
            f"Weather FG does not contain enough future rows for {HORIZON_DAYS} days "
            f"(found {len(future_weather)} rows between {start_dt.date()} and {end_dt.date()}). "
            "Did daily_ingest insert weather forecasts for upcoming days?"
        )

    # Rename weather feature columns to match FV/model schema: weather_*
    future_weather = future_weather.rename(
        columns={c: f"weather_{c}" for c in future_weather.columns if c != "date"}
    )

    preds_wide = pd.DataFrame({"date": future_weather["date"]})
    model_versions = {}

    # Predict one pollen type at a time
    for pollen_type in POLLEN_TYPES:
        model_name = f"model_{pollen_type}"
        print(f"\n--- Inference for {pollen_type} using {model_name} ---")

        # Anchor row from pollen FG (yesterday must exist)
        anchor = df_pollen_hist.loc[df_pollen_hist["date"].dt.date == yesterday]
        if anchor.empty:
            raise RuntimeError(
                f"No pollen row found for yesterday={yesterday}. "
                "Did daily_ingest run and insert yesterday's pollen?"
            )
        anchor_row = anchor.sort_values("date").iloc[-1]

        # Build base rows for this pollen type: future weather + lag placeholders
        base_rows = future_weather.copy()

        for k in [1, 2, 3]:
            col = f"{pollen_type}_lag{k}"
            if col not in base_rows.columns:
                base_rows[col] = 0.0  # overwritten each step in recursion

        # Download model
        dl_dir = ROOT_DIR / f"downloads/{model_name}"
        dl_dir.mkdir(parents=True, exist_ok=True)

        local_model_dir, meta = get_latest_model_from_registry(mr, model_name, dl_dir)
        model_versions[pollen_type] = str(meta.get("version", "unknown"))

        model_path = local_model_dir / "model.json"
        if not model_path.exists():
            raise RuntimeError(f"Downloaded model is missing model.json at: {model_path}")

        xgb = XGBRegressor()
        xgb.load_model(str(model_path))

        # Use model feature names (guarantees correct order + names)
        feature_cols = xgb.get_booster().feature_names
        if not feature_cols:
            raise RuntimeError(
                f"Model {model_name} has no stored feature names. "
                "Retrain using a pandas DataFrame and re-upload."
            )

        missing = [c for c in feature_cols if c not in base_rows.columns]
        if missing:
            print("Sample base_rows columns:", list(base_rows.columns)[:30])
            print("Model expects:", feature_cols[:30])
            raise RuntimeError(f"Missing required features for {pollen_type}: {missing}")

        # Force numeric dtypes for all model features
        base_rows[feature_cols] = base_rows[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Also ensure the lag placeholders are numeric (extra safety)
        for k in [1, 2, 3]:
            col = f"{pollen_type}_lag{k}"
            base_rows[col] = pd.to_numeric(base_rows[col], errors="coerce").fillna(0.0)

        print(base_rows[feature_cols].dtypes)
        y_hat = forecast_recursive(
            model=xgb,
            row_anchor=anchor_row,
            rows_to_predict=base_rows,
            feature_cols=feature_cols,
            max_horizon=HORIZON_DAYS,
            pollen_type=pollen_type,
        )

        # Clamp negative values
        y_hat = [max(0.0, float(v)) for v in y_hat]

        preds_wide[pollen_type] = y_hat

        # Plot
        out_img = ROOT_DIR / f"pollen_model_{pollen_type}/images/forecast.png"
        plot_forecast(list(preds_wide["date"]), y_hat, pollen_type, out_img)
        print(f"Saved forecast plot -> {out_img}")

    # Save predictions to monitoring Feature Group
    preds_wide["model_versions"] = str(model_versions)
    preds_wide["created_at"] = pd.to_datetime(datetime.now(timezone.utc))
    preds_wide["run_date"] = pd.Timestamp.now(tz="Europe/Stockholm").normalize().tz_convert(None)
    preds_wide["date"] = pd.to_datetime(preds_wide["date"]).dt.normalize()

    pred_fg = ensure_prediction_feature_group(fs)
    if pred_fg is None:
        raise RuntimeError("Prediction Feature Group is None. Creation failed earlier.")
    pred_fg.insert(preds_wide, wait=True)

    print("Reading back predictions FG...")
    pred_fg = fs.get_feature_group(name=PRED_FG_NAME, version=PRED_FG_VERSION)
    df_check = pred_fg.read()
    print(df_check.sort_values(["run_date", "date"]).tail(10))

    print(
        "\nInserted predictions into Feature Group "
        f"'{PRED_FG_NAME}' v{PRED_FG_VERSION} (rows: {len(preds_wide)})"
    )

    dist_dir = ROOT_DIR / ".dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Remove internal-only columns before publishing
    publish_df = preds_wide.drop(columns=["model_versions"], errors="ignore")

    # Save clean CSV for GitHub Pages
    out_csv = dist_dir / "latest_predictions.csv"
    publish_df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()