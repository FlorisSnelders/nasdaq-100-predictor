from pathlib import Path

import joblib

from src.storage import load_model_frame, save_predictions

MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "prod_model_training_xgboost.joblib"
)


def write_predictions(model_path=MODEL_PATH):
    bundle = joblib.load(model_path)
    model, features = bundle["model"], bundle["features"]
    df = load_model_frame()
    latest = df.sort_values("date").groupby("ticker", as_index=False).tail(1)
    proba = model.predict_proba(latest[features])[:, 1]
    latest = latest.assign(prediction=(proba >= 0.5).astype(int), probability=proba)
    return save_predictions(latest[["ticker", "date", "prediction", "probability"]])