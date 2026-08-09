import joblib
import pandas as pd

def predict_drop_episode(anime_data: dict) -> float:
    model1 = joblib.load("models/model_pipeline.joblib")
    model2 = joblib.load("models/random_forest_model.pkl")
    row = pd.DataFrame([anime_data])
    return float(0.5 * model1.predict(row)[0] + 0.5 * model2.predict(row)[0])