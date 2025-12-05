import joblib
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
    model = joblib.load(model_path)
    return model
