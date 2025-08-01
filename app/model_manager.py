import joblib
import os

class ModelManager:
    def __init__(self, model_dir="models", default_model_file="model_v1.pkl"):
        self.model_dir = model_dir
        self.model_version = None
        self.model = None
        self.load_model(default_model_file)

    def load_model(self, filename):
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {filename} not found in {self.model_dir}")
        self.model_version = filename 
        self.model = joblib.load(model_path)

    def predict(self, features):
        return self.model.predict(features)